#!/usr/bin/env bash
# End-to-end unattended pipeline: wait for data -> prepare -> tokenize -> probe -> train.
#
# Every stage writes a marker under $RUN/stage/ so a restart resumes instead of
# redoing hours of work. Stages are deliberately separate processes: the audio
# tokeniser and the trainer both want the whole GPU, and neither should be able
# to take the other down.
set -uo pipefail

RUN=${RUN:-/opt/omnivoice-train/run}
SRC=${SRC:-/opt/OmniVoice-src}
PKG=${PKG:-/opt/omnivoice-train/omnivoice_training}
PY=${PY:-/opt/omnivoice-train/.venv/bin/python}
ACC=${ACC:-/opt/omnivoice-train/.venv/bin/accelerate}

HOURS_PER_LANG=${HOURS_PER_LANG:-25}
DEV_MINUTES=${DEV_MINUTES:-12}
MAX_SEC=${MAX_SEC:-25}
MIN_SEC=${MIN_SEC:-1.5}
OHUN_STATUS_URL=${OHUN_STATUS_URL:-http://205.147.102.70/ohun/status.json}
WAIT_FOR_DATA=${WAIT_FOR_DATA:-1}
DATA_REPO=${DATA_REPO:-kapturecx/ohun}

mkdir -p "$RUN"/{stage,logs,prep,tokens,exp,eval_wav}
export PYTHONPATH="$SRC:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RUN/logs/pipeline.log"; }
done_stage() { [ -f "$RUN/stage/$1.done" ]; }
mark()  { touch "$RUN/stage/$1.done"; }

# ---------------------------------------------------------------- 0: wait
if [ "$WAIT_FOR_DATA" = "1" ] && ! done_stage wait; then
  log "stage 0: waiting for the ohun upload to finish ($OHUN_STATUS_URL)"
  while true; do
    js=$(curl -s --max-time 20 "$OHUN_STATUS_URL" || echo '{}')
    phase=$(echo "$js" | $PY -c 'import json,sys;print(json.load(sys.stdin).get("phase","?"))' 2>/dev/null || echo '?')
    dn=$(echo "$js" | $PY -c 'import json,sys;d=json.load(sys.stdin);print(d.get("done",0))' 2>/dev/null || echo 0)
    tt=$(echo "$js" | $PY -c 'import json,sys;d=json.load(sys.stdin);print(d.get("total",1))' 2>/dev/null || echo 1)
    if [ "$phase" = "done" ]; then log "  data upload COMPLETE ($dn/$tt)"; break; fi
    log "  waiting: phase=$phase $dn/$tt"
    sleep 120
  done
  mark wait
fi

# ---------------------------------------------------------------- 1: prepare
if ! done_stage prepare; then
  log "stage 1: preparing ${HOURS_PER_LANG}h/language from $DATA_REPO"
  $PY "$PKG/ohun_prepare.py" --repo "$DATA_REPO" --out "$RUN/prep" \
      --hours-per-lang "$HOURS_PER_LANG" --dev-minutes-per-lang "$DEV_MINUTES" \
      --min-sec "$MIN_SEC" --max-sec "$MAX_SEC" \
      --status "$RUN/ui/prepare_status.json" \
      >> "$RUN/logs/prepare.log" 2>&1 || { log "prepare FAILED"; exit 1; }
  mark prepare
  log "  prepare done: $(du -sh "$RUN/prep" | cut -f1)"
fi

# ---------------------------------------------------------------- 2: eval set
if ! done_stage evalset; then
  log "stage 2: building fixed eval prompt set"
  $PY "$PKG/make_eval_set.py" --prep-dir "$RUN/prep" \
      --out "$RUN/eval_set.json" --per-lang 3 \
      >> "$RUN/logs/evalset.log" 2>&1 || { log "eval set FAILED"; exit 1; }
  mark evalset
fi

# ---------------------------------------------------------------- 3: tokenize
for split in train dev; do
  if done_stage "tokens_$split"; then continue; fi
  log "stage 3: tokenizing $split (GPU)"
  cat "$RUN"/prep/*_"$split".jsonl > "$RUN/prep/all_$split.jsonl"
  n=$(wc -l < "$RUN/prep/all_$split.jsonl")
  log "  $n utterances -> webdataset shards"
  mkdir -p "$RUN/tokens/$split"/{audios,txts}
  $PY -m omnivoice.scripts.extract_audio_tokens \
      --input_jsonl "$RUN/prep/all_$split.jsonl" \
      --tar_output_pattern "$RUN/tokens/$split/audios/shard-%06d.tar" \
      --jsonl_output_pattern "$RUN/tokens/$split/txts/shard-%06d.jsonl" \
      --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
      --min_length "$MIN_SEC" --max_length "$MAX_SEC" \
      --nj_per_gpu "${NJ_PER_GPU:-4}" --loader_workers "${LOADER_WORKERS:-32}" \
      --skip_errors --shuffle True \
      >> "$RUN/logs/tokenize_$split.log" 2>&1 || { log "tokenize $split FAILED"; exit 1; }
  lst="$RUN/tokens/$split/data.lst"
  [ -s "$lst" ] || { log "tokenize $split produced no data.lst"; exit 1; }
  log "  $split: $(wc -l < "$lst") shards, $(awk '{s+=$4} END{printf \"%.1f\", s/3600}' "$lst")h"
  mark "tokens_$split"
done

# ---------------------------------------------------------------- 4: data config
cat > "$RUN/data_config.json" <<JSON
{
  "train": [{"manifest_path": ["$RUN/tokens/train/data.lst"]}],
  "dev":   [{"manifest_path": ["$RUN/tokens/dev/data.lst"]}]
}
JSON
log "stage 4: data config written"

# ---------------------------------------------------------------- 5: vram probe
if ! done_stage probe; then
  log "stage 5: probing max batch_tokens on this GPU"
  $PY "$PKG/vram_probe.py" --train-config "$PKG/configs/train_a100_40gb.json" \
      --data-config "$RUN/data_config.json" --out "$RUN/batch_tokens.json" \
      --workdir "$RUN/probe" \
      >> "$RUN/logs/probe.log" 2>&1 || log "  probe failed; will use config default"
  mark probe
fi
BT=$($PY -c "import json;print(json.load(open('$RUN/batch_tokens.json'))['batch_tokens'])" 2>/dev/null || echo "")
log "stage 5: batch_tokens=${BT:-<config default>}"

# ---------------------------------------------------------------- 6: train
log "stage 6: handing off to the training supervisor"
exec "$PKG/supervise_train.sh" "$RUN" "${BT:-}"
