#!/usr/bin/env bash
# End-to-end unattended pipeline: wait for data -> prepare -> tokenize -> probe -> train.
#
# Every stage writes a marker under $RUN/stage/ so a restart resumes instead of
# redoing hours of work. Stages are deliberately separate processes: the audio
# tokeniser and the trainer both want the whole GPU, and neither should be able
# to take the other down.
set -uo pipefail

BASE_DIR=${BASE_DIR:-/home/jovyan/omnivoice-train}
RUN=${RUN:-$BASE_DIR/run}
SRC=${SRC:-$BASE_DIR/OmniVoice-src}
PKG=${PKG:-$BASE_DIR/omnivoice_training}
PY=${PY:-$BASE_DIR/.venv/bin/python}
ACC=${ACC:-$BASE_DIR/.venv/bin/accelerate}

HOURS_PER_LANG=${HOURS_PER_LANG:-90}
declare -A RW=( [ibo]=32 [yor]=14 [hau]=14 [pcm]=14 )
DEV_MINUTES=${DEV_MINUTES:-12}
MAX_SEC=${MAX_SEC:-25}
MIN_SEC=${MIN_SEC:-1.5}
OHUN_STATUS_URL=${OHUN_STATUS_URL:-http://205.147.102.70/ohun/status.json}
WAIT_FOR_DATA=${WAIT_FOR_DATA:-1}
DATA_REPO=${DATA_REPO:-kapturecx/ohun}
# FROM_SOURCES=1 reads the upstream Africanvoice corpora (already complete and
# byte-identical to what ohun repackages) so the GPU need not wait on the merge.
FROM_SOURCES=${FROM_SOURCES:-0}
SRC_FLAG=""; [ "$FROM_SOURCES" = "1" ] && SRC_FLAG="--from-sources"

mkdir -p "$RUN"/{stage,logs,prep,tokens,exp,eval_wav}
export PYTHONPATH="$SRC:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_TOKEN=${HF_TOKEN:-$(cat "$BASE_DIR/token.read" 2>/dev/null)}
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
# One process per language, in parallel. A single sequential reader is
# latency-bound (~1.8 MB/s of parquet); the languages are independent, so
# running them concurrently multiplies throughput on the same link.
if ! done_stage prepare; then
  log "stage 1: preparing ${HOURS_PER_LANG}h/language (4 languages in parallel)"
  pids=""; langs="ibo yor hau pcm"
  for lg in $langs; do
    $PY "$PKG/ohun_prepare.py" --repo "$DATA_REPO" --out "$RUN/prep" \
        --langs "$lg" --hours-per-lang "$HOURS_PER_LANG" \
        --dev-minutes-per-lang "$DEV_MINUTES" \
        --min-sec "$MIN_SEC" --max-sec "$MAX_SEC" \
        --status "$RUN/ui/prepare_$lg.json" $SRC_FLAG \
        --read-workers "${RW[$lg]:-12}" \
        >> "$RUN/logs/prepare_$lg.log" 2>&1 &
    pids="$pids $!"
    log "  [$lg] pid $!"
  done
  fail=0
  for p in $pids; do wait "$p" || fail=1; done
  if [ "$fail" = "1" ]; then
    log "at least one prepare worker failed; see $RUN/logs/prepare_*.log"
    # Continue anyway if we got usable data from the others - a missing
    # language is far better than no training run at all.
    n=$(cat "$RUN"/prep/*_train.jsonl 2>/dev/null | wc -l)
    [ "$n" -lt 500 ] && { log "only $n train utterances total - aborting"; exit 1; }
    log "  continuing with $n train utterances from the successful languages"
  fi
  mark prepare
  log "  prepare done: $(du -sh "$RUN/prep" | cut -f1), $(cat "$RUN"/prep/*_train.jsonl 2>/dev/null | wc -l) train utts"
fi

# ---------------------------------------------------------------- 1b: balance
if ! done_stage balance; then
  log "stage 1b: balancing languages to equal hours (before tokenising)"
  $PY "$PKG/balance_tokens.py" --prep-dir "$RUN/prep" \
      2>&1 | tee -a "$RUN/logs/balance.log" | sed 's/^/    /' | tee -a "$RUN/logs/pipeline.log"
  mark balance
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
