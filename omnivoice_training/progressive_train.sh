#!/usr/bin/env bash
# Progressive fine-tuning: train on chunks as the producer creates them.
#
# OmniVoice's WebDatasetReader reads data.lst once at epoch start, so new shards
# are only picked up on a restart. Each cycle therefore waits for +CYCLE_HOURS
# of fresh data, then restarts from the previous checkpoint with the grown
# manifest. Continues from the ohun fine-tune and pushes to OmniNaija, so the
# previous model is never overwritten.
#
# Trains on the NEW chunks only (run_prog), and evaluates on the held-out dev
# set from the 239h run so the quality curve is comparable across cycles.
set -uo pipefail

BASE=${BASE:-/home/jovyan/omnivoice-train}
PROG=$BASE/run_prog
PKG=$BASE/omnivoice_training
SRC=$BASE/OmniVoice-src
PY=$BASE/.venv/bin/python
ACC=$BASE/.venv/bin/accelerate
DEV_LST=${DEV_LST:-$BASE/run/tokens/dev/data.lst}
EVAL_SET=${EVAL_SET:-$BASE/run/eval_set.json}

CYCLE_HOURS=${CYCLE_HOURS:-10}
STEPS_PER_CYCLE=${STEPS_PER_CYCLE:-2000}
# Training must leave room for the producer (~14GB while aligning), so this is
# deliberately well under what the GPU could take on its own.
BATCH_TOKENS=${BATCH_TOKENS:-3072}
MAX_CYCLES=${MAX_CYCLES:-100}
SEED_MODEL=${SEED_MODEL:-kapturecx/ohun-omnivoice}

export PYTHONPATH="$SRC:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat $BASE/token.read 2>/dev/null)
export OHUN_HF_TOKEN=$(cat $BASE/token.write 2>/dev/null)
export OHUN_HF_REPO=${OHUN_HF_REPO:-kapturecx/OmniNaija}
export OHUN_HF_MIN_DELTA=${OHUN_HF_MIN_DELTA:-0.002}
export OHUN_HF_MIN_STEP=${OHUN_HF_MIN_STEP:-200}
export OHUN_EVAL_SET=$EVAL_SET
export OHUN_EVAL_WAV_DIR=$PROG/eval_wav
export OHUN_EVAL_INFER_EVERY=${OHUN_EVAL_INFER_EVERY:-1}
export OHUN_INFER_TIMEOUT=${OHUN_INFER_TIMEOUT:-1200}
export OHUN_ASR_URL=${OHUN_ASR_URL:-http://127.0.0.1:8899}
export OHUN_ASR_MODEL=${OHUN_ASR_MODEL:-Axiveri/NaijaVox-2.0}

mkdir -p "$PROG"/{logs,exp,eval_wav}
plog(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$PROG/logs/progressive.log"; }

produced_hours(){
  $PY - "$PROG/producer_state.json" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    print(f"{sum(json.load(open(sys.argv[1]))['hours'].values()):.4f}")
except Exception:
    print("0")
PY
}

manifest_hours(){
  local f="$PROG/tokens/train/data.lst"
  [ -s "$f" ] || { echo 0; return; }
  awk '{t+=$4} END{printf "%.4f", t/3600}' "$f"
}

latest_ckpt(){
  ls -d "$PROG"/exp/checkpoint-* 2>/dev/null \
    | sed 's/.*checkpoint-//' | sort -n | tail -1
}

plog "=== progressive trainer: +${CYCLE_HOURS}h per cycle, ${STEPS_PER_CYCLE} steps, batch_tokens=${BATCH_TOKENS} ==="
plog "    seed=$SEED_MODEL  push->$OHUN_HF_REPO  dev=$DEV_LST"

trained_at=0
cycle=0
while [ "$cycle" -lt "$MAX_CYCLES" ]; do
  # wait for enough NEW data
  while :; do
    have=$(manifest_hours)
    need=$(awk "BEGIN{print $trained_at + $CYCLE_HOURS}")
    if awk "BEGIN{exit !($have >= $need)}"; then break; fi
    plog "waiting for data: ${have}h produced, need ${need}h (cycle $((cycle+1)))"
    sleep 300
  done

  cycle=$((cycle+1))
  have=$(manifest_hours)
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  LOG="$PROG/logs/train-c${cycle}-$ts.log"
  plog "=== cycle $cycle: training on ${have}h of new chunks -> $LOG"

  cat > "$PROG/data_config.json" <<JSON
{
  "train": [{"manifest_path": ["$PROG/tokens/train/data.lst"]}],
  "dev":   [{"manifest_path": ["$DEV_LST"]}]
}
JSON

  ck=$(latest_ckpt)
  $PY - "$PKG/configs/train_a100_40gb.json" "$PROG/train_config.json" \
       "$BATCH_TOKENS" "$STEPS_PER_CYCLE" "$SEED_MODEL" "${ck:-}" "$PROG/exp" <<'PY'
import json, sys
src, dst, bt, steps, seed, ck, expdir = sys.argv[1:8]
c = json.load(open(src))
c["batch_tokens"] = int(bt)
c["num_workers"] = 2                 # leave CPU for the producer
c["eval_steps"] = 250                # quality curve resolution
c["save_steps"] = 250
c["logging_steps"] = 25
c["keep_last_n_checkpoints"] = 3
if ck:
    # resume keeps the optimizer state AND the step count (parsed from the dir
    # name), so the LR schedule continues instead of re-warming every cycle
    c["resume_from_checkpoint"] = f"{expdir}/checkpoint-{ck}"
    c["init_from_checkpoint"] = None
    c["steps"] = int(ck) + int(steps)
else:
    c["resume_from_checkpoint"] = None
    c["init_from_checkpoint"] = seed
    c["steps"] = int(steps)
json.dump(c, open(dst, "w"), indent=1)
print(f"cycle config: steps={c['steps']} resume={c['resume_from_checkpoint']} "
      f"init={c['init_from_checkpoint']} batch_tokens={c['batch_tokens']}")
PY

  "$ACC" launch --num_processes 1 --mixed_precision bf16 \
      "$PKG/train_ohun.py" --train_config "$PROG/train_config.json" \
      --data_config "$PROG/data_config.json" --output_dir "$PROG/exp" \
      >> "$LOG" 2>&1
  rc=$?
  plog "cycle $cycle exited rc=$rc (checkpoint $(latest_ckpt))"

  if [ "$rc" -ne 0 ]; then
    if grep -qE "CUDA out of memory|OutOfMemoryError" "$LOG"; then
      BATCH_TOKENS=$(( BATCH_TOKENS * 3 / 4 ))
      [ "$BATCH_TOKENS" -lt 1024 ] && BATCH_TOKENS=1024
      plog "  OOM -> batch_tokens reduced to $BATCH_TOKENS"
    else
      plog "  non-OOM failure; tail:"; tail -6 "$LOG" | sed 's/^/    /' \
        | tee -a "$PROG/logs/progressive.log"
    fi
    sleep 60
    continue          # retry the same cycle rather than skipping data
  fi

  m="$PROG/exp/last_eval_metrics.json"
  [ -f "$m" ] && plog "  metrics: $(tr -d '\n' < "$m")"
  trained_at=$have
done
plog "=== reached MAX_CYCLES ==="
