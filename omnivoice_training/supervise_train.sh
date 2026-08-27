#!/usr/bin/env bash
# Keep training alive overnight.
#
# Three distinct failure classes, three distinct responses:
#   * CUDA OOM            -> shrink batch_tokens 25% and restart
#   * flex_attention bug  -> fall back to the sdpa path and restart
#   * anything else       -> restart, resuming from the newest checkpoint
# Exit 0 only when the trainer itself exits 0.
set -uo pipefail

RUN=${1:?run dir}
BT_OVERRIDE=${2:-}
PKG=${PKG:-/opt/omnivoice-train/omnivoice_training}
SRC=${SRC:-/opt/OmniVoice-src}
PY=${PY:-/opt/omnivoice-train/.venv/bin/python}
ACC=${ACC:-/opt/omnivoice-train/.venv/bin/accelerate}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-60}
EXP="$RUN/exp"

export PYTHONPATH="$SRC:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export OHUN_EVAL_SET="$RUN/eval_set.json"
export OHUN_EVAL_WAV_DIR="$RUN/eval_wav"
export OHUN_EVAL_INFER_EVERY=${OHUN_EVAL_INFER_EVERY:-1}
export OHUN_INFER_TIMEOUT=${OHUN_INFER_TIMEOUT:-1200}
export OHUN_HF_REPO=${OHUN_HF_REPO:-kapturecx/ohun-omnivoice}
export OHUN_HF_MIN_DELTA=${OHUN_HF_MIN_DELTA:-0.002}
export OHUN_HF_MIN_STEP=${OHUN_HF_MIN_STEP:-500}
[ -f /opt/omnivoice-train/token.write ] && \
  export OHUN_HF_TOKEN=$(cat /opt/omnivoice-train/token.write)
export HF_TOKEN=${OHUN_HF_TOKEN:-}

mkdir -p "$EXP"
slog() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RUN/logs/train_supervisor.log"; }

CFG="$RUN/train_config.effective.json"
if [ ! -f "$CFG" ]; then
  $PY - "$PKG/configs/train_a100_40gb.json" "$CFG" "${BT_OVERRIDE:-}" <<'PY'
import json, sys
src, dst, bt = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(src))
if bt:
    c["batch_tokens"] = int(bt)
json.dump(c, open(dst, "w"), indent=1)
print(f"effective batch_tokens={c['batch_tokens']} attn={c['attn_implementation']}")
PY
fi

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt+1))
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  LOG="$RUN/logs/train-$ts.log"

  # resume from the newest checkpoint-N (the dir name carries the step, so it
  # must NOT be renamed - load_checkpoint parses the step out of the basename)
  latest=$(ls -d "$EXP"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  if [ -n "${latest:-}" ]; then
    $PY - "$CFG" "$EXP/checkpoint-$latest" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["resume_from_checkpoint"] = sys.argv[2]
json.dump(c, open(sys.argv[1], "w"), indent=1)
PY
    slog "attempt $attempt/$MAX_ATTEMPTS resuming from checkpoint-$latest -> $LOG"
  else
    slog "attempt $attempt/$MAX_ATTEMPTS starting fresh -> $LOG"
  fi

  "$ACC" launch --num_processes 1 --mixed_precision bf16 \
      "$PKG/train_ohun.py" --train_config "$CFG" \
      --data_config "$RUN/data_config.json" --output_dir "$EXP" \
      >> "$LOG" 2>&1
  rc=$?
  slog "attempt $attempt exited rc=$rc"
  [ "$rc" -eq 0 ] && { slog "TRAINING COMPLETE"; exit 0; }

  tail_log=$(tail -n 400 "$LOG")
  if grep -qE "CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" <<<"$tail_log"; then
    newbt=$($PY - "$CFG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["batch_tokens"] = max(2048, int(c["batch_tokens"] * 0.75) // 512 * 512)
json.dump(c, open(p, "w"), indent=1); print(c["batch_tokens"])
PY
)
    slog "  OOM detected -> batch_tokens reduced to $newbt"
    sleep 20; continue
  fi
  if grep -qiE "flex_attention|flexattention|torch\.compile" <<<"$tail_log" \
     && grep -qiE "error|not supported|failed" <<<"$tail_log"; then
    cur=$($PY -c "import json;print(json.load(open('$CFG'))['attn_implementation'])")
    if [ "$cur" = "flex_attention" ]; then
      $PY - "$CFG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["attn_implementation"] = "sdpa"     # padding path instead of packing
json.dump(c, open(p, "w"), indent=1)
PY
      slog "  flex_attention problem -> falling back to sdpa"
      sleep 10; continue
    fi
  fi
  slog "  unclassified failure; retrying in 60s. tail:"
  tail -n 12 "$LOG" | sed 's/^/    /' | tee -a "$RUN/logs/train_supervisor.log"
  sleep 60
done
slog "gave up after $MAX_ATTEMPTS attempts"
exit 1
