#!/usr/bin/env bash
# Run bench configs SEQUENTIALLY. They share one GPU, so running them in
# parallel would both slow everything down and make the timing metrics
# (rtfx/ttfb) meaningless.
set -uo pipefail
cd /home/jovyan/tts-bench
export HF_TOKEN=$(cat /home/jovyan/omnivoice-train/token.read)
export PYTHONPATH=/home/jovyan/omnivoice-train/OmniVoice-src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
mkdir -p /home/jovyan/bench_logs
PY=/home/jovyan/omnivoice-train/.venv/bin/python
for CFG in "$@"; do
  TAG=$(basename "$CFG" .yaml)
  echo "[$(date -u +%H:%M:%S)] === $TAG ===" | tee -a /home/jovyan/bench_logs/driver.log
  "$PY" -m tts_bench.cli bench --config "$CFG" > "/home/jovyan/bench_logs/$TAG.log" 2>&1
  rc=$?
  ok=$(tr '\r' '\n' < "/home/jovyan/bench_logs/$TAG.log" | grep -oE '[0-9]+ ok, [0-9]+ skipped, [0-9]+ failed' | tail -1)
  echo "[$(date -u +%H:%M:%S)] $TAG rc=$rc | $ok" | tee -a /home/jovyan/bench_logs/driver.log
done
echo "[$(date -u +%H:%M:%S)] ALL DONE" | tee -a /home/jovyan/bench_logs/driver.log
