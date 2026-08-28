#!/usr/bin/env bash
# Run one tts-bench config detached, logging to a predictable path.
set -uo pipefail
CFG=${1:?config path}
TAG=$(basename "$CFG" .yaml)
cd /home/jovyan/tts-bench
export HF_TOKEN=$(cat /home/jovyan/omnivoice-train/token.read)
export PYTHONPATH=/home/jovyan/omnivoice-train/OmniVoice-src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
mkdir -p /home/jovyan/bench_logs
setsid /home/jovyan/omnivoice-train/.venv/bin/python -m tts_bench.cli bench --config "$CFG" \
  > "/home/jovyan/bench_logs/$TAG.log" 2>&1 < /dev/null &
echo "started $TAG -> /home/jovyan/bench_logs/$TAG.log"
