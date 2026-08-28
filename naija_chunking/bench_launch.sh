#!/usr/bin/env bash
# Detach INSIDE the script: an inline `setsid ... &` over ssh dies with the
# session, which cost several restarts.
set -uo pipefail
LIST="/home/jovyan/bench_logs/queue.txt"
mkdir -p /home/jovyan/bench_logs
if pgrep -f run_bench_seq.sh >/dev/null; then
  echo "bench driver already running"; exit 1
fi
printf '%s\n' "$@" > "$LIST"
setsid nohup /home/jovyan/run_bench_seq.sh "$@" \
  >> /home/jovyan/bench_logs/driver_stdout.log 2>&1 < /dev/null &
disown || true
sleep 3
echo "launched $# configs; driver pid $(pgrep -f run_bench_seq.sh | head -1)"
