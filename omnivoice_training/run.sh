#!/usr/bin/env bash
# Launch the whole unattended pipeline detached, guarded by a pidfile.
set -uo pipefail
BASE=/home/jovyan/omnivoice-train
RUN=${RUN:-$BASE/run}
PKG=$BASE/omnivoice_training
mkdir -p "$RUN/logs"

if [ -f "$RUN/pipeline.pid" ] && kill -0 "$(cat "$RUN/pipeline.pid")" 2>/dev/null; then
  echo "pipeline already running as pid $(cat "$RUN/pipeline.pid")"; exit 1
fi
[ -s "$BASE/token.write" ] || { echo "missing $BASE/token.write (HF write token)"; exit 1; }

"$PKG/serve.sh" || echo "WARN: tensorboard/nginx setup had a problem; continuing"

ts=$(date -u +%Y%m%dT%H%M%SZ)
setsid "$PKG/pipeline.sh" >> "$RUN/logs/pipeline-$ts.log" 2>&1 < /dev/null &
echo $! > "$RUN/pipeline.pid"
sleep 3
echo "pipeline pid $(cat "$RUN/pipeline.pid") -> $RUN/logs/pipeline-$ts.log"
echo "tensorboard: http://101.53.139.167/tb/"
