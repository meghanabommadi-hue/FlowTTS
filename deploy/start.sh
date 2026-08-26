#!/usr/bin/env bash
# Supervised launcher for the FlowTTS / OmniVoice service.
#
# Restarts on a non-zero exit: the two-strikes CUDA-OOM path in
# flowtts/api/service.py exits 1 deliberately so a clean process is started
# rather than a poisoned one limping along.
set -u
cd /root/omnivoice-svc

set -a; . /root/omnivoice-svc/omnivoice.env; set +a

VENV=${VENV:-/home/jovyan/FlowTTS/omnivoice/.venv}
PROFILE=${PROFILE:-balanced}
LOG=/root/omnivoice-svc/omnivoice.log

echo "[$(date -Is)] starting omnivoice (profile=$PROFILE)" >> "$LOG"
while true; do
    "$VENV/bin/python" -m flowtts.service \
        --profile "$PROFILE" \
        --http-port "${FLOWTTS_SERVER__HTTP_PORT:-9000}" \
        --ctrl-port "${FLOWTTS_SERVER__CTRL_PORT:-9764}" \
        --base-port "${FLOWTTS_SERVER__WS_BASE_PORT:-9080}" \
        --ws-ports  "${FLOWTTS_SERVER__WS_PORTS:-2}" \
        >> "$LOG" 2>&1
    code=$?
    echo "[$(date -Is)] omnivoice exited code=$code" >> "$LOG"
    [ $code -eq 0 ] && break
    sleep 10
done
