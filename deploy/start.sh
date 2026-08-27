#!/usr/bin/env bash
# Supervised launcher for the FlowTTS / OmniVoice service.
#
# Restarts on a non-zero exit: the two-strikes CUDA-OOM path in the service
# exits 1 deliberately so a clean process is started rather than a poisoned one
# limping along.
#
# Writes its own PID to omnivoice.pid so stop.sh can target it exactly. Matching
# a supervisor by command line does not work: this is launched as `./start.sh`,
# so a pattern built from the absolute path never matches, and a bare "start.sh"
# pattern would also catch the other services on this box. Without a reliable
# handle on the supervisor, every stop killed only the child and left the loop
# alive to restart it — fifteen of them accumulated that way.
set -u
cd /root/omnivoice-svc

PIDFILE=/root/omnivoice-svc/omnivoice.pid
LOG=/root/omnivoice-svc/omnivoice.log

# Refuse to start a second supervisor rather than silently stacking them.
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    echo "[$(date -Is)] already running as PID $(cat "$PIDFILE"); run ./stop.sh first" >&2
    exit 1
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

set -a; . /root/omnivoice-svc/omnivoice.env; set +a

VENV=${VENV:-/home/jovyan/FlowTTS/omnivoice/.venv}
PROFILE=${PROFILE:-balanced}

echo "[$(date -Is)] starting omnivoice (profile=$PROFILE, supervisor=$$)" >> "$LOG"
while true; do
    "$VENV/bin/python" -m flowtts.service \
        --profile "$PROFILE" \
        --http-port "${FLOWTTS_SERVER__HTTP_PORT:-9000}" \
        --ctrl-port "${FLOWTTS_SERVER__CTRL_PORT:-9764}" \
        --base-port "${FLOWTTS_SERVER__WS_BASE_PORT:-9080}" \
        --ws-ports  "${FLOWTTS_SERVER__WS_PORTS:-2}" \
        >> "$LOG" 2>&1 &
    child=$!
    # Record the child too, so stop.sh can end it without a pattern match.
    echo "$child" > /root/omnivoice-svc/omnivoice.child.pid
    wait "$child"
    code=$?
    echo "[$(date -Is)] omnivoice exited code=$code" >> "$LOG"
    [ $code -eq 0 ] && break
    sleep 10
done
