#!/bin/bash
# Scrapes active IPs from nginx dashboard, updates nodes, restarts server.
# Runs in a loop every hour.

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS_URL="http://34.126.91.56/dashboard/status.json"
IPS_FILE="$ROOT/data/ips.txt"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"

run_once() {
  echo "[$(date '+%H:%M:%S')] Fetching active IPs from $STATUS_URL..."

  NEW_IPS=$(curl -s --max-time 10 "$STATUS_URL" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for b in d['backends']:
    if b.get('up') and b.get('external_ip'):
        print(b['external_ip'])
")

  if [ -z "$NEW_IPS" ]; then
    echo "[$(date '+%H:%M:%S')] ERROR: no IPs fetched, skipping update"
    return
  fi

  COUNT=$(echo "$NEW_IPS" | wc -l | tr -d ' ')
  echo "[$(date '+%H:%M:%S')] Got $COUNT active IPs"

  {
    echo "# TTS GPU VMs — auto-updated $(date '+%Y-%m-%d %H:%M:%S')"
    echo "$NEW_IPS"
  } > "$IPS_FILE"

  echo "[$(date '+%H:%M:%S')] Updated $IPS_FILE"

  bash "$ROOT/shell/update_nodes.sh"

  echo "[$(date '+%H:%M:%S')] Restarting server..."
  lsof -ti :8080 | xargs kill -9 2>/dev/null || true
  sleep 1
  $PYTHON "$ROOT/scripts/server.py" &
  sleep 3

  if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/ | grep -q "200"; then
    echo "[$(date '+%H:%M:%S')] Server up at http://localhost:8080 with $COUNT nodes"
  else
    echo "[$(date '+%H:%M:%S')] ERROR: server failed to start"
  fi
}

run_once

echo "[$(date '+%H:%M:%S')] Scheduling refresh every 1 hour..."
while true; do
  sleep 3600
  run_once
done
