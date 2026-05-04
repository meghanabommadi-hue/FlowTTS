#!/usr/bin/env bash
# Update nodes in dashboard + prometheus, then send GChat alert
set -e

MONITOR_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IPS_FILE="${1:-$MONITOR_DIR/data/ips.txt}"
PYTHON="/Users/meghana.bommadi/.pyenv/versions/3.8.18/bin/python3"

echo "==> Updating dashboard nodes..."
bash "$MONITOR_DIR/shell/update_nodes.sh" "$IPS_FILE"

echo "==> Updating prometheus targets..."
bash "$MONITOR_DIR/shell/update_ips.sh" "$IPS_FILE"

echo "==> Sending GChat alert..."
"$PYTHON" "$MONITOR_DIR/scripts/call_status_cron.py"
