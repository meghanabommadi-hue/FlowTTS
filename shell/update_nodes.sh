#!/bin/bash
# Usage: ./update_nodes.sh [ips_file]
# Default: ips.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IPS_FILE="${1:-$SCRIPT_DIR/../data/ips.txt}"
DASHBOARD="$SCRIPT_DIR/../html/dashboard.html"

if [ ! -f "$IPS_FILE" ]; then
  echo "Error: $IPS_FILE not found"
  exit 1
fi

# Build the NODES array string from the IPs file (skip comments and blank lines)
NODES="const NODES = [\n"
while IFS= read -r line; do
  # stop before any secondary cluster section
  [[ "$line" == "# cluster:ext2" || "$line" == "# cluster:multilingual" ]] && break
  line="${line%%#*}"   # strip inline comments
  line="${line// /}"   # strip spaces
  [ -z "$line" ] && continue
  if [[ "$line" == *:* ]]; then
    NODES+="  \"${line}\",\n"
  else
    NODES+="  \"${line}:8764\",\n"
  fi
done < "$IPS_FILE"
NODES+="];"

# Replace the NODES block in dashboard.html
python3 - "$DASHBOARD" "$NODES" <<'EOF'
import sys, re

dashboard = sys.argv[1]
new_nodes = sys.argv[2]

with open(dashboard, "r", encoding="utf-8") as f:
    content = f.read()

# Replace from 'const NODES = [' to '];'
updated = re.sub(
    r'const NODES = \[.*?\];',
    new_nodes,
    content,
    flags=re.DOTALL
)

with open(dashboard, "w", encoding="utf-8") as f:
    f.write(updated)

print(f"Updated {dashboard} with nodes from {sys.argv[1]}")
EOF

echo "Done. IPs loaded from: $IPS_FILE"
