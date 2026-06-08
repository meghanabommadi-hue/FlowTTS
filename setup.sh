#!/bin/bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Creating uv venv..."
uv venv "$ROOT/.venv"

echo "==> Installing requirements..."
uv pip install -r "$ROOT/requirements.txt" --python "$ROOT/.venv/bin/python"

echo "==> Updating nodes..."
bash "$ROOT/shell/update_nodes.sh"

echo "==> Starting server..."
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/server.py"
