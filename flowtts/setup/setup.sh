#!/usr/bin/env bash
# One-time setup for the FlowTTS gateway (CPU-only). The GPU model (Fish S2 Pro) runs
# in the SEPARATE sglang-omni backend — nothing GPU/torch is installed here.
#
# Uses uv (fast, compact). Install it once if missing:  https://docs.astral.sh/uv/
#   curl -LsSf https://astral.sh/uv/install.sh | sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv (not found on PATH)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Creating venv (.venv) with uv"
uv venv "$REPO_ROOT/.venv" -p 3.12
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

echo "==> Installing requirements.txt with uv"
uv pip install -r "$REPO_ROOT/requirements.txt"

echo "==> Building voice references from voices/manifest.json (needs ref_text; no GPU)"
python3 -m flowtts.voices.clone --build-all --manifest "$REPO_ROOT/voices/manifest.json" || \
    echo "    (no voices built — add ref_text entries to voices/manifest.json)"

echo "==> Installed voices:"
python3 -m flowtts.voices.clone --list || true

echo "==> Setup complete. Start the GPU backend (sgl-omni serve …), then:"
echo "    bash run.sh --ctrl-port 8764 --ports 1 --backend-url http://127.0.0.1:8000"
