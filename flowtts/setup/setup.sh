#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "==> Installing 2requirements.txt"
uv pip install -r "$REPO_ROOT/2requirements.txt"
echo "==> Installing 1requirements.txt"
uv pip install -r "$REPO_ROOT/1requirements.txt"


echo "==> Installing FlashSR from git"
uv pip install git+https://github.com/ysharma3501/FlashSR.git@2a69326250613c0a0f6c1c8d9f0c48cb779842b8

echo "==> Installing FastBiCodec from git"
uv pip install git+https://github.com/ysharma3501/FastBiCodec.git@612ba9e29d14b9752dc3174616a6cb5bafe5af15

export HF_TOKEN="${HF_TOKEN:-}"
echo "==> Downloading models"
python3 "$SCRIPT_DIR/download_models.py"

echo "==> Setup complete."
