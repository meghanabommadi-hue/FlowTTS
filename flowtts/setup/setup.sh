#!/usr/bin/env bash
# One-time setup for the FlowTTS OmniVoice server (run on the H200 box).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "==> Install PyTorch (CUDA build for your box) BEFORE this if not already present, e.g.:"
echo "    pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128"

echo "==> Installing requirements.txt"
pip install -r "$REPO_ROOT/requirements.txt"

echo "==> Downloading OmniVoice weights into the HF cache"
python3 -m flowtts.setup.download_models

echo "==> Building voice-clone npz artifacts from sample_files/ (+ optional voices/manifest.json)"
python3 -m flowtts.voices.clone --build-all --manifest "$REPO_ROOT/voices/manifest.json" || \
    python3 -m flowtts.voices.clone --build-all

echo "==> Installed voices:"
python3 -m flowtts.voices.clone --list || true

echo "==> Setup complete. Start with:  bash run.sh --ctrl-port 8764 --ports 1"
