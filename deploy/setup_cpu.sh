#!/usr/bin/env bash
# Chaashini orchestrator environment bootstrap. Additive only: everything under /opt/chaashini (+ deno, libc++1).
set -uo pipefail
export PATH=/usr/local/bin:$HOME/.local/bin:$PATH
cd /opt/chaashini
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
log 'uv: '$(uv --version)
uv python install 3.11 || true
[ -d venv ] || uv venv --python 3.11 venv
uv pip install --python venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchaudio 2>&1 | tail -2
uv pip install --python venv/bin/python yt-dlp fastapi 'uvicorn[standard]' numpy soundfile scipy librosa onnxruntime ten-vad 'huggingface_hub[hf_xet]' pyarrow httpx pydantic pyyaml tqdm panns-inference psutil python-multipart datasets 2>&1 | tail -3
(apt-get install -y -q libc++1 >/dev/null 2>&1 && log 'libc++1 ok') || log 'libc++1 install failed (check manually)'
for f in sig_bak_ovr.onnx model_v8.onnx; do [ -s models/dnsmos/$f ] || curl -sL --max-time 120 https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/$f -o models/dnsmos/$f; done
ls -la models/dnsmos
venv/bin/python - <<'PY'
import numpy as np, onnxruntime as ort, torch
print('torch', torch.__version__)
from ten_vad import TenVad
v = TenVad(256, 0.5); print('tenvad ok', v.process(np.zeros(256, dtype=np.int16)))
s = ort.InferenceSession('/opt/chaashini/models/dnsmos/sig_bak_ovr.onnx', providers=['CPUExecutionProvider']); print('dnsmos ok', [i.name for i in s.get_inputs()])
PY
# PANNs CNN14 checkpoint (music/speech tagging); panns_inference downloads on first use, pre-trigger here
venv/bin/python -c "from panns_inference import AudioTagging; AudioTagging(checkpoint_path=None, device='cpu'); print('panns ok')" 2>&1 | tail -2
venv/bin/yt-dlp --version
log SETUP_DONE
