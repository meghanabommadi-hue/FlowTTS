#!/usr/bin/env bash
# Chaashini GPU-side environment bootstrap. Additive only: everything lives under /opt/chaashini.
set -uo pipefail
export PATH=/usr/local/bin:$HOME/.local/bin:$PATH
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /opt/chaashini
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
log 'uv: '$(uv --version)
uv python install 3.11 || true
# ---- venv-asr: diarization (NeMo) + ASR (transformers custom code) + service deps
[ -d venv-asr ] || uv venv --python 3.11 venv-asr
uv pip install --python venv-asr/bin/python --index-url https://download.pytorch.org/whl/cu128 torch torchaudio 2>&1 | tail -3
uv pip install --python venv-asr/bin/python 'nemo_toolkit[asr]' 2>&1 | tail -3
uv pip install --python venv-asr/bin/python 'transformers' sentencepiece soundfile fastapi 'uvicorn[standard]' python-multipart numpy 'huggingface_hub[hf_transfer]' hf_transfer cuda-python 2>&1 | tail -3
venv-asr/bin/python -c 'import torch, transformers, nemo; print("asr venv ok torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__, "nemo", nemo.__version__)' 2>&1 | tail -2
# ---- venv-enhance: resemble-enhance with its own (old) torch pins
[ -d venv-enhance ] || uv venv --python 3.11 venv-enhance
uv pip install --python venv-enhance/bin/python torch==2.1.1 torchaudio==2.1.1 torchvision==0.16.1 2>&1 | tail -2
uv pip install --python venv-enhance/bin/python --no-build-isolation deepspeed==0.12.4 2>&1 | tail -2
uv pip install --python venv-enhance/bin/python resemble-enhance fastapi 'uvicorn[standard]' python-multipart 2>&1 | tail -3
venv-enhance/bin/python -c 'import torch; from resemble_enhance.enhancer.inference import denoise, enhance; print("enhance venv ok torch", torch.__version__, "cuda", torch.cuda.is_available())' 2>&1 | tail -2
# ---- models
# ---- models (the ASR repo is gated: /opt/chaashini/.hf_token_models must hold a token with access)
TOK=$(cat /opt/chaashini/.hf_token_models)
venv-asr/bin/python - <<PYEOF
from huggingface_hub import snapshot_download, hf_hub_download
print(snapshot_download("ARTPARK-IISc/SraVaani-0.5-live", token="$TOK", local_dir="/opt/chaashini/models/sravaani-0.5-live"))
print(hf_hub_download("nvidia/diar_streaming_sortformer_4spk-v2.1", "diar_streaming_sortformer_4spk-v2.1.nemo", local_dir="/opt/chaashini/models/sortformer"))
print(snapshot_download("ResembleAI/resemble-enhance", local_dir="/opt/chaashini/models/resemble-enhance"))
PYEOF
ls -la models/sravaani-0.5-live models/sortformer models/resemble-enhance
log SETUP_DONE
