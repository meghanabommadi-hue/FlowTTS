#!/usr/bin/env bash
# One-command setup for FlowTTS's miotts backend (flowtts/synthesis/miotts.py,
# model_type=miotts): clones ~/miotts if missing, creates .venv_mio, installs
# everything into it, and pre-downloads both model checkpoints -- after this
# script, `./run.sh --model-type miotts` should start clean with no further
# manual steps.
#
# Unlike OmniVoice (two genuinely incompatible dependency sets, needing two
# separate venvs), miotts's split into .venv_vllm (Python 3.10) + .venv
# (Python 3.12) in its OWN checkout (~/miotts) is driven by miocodec's package
# metadata declaring `Requires-Python >=3.12` -- simply incompatible with
# .venv_vllm's 3.10 interpreter in that checkout, not a real conflict between
# the packages themselves. This was verified directly: vllm==0.8.5,
# transformers==4.51.3, miocodec, and torch all install and import together
# cleanly on Python 3.12, with no flashinfer pulled in and no version
# resolver downgrades. See README.md's "Model dependency matrix" section.
#
# So this script creates ONE venv, .venv_mio (Python 3.12), with everything
# miotts needs installed together. FlowTTS's MiottsSynthesizer still spawns
# two separate child processes from this one venv (vLLM server + codec
# server) -- that's for GPU-residency/restart isolation (matching miotts's
# own run.sh design), not a Python-version workaround.
#
# Usage:
#   ./setup_mio.sh                                    # clone/reuse ~/miotts, set up .venv_mio, download weights
#   MIOTTS_REPO_DIR=/path/to/miotts ./setup_mio.sh    # use an existing checkout elsewhere
#   MIOTTS_REPO_URL=git@github.com:you/miotts ./setup_mio.sh   # clone from a different remote
#   VENV_DIR_NAME=.venv_mio2 ./setup_mio.sh           # custom venv dir name
#   SKIP_MODEL_DOWNLOAD=1 ./setup_mio.sh              # venv only, skip weight pre-download
#
# After setup, run FlowTTS with:
#   ./run.sh --model-type miotts
#   FLOWTTS_MODEL_TYPE=miotts ./run.sh   # equivalent
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIOTTS_REPO_DIR="${MIOTTS_REPO_DIR:-${HOME}/miotts}"
MIOTTS_REPO_URL="${MIOTTS_REPO_URL:-https://github.com/meghanabommadi-hue/mio_test.git}"
VENV_DIR_NAME="${VENV_DIR_NAME:-.venv_mio}"
VENV_DIR="${SCRIPT_DIR}/${VENV_DIR_NAME}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"

# ── 1. miotts checkout (clone if missing) ──────────────────────────────────
if [ -d "${MIOTTS_REPO_DIR}" ]; then
    echo "Using existing miotts checkout at ${MIOTTS_REPO_DIR}"
else
    echo "No miotts checkout at ${MIOTTS_REPO_DIR} -- cloning from ${MIOTTS_REPO_URL}"
    git clone "${MIOTTS_REPO_URL}" "${MIOTTS_REPO_DIR}"
fi

if [ ! -f "${MIOTTS_REPO_DIR}/miotts/codec_server.py" ]; then
    echo "error: ${MIOTTS_REPO_DIR}/miotts/codec_server.py not found -- is this really miotts?" >&2
    exit 1
fi

if ! command -v python3.12 >/dev/null 2>&1; then
    echo "error: python3.12 not found on PATH -- miocodec requires Python >=3.12" >&2
    exit 1
fi

# ── 2. .venv_mio (Python 3.12) ─────────────────────────────────────────────
if [ -d "${VENV_DIR}" ]; then
    echo "${VENV_DIR} already exists -- leaving as-is (delete it first to rebuild from scratch)"
else
    echo "Creating miotts virtualenv at ${VENV_DIR} (Python 3.12)"
    python3.12 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

if [ -f "${SCRIPT_DIR}/requirements-miotts.txt" ]; then
    echo "Installing from requirements-miotts.txt (pinned, previously verified) ..."
    pip install -r "${SCRIPT_DIR}/requirements-miotts.txt"
else
    echo "requirements-miotts.txt not found -- installing from miotts's own pins instead"
    # vllm's pin from ~/miotts/requirements-vllm.txt (Python-version constraint
    # there was about the venv's own interpreter, not these package versions).
    pip install vllm==0.8.5 transformers==4.51.3
    # miocodec from git, per ~/miotts/requirements.txt.
    pip install "git+https://github.com/Aratako/MioCodec"
    # Serving-layer deps used by miotts.codec_server (FastAPI) + FlowTTS's
    # HTTP client side (aiohttp), not pulled in by the above.
    pip install fastapi uvicorn requests aiohttp structlog
fi

# ── 3. Pre-download model weights (SPRINGLab/Indic-Mio + MioCodec) ────────
# from_pretrained() would fetch these lazily on first request anyway, but
# doing it here means the first real request after ./run.sh isn't stalled by
# a cold multi-GB download -- and it surfaces auth/network failures now,
# with a clear error, instead of inside a spawned child process's log file.
if [ "${SKIP_MODEL_DOWNLOAD}" = "1" ]; then
    echo "SKIP_MODEL_DOWNLOAD=1 -- skipping weight pre-download"
else
    echo "Pre-downloading model weights (SPRINGLab/Indic-Mio, Aratako/MioCodec-25Hz-44.1kHz-v2) ..."
    python3 - <<'PYEOF'
from huggingface_hub import snapshot_download

for repo_id in ("SPRINGLab/Indic-Mio", "Aratako/MioCodec-25Hz-44.1kHz-v2"):
    print(f"  downloading {repo_id} ...")
    snapshot_download(repo_id=repo_id)
    print(f"  {repo_id} ready (cached in ~/.cache/huggingface/hub)")
PYEOF
fi

deactivate
echo
echo "miotts venv ready: ${VENV_DIR}"
echo
echo "FlowTTS defaults to this layout (flowtts/core/config.py: MiottsSettings):"
echo "  repo_dir    = ${HOME}/miotts"
echo "  venv_python = ${SCRIPT_DIR}/.venv_mio/bin/python"
echo
echo "If you used a different location or venv name, point FlowTTS at it with:"
echo "  export FLOWTTS_MIOTTS__REPO_DIR=${MIOTTS_REPO_DIR}"
echo "  export FLOWTTS_MIOTTS__VENV_PYTHON=${VENV_DIR}/bin/python"
echo
echo "Then run FlowTTS with the miotts backend, e.g.:"
echo "  ./run.sh --model-type miotts"
