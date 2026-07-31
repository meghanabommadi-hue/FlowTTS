#!/usr/bin/env bash
# Sets up the OmniVoice backend used by FlowTTS's OmniVoiceSynthesizer
# (flowtts/synthesis/omnivoice.py, model_type=omnivoice).
#
# The OmniVoice server code (model.py, microbatch_server.py, batched_decode.py)
# is vendored in-tree at FlowTTS/omnivoice/ — this script has NO dependency on
# a separate omnivoice_scaled checkout.
#
# OmniVoice's own dependencies (transformers>=5.3.0) conflict with the
# transformers==4.57.3 pin FlowTTS/sglang need for the Mira path, so
# OmniVoice is NOT installed into FlowTTS's own venv. Instead this script
# creates a SEPARATE venv inside FlowTTS/omnivoice/; FlowTTS spawns that
# venv's python as a child process (omnivoice/model.py) and talks to it
# over loopback HTTP — see flowtts/synthesis/omnivoice.py.
#
# It also makes sure FlowTTS itself has *some* venv to run server.py from.
# server.py never imports sglang/torch-tensorrt/nanovllm_voxcpm at module
# load time in omnivoice mode (Mira/VoxCPM only import those lazily inside
# their own initialize()), so if ~/FlowTTS/.venv doesn't exist yet, this
# script creates a minimal one with just what server.py needs to run. This
# is purely additive: if flowtts/setup/setup.sh has already been run (or is
# run afterwards), its venv/deps are left untouched — that full setup still
# happens exactly as before for Mira/VoxCPM, this script just doesn't force
# you to run it first only to try OmniVoice.
#
# Usage:
#   ./setup_omni.sh
#   OMNIVOICE_VENV=.venv2 ./setup_omni.sh             # custom venv dir name
#   SKIP_FLOWTTS_VENV=1 ./setup_omni.sh               # only set up the omnivoice venv
#
# After setup, run FlowTTS with:
#   FLOWTTS_MODEL_TYPE=omnivoice ./run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNIVOICE_DIR="${SCRIPT_DIR}/omnivoice"
VENV_DIR_NAME="${OMNIVOICE_VENV:-.venv}"

if [ ! -f "${OMNIVOICE_DIR}/model.py" ]; then
    echo "error: ${OMNIVOICE_DIR}/model.py not found — vendored omnivoice server code is missing" >&2
    exit 1
fi

# ── 1. Vendored omnivoice server's own venv (hosts the model + microbatch server) ──
(
    cd "${OMNIVOICE_DIR}"
    VENV_DIR="${OMNIVOICE_DIR}/${VENV_DIR_NAME}"

    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating OmniVoice virtualenv at ${VENV_DIR}"
        python3 -m venv "${VENV_DIR}"
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    pip install --upgrade pip

    # requirements.txt keeps vllm/vllm-omni commented out (see the file for why
    # — torch>=2.11/CUDA 13 mismatch with this host's driver); everything else
    # in it (omnivoice, soundfile, torch) installs normally.
    pip install -r "${OMNIVOICE_DIR}/requirements.txt"

    # A bare `torch` resolves to the newest release, which now ships CUDA-13
    # wheels by default — too new for this host's driver (550.127.08, CUDA
    # 12.4 max), and torchaudio then fails to load against it (libcudart.so.13
    # not found). Pin torch+torchaudio to a matching cu124 build explicitly;
    # this satisfies omnivoice's own torch>=2.4/torchaudio>=2.4 requirement.
    pip install --index-url https://download.pytorch.org/whl/cu124 \
        "torch==2.6.0" "torchaudio==2.6.0"

    # Serving-layer deps used by model.py, not listed in requirements.txt.
    pip install fastapi uvicorn requests pydantic aiohttp

    deactivate
    echo "OmniVoice venv ready: ${VENV_DIR}"
)

# ── 2. FlowTTS's own venv, only if it doesn't already exist ────────────────
# If flowtts/setup/setup.sh already ran (full Mira/VoxCPM install), this is a
# no-op — that venv already has everything below plus much more.
FLOWTTS_VENV_DIR="${SCRIPT_DIR}/.venv"
if [ "${SKIP_FLOWTTS_VENV:-0}" = "1" ]; then
    echo "SKIP_FLOWTTS_VENV=1 — leaving FlowTTS's venv untouched"
elif [ -d "${FLOWTTS_VENV_DIR}" ]; then
    echo "FlowTTS venv already exists at ${FLOWTTS_VENV_DIR} — leaving as-is"
else
    echo "No FlowTTS venv found — creating a minimal one to run server.py in omnivoice mode"
    python3 -m venv "${FLOWTTS_VENV_DIR}"
    # shellcheck disable=SC1091
    source "${FLOWTTS_VENV_DIR}/bin/activate"
    pip install --upgrade pip
    # Everything server.py imports at module load time, independent of
    # model_type (sglang/torch-tensorrt/nanovllm_voxcpm are lazily imported
    # only inside Mira's/VoxCPM's own initialize() and are NOT needed here).
    pip install \
        numpy soundfile structlog "prometheus_client" \
        aiohttp websockets "pydantic>=2" "pydantic-settings"
    deactivate
    echo "Minimal FlowTTS venv ready: ${FLOWTTS_VENV_DIR}"
    echo "(run flowtts/setup/setup.sh later if/when you need the Mira or VoxCPM paths)"
fi

echo
echo "Setup complete."
echo
echo "FlowTTS defaults to this layout (flowtts/core/config.py: OmniVoiceSettings):"
echo "  repo_dir     = ${OMNIVOICE_DIR}"
echo "  venv_python  = ${OMNIVOICE_DIR}/.venv/bin/python"
echo
echo "If you used a different venv name, point FlowTTS at it with:"
echo "  export FLOWTTS_OMNIVOICE__VENV_PYTHON=${OMNIVOICE_DIR}/${VENV_DIR_NAME}/bin/python"
echo
echo "Then run FlowTTS with the OmniVoice backend, e.g.:"
echo "  FLOWTTS_MODEL_TYPE=omnivoice ./run.sh"
