#!/usr/bin/env bash
# Sets up the OmniVoice backend used by FlowTTS's OmniVoiceSynthesizer
# (flowtts/synthesis/omnivoice.py, model_type=omnivoice).
#
# OmniVoice's own dependencies (transformers>=5.3.0) conflict with the
# transformers pin the Mira/VoxCPM paths need, so OmniVoice is installed
# into its own dedicated venv (.venv_omni) inside the omnivoice_scaled
# checkout; FlowTTS spawns that venv's python as a child process
# (src/model.py) and talks to it over loopback HTTP.
#
# Usage:
#   ./flowtts/setup/setup_omni.sh
#   OMNIVOICE_REPO_DIR=/path/to/omnivoice_scaled ./flowtts/setup/setup_omni.sh
set -euo pipefail

OMNIVOICE_REPO_DIR="${OMNIVOICE_REPO_DIR:-${HOME}/omnivoice_scaled}"
VENV_DIR_NAME=".venv_omni"

if [ ! -d "${OMNIVOICE_REPO_DIR}" ]; then
    echo "error: omnivoice_scaled checkout not found at ${OMNIVOICE_REPO_DIR}" >&2
    echo "       set OMNIVOICE_REPO_DIR to point at your omnivoice_scaled checkout" >&2
    exit 1
fi

if [ ! -f "${OMNIVOICE_REPO_DIR}/src/model.py" ]; then
    echo "error: ${OMNIVOICE_REPO_DIR}/src/model.py not found — is this really omnivoice_scaled?" >&2
    exit 1
fi

(
    cd "${OMNIVOICE_REPO_DIR}"
    VENV_DIR="${OMNIVOICE_REPO_DIR}/${VENV_DIR_NAME}"

    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating OmniVoice virtualenv at ${VENV_DIR}"
        python3 -m venv "${VENV_DIR}"
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    pip install --upgrade pip
    pip install -r "${OMNIVOICE_REPO_DIR}/requirements.txt"

    # A bare `torch` resolves to the newest release, which now ships CUDA-13
    # wheels by default — too new for this host's driver. Pin to a matching
    # cu124 build explicitly.
    pip install --index-url https://download.pytorch.org/whl/cu124 \
        "torch==2.6.0" "torchaudio==2.6.0"

    # Serving-layer deps used by src/model.py, not listed in requirements.txt.
    pip install fastapi uvicorn requests pydantic aiohttp

    deactivate
    echo "OmniVoice venv ready: ${VENV_DIR}"
)

echo
echo "Setup complete."
echo
echo "Point FlowTTS at this venv with:"
echo "  export FLOWTTS_OMNIVOICE__REPO_DIR=${OMNIVOICE_REPO_DIR}"
echo "  export FLOWTTS_OMNIVOICE__VENV_PYTHON=${OMNIVOICE_REPO_DIR}/${VENV_DIR_NAME}/bin/python"
echo
echo "Then run FlowTTS with the OmniVoice backend, e.g.:"
echo "  FLOWTTS_MODEL_TYPE=omnivoice ./run.sh"
