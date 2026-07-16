#!/usr/bin/env bash
# Sets up the VoxCPM2 backend used by FlowTTS's VoxCpmSynthesizer
# (flowtts/synthesis/voxcpm.py, model_type=voxcpm).
#
# Installed into its own dedicated venv (.venv_voxcpm) so its deps
# (nanovllm_voxcpm, flow_voxcpm) don't collide with the Mira/TRT path's
# pins. flowtts/synthesis/voxcpm.py expects ~/flow_voxcpm on sys.path and
# imports nanovllm_voxcpm directly, so both must be importable from
# whichever interpreter actually runs server.py in voxcpm mode.
#
# Usage:
#   ./flowtts/setup/setup_voxcpm.sh
#   FLOW_VOXCPM_DIR=/path/to/flow_voxcpm ./flowtts/setup/setup_voxcpm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FLOW_VOXCPM_DIR="${FLOW_VOXCPM_DIR:-${HOME}/flow_voxcpm}"
VENV_DIR="${REPO_ROOT}/.venv_voxcpm"

if [ ! -d "${FLOW_VOXCPM_DIR}" ]; then
    echo "error: flow_voxcpm checkout not found at ${FLOW_VOXCPM_DIR}" >&2
    echo "       set FLOW_VOXCPM_DIR to point at your flow_voxcpm checkout" >&2
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating VoxCPM virtualenv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
uv pip install -r "$REPO_ROOT/requirements.txt"
pip install nanovllm_voxcpm

deactivate

echo
echo "Setup complete."
echo
echo "VoxCPM venv ready: ${VENV_DIR}"
echo "Run FlowTTS with the VoxCPM backend using this venv's python, e.g.:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  FLOWTTS_MODEL_TYPE=voxcpm ./run.sh"
