#!/usr/bin/env bash
# Sets up the miotts backend used by FlowTTS's MiottsSynthesizer
# (flowtts/synthesis/miotts.py, model_type=miotts).
#
# Installed into its own dedicated venv (.venv_mio, Python 3.12) since
# miocodec's package metadata declares Requires-Python >=3.12.
#
# Usage:
#   ./flowtts/setup/setup_mio.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv_mio"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "Creating miotts virtualenv at ${VENV_DIR}"
    uv venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

uv pip install -r "$REPO_ROOT/requirements-miotts.txt"

deactivate

echo
echo "Setup complete."
echo
echo "miotts venv ready: ${VENV_DIR}"
echo "Run FlowTTS with the miotts backend using this venv's python, e.g.:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  FLOWTTS_MODEL_TYPE=miotts ./run.sh"
