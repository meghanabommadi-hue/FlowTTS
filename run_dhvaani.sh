#!/usr/bin/env bash
# DhVaani launcher -- one process, one model load, WebSocket + REST + control API.
#
# Usage:
#   ./run_dhvaani.sh                                   # 1 WS port at 8080, REST 8000
#   ./run_dhvaani.sh --ports 4 --ctrl-port 8764
#   ./run_dhvaani.sh --profile fast --backend trt      # throughput configuration
#   ./run_dhvaani.sh --profile quality                 # 16-step, model-card quality
#
# Profiles (see flowtts/dhvaani/config.py::PROFILES):
#   fast      4 steps, CFG off   -- highest throughput, lowest TTFB
#   balanced  8 steps, CFG to t=0.5  (default)
#   quality  16 steps, CFG on    -- model-card settings, ~4x the FLOPs of fast
#
# Backends:
#   torch   always works; CUDA graphs on by default
#   trt     in-process TensorRT; run setup.build_trt first
#   triton  NVIDIA Triton Inference Server client (see triton/README.md)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VIRTUAL_ENV:-${SCRIPT_DIR}/.venv}"
PYTHON="${VENV}/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export PATH="${VENV}/bin:${PATH}"

# onnxruntime/TensorRT/cuDNN ship as pip wheels whose .so files are not on the
# system loader path. LD_LIBRARY_PATH is read once at process start, so it has
# to be set here rather than from inside Python.
PYVER="$("$PYTHON" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo python3.10)"
SP="${VENV}/lib/${PYVER}/site-packages"
export LD_LIBRARY_PATH="${SP}/torch/lib:${SP}/nvidia/cudnn/lib:${SP}/nvidia/cublas/lib:${SP}/tensorrt_libs:${LD_LIBRARY_PATH:-}"

# expandable_segments is the single most effective defence against the caching
# allocator fragmenting under variable-length vocoder outputs. config.py sets a
# default too; this makes it explicit for anyone reading the launcher.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${HF_TOKEN:-}" ]] && [[ ! -d "${HOME}/models/DhVaani-0.5" ]]; then
    echo "[DhVaani] HF_TOKEN is not set and no local snapshot exists."
    echo "          The model repo is gated: accept its terms at"
    echo "          https://huggingface.co/ARTPARK-IISc/DhVaani-0.5, then"
    echo "            export HF_TOKEN=hf_xxxxxxxx"
    echo "            python -m flowtts.dhvaani.setup.fetch_model"
    exit 1
fi

RESTART_DELAY=5
LOG_FILE="${SCRIPT_DIR}/dhvaani.log"

while true; do
    echo "[DhVaani] starting: $* "
    "$PYTHON" -m flowtts.dhvaani.server "$@" 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}

    # 0 = clean shutdown (SIGINT/SIGTERM). Anything else -- notably the
    # two-strikes OOM path in server.py -- means recycle the process.
    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[DhVaani] clean exit."
        break
    fi
    echo "[DhVaani] exited with ${EXIT_CODE}; restarting in ${RESTART_DELAY}s..."
    sleep "${RESTART_DELAY}"
done
