#!/usr/bin/env bash
# Orchestrator: start FlowTTS server, generate simran audio, stream-push to HF.
# Resume-safe: skips already-generated wavs, picks up where it stopped.
#
# Usage:
#   cd /home/ubuntu/FlowTTS
#   export HF_TOKEN=<your_hf_token>
#   bash cache_generation/run_simran_cache.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOWTTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${FLOWTTS_DIR}/.venv/bin/python"
PORT=8765
LOG="${SCRIPT_DIR}/simran_cache_run.log"

export HF_TOKEN="${HF_TOKEN:-<your_hf_token>}"

echo "[INFO] FlowTTS dir : ${FLOWTTS_DIR}"
echo "[INFO] Log file    : ${LOG}"
echo "[INFO] Port        : ${PORT}"

# ── Step 1: Start FlowTTS server in background ──────────────────────────────
echo "[INFO] Starting FlowTTS server on port ${PORT}..."

# Kill any stale server on this port
fuser -k "${PORT}/tcp" 2>/dev/null || true

VENV="${FLOWTTS_DIR}/.venv"
export PYTHONPATH="${FLOWTTS_DIR}:${PYTHONPATH:-}"
export PATH="${VENV}/bin:${PATH}"
PY_VER="$("${PYTHON}" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
export LD_LIBRARY_PATH="${VENV}/lib/${PY_VER}/site-packages/torch/lib:${VENV}/lib/${PY_VER}/site-packages/nvidia/cudnn/lib:${VENV}/lib/${PY_VER}/site-packages/nvidia/cu13/lib:${VENV}/lib/${PY_VER}/site-packages/tensorrt_libs:${LD_LIBRARY_PATH:-}"

"${PYTHON}" -m flowtts.server --ports 1 --base-port "${PORT}" \
    >> "${LOG}" 2>&1 &
SERVER_PID=$!
echo "[INFO] Server PID: ${SERVER_PID}"

# ── Step 2: Wait for server to be ready ──────────────────────────────────────
echo "[INFO] Waiting for server to be ready..."
MAX_WAIT=300
ELAPSED=0
while ! ss -tlnp | grep -q ":${PORT}"; do
    sleep 3
    ELAPSED=$((ELAPSED + 3))
    if [[ ${ELAPSED} -ge ${MAX_WAIT} ]]; then
        echo "[ERROR] Server did not start within ${MAX_WAIT}s. Check ${LOG}"
        kill "${SERVER_PID}" 2>/dev/null || true
        exit 1
    fi
    echo "  waiting... ${ELAPSED}s"
done
echo "[INFO] Server is up. Waiting 10s for warmup..."
sleep 10

# ── Step 3: Run generate + push ──────────────────────────────────────────────
echo "[INFO] Starting audio generation and HF push..."

"${PYTHON}" "${SCRIPT_DIR}/generate_and_push_hf.py" \
    --voice simran \
    --port "${PORT}" \
    --concurrency 4 \
    --batch-size 500 \
    --token "${HF_TOKEN}" \
    2>&1 | tee -a "${LOG}"

EXIT_CODE=${PIPESTATUS[0]}

# ── Step 4: Shutdown server ──────────────────────────────────────────────────
echo "[INFO] Shutting down server (PID ${SERVER_PID})..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "[DONE] All done. Check log: ${LOG}"
else
    echo "[ERROR] Generation exited with code ${EXIT_CODE}. Run again to resume — already-generated wavs are skipped."
fi

exit "${EXIT_CODE}"
