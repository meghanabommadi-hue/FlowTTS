#!/usr/bin/env bash
# FlowTTS launcher — single process, one model load, N WebSocket ports.
#
# Usage:
#   ./run.sh                                          # 1 port at 8080, defaults
#   ./run.sh --ports 3                                # ports 8080, 8081, 8082
#   ./run.sh --ports 3 --port 9000                    # ports 9000, 9001, 9002
#   ./run.sh --max-batch 128 --batch-timeout-ms 0.2   # decoder tuning
#   ./run.sh --gpu-chunk-size 256 --onnx-workers 4
#   ./run.sh --test                                   # quick smoke test against running server
#   ./run.sh --test --ports 3 --port 8080
#
# Decoder flags (passed as FLOWTTS_DECODER__ env vars to pydantic-settings):
#   --max-batch N          max tokens per decode batch       (default: 256)
#   --batch-timeout-ms N   ms to wait before batch dispatch  (default: 0.5)
#   --gpu-chunk-size N     tokens per GPU iteration          (default: 160)
#   --onnx-workers N       ONNX threads feeding GPU          (default: 2)

set -uo pipefail

VENV="${VIRTUAL_ENV:-${HOME}/FlowTTS/.venv}"
PYTHON="${VENV}/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export PATH="${VENV}/bin:${PATH}"

# onnxruntime CUDAExecutionProvider needs libcudnn.so.9 (installed via pip as nvidia-cudnn-cu12).
# torch-tensorrt needs libcudart.so.13 (installed via cuda-toolkit pip package).
# Neither is on the system LD path, so we prepend the venv nvidia lib dirs here.
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/torch/lib:${VENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib:${VENV}/lib/python3.12/site-packages/tensorrt_libs:${LD_LIBRARY_PATH:-}"

# ── Parse args ────────────────────────────────────────────────────────────────
BASE_PORT=8080
N_PORTS=1
TEST=0
TEST_HOST="localhost"
SAVE_AUDIO=""
CTRL_PORT=""
# Decoder overrides — empty means "use config.py default"
DECODER_MAX_BATCH=""
DECODER_BATCH_TIMEOUT_MS=""
DECODER_GPU_CHUNK_SIZE=""
DECODER_ONNX_WORKERS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            TEST=1; shift ;;
        --port)
            BASE_PORT="$2"; shift 2 ;;
        --ports)
            N_PORTS="$2"; shift 2 ;;
        --host)
            TEST_HOST="$2"; shift 2 ;;
        --save-audio)
            SAVE_AUDIO="$2"; shift 2 ;;
        --ctrl-port)
            CTRL_PORT="$2"; shift 2 ;;
        --max-batch)
            DECODER_MAX_BATCH="$2"; shift 2 ;;
        --batch-timeout-ms)
            DECODER_BATCH_TIMEOUT_MS="$2"; shift 2 ;;
        --gpu-chunk-size)
            DECODER_GPU_CHUNK_SIZE="$2"; shift 2 ;;
        --onnx-workers)
            DECODER_ONNX_WORKERS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--ports N] [--port BASE] [--ctrl-port PORT] [--save-audio DIR]"
            echo "          [--max-batch N] [--batch-timeout-ms N] [--gpu-chunk-size N] [--onnx-workers N]"
            echo "          [--test [--host H]]"
            exit 1 ;;
    esac
done

# ── Test mode — quick smoke test against a running server ─────────────────────
if [[ $TEST -eq 1 ]]; then
    echo "[FlowTTS] Smoke test: ${N_PORTS} port(s) from ${BASE_PORT} on ${TEST_HOST}"
    mkdir -p "${SCRIPT_DIR}/test"
    "$PYTHON" - <<PYEOF
import asyncio, json, time, uuid
from pathlib import Path

HOST    = "${TEST_HOST}"
BASE    = ${BASE_PORT}
N       = ${N_PORTS}
OUT_DIR = Path("${SCRIPT_DIR}/test")

TEXTS = [
    "నమస్కారం, మీరు ఎలా ఉన్నారు?",
    "తెలుగు భాషలో మాట్లాడటం చాలా అందంగా ఉంటుంది.",
]

import websockets

async def test_port(port):
    call_id = f"{HOST}:{port}"
    url = f"ws://{HOST}:{port}/ws/{call_id}"
    print(f"[port {port}] Connecting to {url}", flush=True)
    try:
        async with websockets.connect(url) as ws:
            print(f"[port {port}] Connected", flush=True)
            for i, text in enumerate(TEXTS):
                text_id = f"p{port}-{i}"
                t0 = time.perf_counter()
                await ws.send(json.dumps({
                    "type": "synthesize",
                    "call_id": call_id,
                    "text_id": text_id,
                    "text": text,
                }))
                raw = await ws.recv()
                t1 = time.perf_counter()
                msg = json.loads(raw)
                if msg.get("type") == "error":
                    print(f"[port {port}][{i}] ERROR: {msg['error']}", flush=True)
                    continue
                llm_s = msg.get("llm_s")
                tokens = msg.get("audio_tokens", "")
                n_tok = tokens.count("<|speech_token_") if tokens else 0
                out = OUT_DIR / f"port{port}_text{i}.json"
                out.write_text(json.dumps({"text": text, "llm_s": llm_s, "total_s": t1-t0, "audio_tokens": tokens}, ensure_ascii=False, indent=2))
                print(f"[port {port}][{i}] OK  {(t1-t0)*1000:.0f}ms  llm={int((llm_s or 0)*1000)}ms  tokens={n_tok}  -> {out.name}", flush=True)
    except Exception as e:
        print(f"[port {port}] FAILED: {e}", flush=True)

async def main():
    ports = [BASE + i for i in range(N)]
    t0 = time.perf_counter()
    await asyncio.gather(*[test_port(p) for p in ports])
    elapsed = time.perf_counter() - t0
    jsons = sorted(OUT_DIR.glob("*.json"))
    print(f"\nDone in {elapsed*1000:.0f}ms. {len(jsons)} JSON(s) in {OUT_DIR}/", flush=True)

asyncio.run(main())
PYEOF
    exit 0
fi

# ── Auto-download/resume WAV cache ───────────────────────────────────────────
_ensure_cache() {
    "$PYTHON" - <<'PYEOF'
import sys
from pathlib import Path
from huggingface_hub import HfApi
import importlib

dl = importlib.import_module("flowtts.setup.download_cache")
token = dl.resolve_token()
api = HfApi(token=token)

for voice, repo in dl.HF_REPOS.items():
    cache_dir = Path.home() / f"FlowTTS/cached_data_{voice}"
    existing = len(list(cache_dir.glob("*.wav"))) if cache_dir.exists() else 0
    try:
        total = sum(1 for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".wav"))
    except Exception as e:
        print(f"[FlowTTS] Could not check '{voice}' repo: {e} — skipping.", flush=True)
        continue
    if existing >= total:
        print(f"[FlowTTS] Cache '{voice}' complete ({existing}/{total}) — skipping.", flush=True)
    else:
        print(f"[FlowTTS] Cache '{voice}' incomplete ({existing}/{total}) — downloading...", flush=True)
        dl.download(voice, token)
PYEOF
}
_ensure_cache

# ── Server mode — one process, one model, N ports ────────────────────────────
echo "[FlowTTS] Starting server: ${N_PORTS} port(s) from ${BASE_PORT}..."

LOG_FILE="${SCRIPT_DIR}/llm.log"
> "${LOG_FILE}"   # truncate on each run

EXTRA_ARGS=()
[[ -n "${SAVE_AUDIO}" ]] && EXTRA_ARGS+=(--save-audio "${SAVE_AUDIO}")
[[ -n "${CTRL_PORT}"  ]] && EXTRA_ARGS+=(--ctrl-port  "${CTRL_PORT}")

# Pass decoder overrides via pydantic-settings env vars (FLOWTTS_DECODER__<FIELD>)
[[ -n "${DECODER_MAX_BATCH}"         ]] && export FLOWTTS_DECODER__MAX_BATCH="${DECODER_MAX_BATCH}"
[[ -n "${DECODER_BATCH_TIMEOUT_MS}"  ]] && export FLOWTTS_DECODER__BATCH_TIMEOUT_MS="${DECODER_BATCH_TIMEOUT_MS}"
[[ -n "${DECODER_GPU_CHUNK_SIZE}"    ]] && export FLOWTTS_DECODER__GPU_CHUNK_SIZE="${DECODER_GPU_CHUNK_SIZE}"
[[ -n "${DECODER_ONNX_WORKERS}"      ]] && export FLOWTTS_DECODER__ONNX_WORKERS="${DECODER_ONNX_WORKERS}"

RESTART_DELAY=5

# ── Run stress test once server is ready ─────────────────────────────────────
_run_stress_test() {
    if [[ -z "${CTRL_PORT}" ]]; then
        return
    fi
    local deadline=$(( $(date +%s) + 300 ))
    echo "[FlowTTS] Waiting for server ready before stress test..."
    while [[ $(date +%s) -lt ${deadline} ]]; do
        if curl -sf "http://127.0.0.1:${CTRL_PORT}/ready" 2>/dev/null | grep -qi '"ready": *true'; then
            break
        fi
        sleep 2
    done
    if [[ $(date +%s) -ge ${deadline} ]]; then
        echo "[FlowTTS] Timed out waiting — skipping stress test."
        return
    fi
    echo "[FlowTTS] Running stress test (60 requests, 1 port)..."
    "$PYTHON" -m flowtts.test.test_pipeline \
        --no-launch \
        --ctrl-port "${CTRL_PORT}" \
        --n-ports 1 \
        --base-port "${BASE_PORT}" \
        --requests 60 \
        2>&1 | tee -a "${LOG_FILE}"
    echo "[FlowTTS] Stress test complete."
}

while true; do
    [[ -n "${CTRL_PORT}" ]] && fuser -k "${CTRL_PORT}/tcp" 2>/dev/null || true
    fuser -k "${BASE_PORT}/tcp" 2>/dev/null || true

    _run_stress_test &
    STRESS_PID=$!

    "$PYTHON" -m flowtts.server --ports "${N_PORTS}" --base-port "${BASE_PORT}" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}

    # Kill stress test if still running (e.g. server crashed mid-test)
    kill "${STRESS_PID}" 2>/dev/null || true
    wait "${STRESS_PID}" 2>/dev/null || true

    # Exit code 0 = clean shutdown (KeyboardInterrupt), don't restart
    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[FlowTTS] Clean exit. Stopping."
        break
    fi
    echo "[FlowTTS] Server exited with code ${EXIT_CODE}. Restarting in ${RESTART_DELAY}s..."
    sleep "${RESTART_DELAY}"
done
