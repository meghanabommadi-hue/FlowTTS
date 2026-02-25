#!/usr/bin/env bash
# FlowTTS launcher — single process, one model load, N WebSocket ports.
#
# Usage:
#   ./run.sh                        # 1 port at 8765
#   ./run.sh --ports 3              # ports 8765, 8766, 8767
#   ./run.sh --ports 3 --port 9000  # ports 9000, 9001, 9002
#   ./run.sh --test                 # quick smoke test against running server
#   ./run.sh --test --ports 3 --port 8765
#
# The venv at /root/CleanTTSData/.venv has all required packages.

set -uo pipefail

VENV="/root/CleanTTSData/.venv"
PYTHON="${VENV}/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export PATH="${VENV}/bin:${PATH}"

# onnxruntime CUDAExecutionProvider needs libcudnn.so.9 (installed via pip as nvidia-cudnn-cu12).
# torch-tensorrt needs libcudart.so.13 (installed via cuda-toolkit pip package).
# Neither is on the system LD path, so we prepend the venv nvidia lib dirs here.
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

# ── Parse args ────────────────────────────────────────────────────────────────
BASE_PORT=8765
N_PORTS=1
TEST=0
TEST_HOST="localhost"
SAVE_AUDIO=""
CTRL_PORT=""
SKIP_DECODER=""

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
        --skip-decoder)
            SKIP_DECODER="1"; shift ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--ports N] [--port BASE] [--ctrl-port PORT] [--save-audio DIR] [--skip-decoder] [--test [--host H]]"
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

# ── Server mode — one process, one model, N ports ────────────────────────────
echo "[FlowTTS] Starting server: ${N_PORTS} port(s) from ${BASE_PORT}..."

# sglang's internal scheduler may crash and send SIGQUIT to the process tree,
# killing the server. Restart automatically with a brief backoff.
RESTART_DELAY=5
while true; do
    "$PYTHON" -m flowtts.server --ports "${N_PORTS}" --base-port "${BASE_PORT}" \
        ${SAVE_AUDIO:+--save-audio "${SAVE_AUDIO}"} \
        ${CTRL_PORT:+--ctrl-port "${CTRL_PORT}"} \
        ${SKIP_DECODER:+--skip-decoder}
    EXIT_CODE=$?
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[FlowTTS] Server exited cleanly (code 0). Stopping."
        break
    fi
    echo "[FlowTTS] Server exited with code ${EXIT_CODE}. Restarting in ${RESTART_DELAY}s..." >&2
    sleep "${RESTART_DELAY}"
done
