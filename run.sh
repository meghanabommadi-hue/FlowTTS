#!/usr/bin/env bash
# FlowTTS gateway launcher — CPU-only WebSocket/control server that proxies to the
# Fish Audio S2 Pro sglang backend. Single process, N WebSocket ports, one event loop.
#
# The GPU model runs SEPARATELY (sglang-omni, see docker/fish_s2pro.Dockerfile or run
# `sgl-omni serve --model-path fishaudio/s2-pro --config .../s2pro_tts.yaml --port 8000`).
#
# Usage:
#   ./run.sh                                   # 1 port at 8080, backend at :8000
#   ./run.sh --ports 100                       # 100 ports: 8080…8179
#   ./run.sh --ports 3 --port 9000             # ports 9000, 9001, 9002
#   ./run.sh --ctrl-port 8764                  # enable HTTP control API
#   ./run.sh --backend-url http://127.0.0.1:8000
#   ./run.sh --test --ports 3                  # quick smoke test against running server
#
# Config → forwarded as FLOWTTS_* env vars to pydantic-settings:
#   --backend-url URL     sglang backend base URL (FLOWTTS_FISH__BACKEND_URL)

set -uo pipefail

VENV="${VIRTUAL_ENV:-${HOME}/FlowTTS/.venv}"
PYTHON="${VENV}/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export PATH="${VENV}/bin:${PATH}"

# ── Parse args ────────────────────────────────────────────────────────────────
BASE_PORT=8080
N_PORTS=1
TEST=0
TEST_HOST="localhost"
SAVE_AUDIO=""
CTRL_PORT=""
BACKEND_URL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)             TEST=1; shift ;;
        --port)             BASE_PORT="$2"; shift 2 ;;
        --ports)            N_PORTS="$2"; shift 2 ;;
        --host)             TEST_HOST="$2"; shift 2 ;;
        --save-audio)       SAVE_AUDIO="$2"; shift 2 ;;
        --ctrl-port)        CTRL_PORT="$2"; shift 2 ;;
        --backend-url)      BACKEND_URL="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--ports N] [--port BASE] [--ctrl-port PORT] [--save-audio DIR]"
            echo "          [--backend-url URL] [--test [--host H]]"
            exit 1 ;;
    esac
done

# ── Test mode — quick smoke test against a running server ─────────────────────
if [[ $TEST -eq 1 ]]; then
    echo "[FlowTTS] Smoke test: ${N_PORTS} port(s) from ${BASE_PORT} on ${TEST_HOST}"
    "$PYTHON" - <<PYEOF
import asyncio, json
HOST, BASE, N = "${TEST_HOST}", ${BASE_PORT}, ${N_PORTS}
TEXTS = [
    "नमस्ते, मैं प्रिया बोल रही हूँ बजाज फाइनेंस से।",
    "आपकी pending EMI बारह सौ रुपए है, जो पांच जुलाई को due है।",
]
import websockets
async def test_port(port):
    call_id = f"{HOST}:{port}"
    url = f"ws://{HOST}:{port}/ws/{call_id}"
    try:
        async with websockets.connect(url, max_size=100*1024*1024) as ws:
            for i, text in enumerate(TEXTS):
                await ws.send(json.dumps({"type":"synthesize","call_id":call_id,
                                          "text_id":f"p{port}-{i}","text":text}))
                while True:
                    raw = await ws.recv()
                    msg = json.loads(raw[:raw.index(b'}')+1]) if isinstance(raw, bytes) else json.loads(raw)
                    if msg.get("type") in ("audio_done","error"):
                        print(f"[port {port}][{i}] {msg.get('type')} rtf={msg.get('rtf')} sr={msg.get('sample_rate')}", flush=True)
                        break
    except Exception as e:
        print(f"[port {port}] FAILED: {e}", flush=True)
async def main():
    await asyncio.gather(*[test_port(BASE + i) for i in range(N)])
asyncio.run(main())
PYEOF
    exit 0
fi

# ── First-run setup: ensure at least one voice reference exists ────────────────
_ensure_setup() {
    "$PYTHON" - <<'PYEOF'
from pathlib import Path
from flowtts.core.config import settings
vdir = Path(settings.voices.voices_dir)
manifests = list(vdir.glob("*.json")) if vdir.is_dir() else []
if not manifests:
    print(f"[FlowTTS] No voice references in {vdir}.", flush=True)
    print("[FlowTTS] Build voices first, e.g.:", flush=True)
    print("  python -m flowtts.voices.clone --add priya --ref-audio sample_files/vikram.wav --ref-text '…' --lang hi", flush=True)
    print("[FlowTTS] Server will fall back to the backend 'default' voice until voices exist.", flush=True)
else:
    print(f"[FlowTTS] {len(manifests)} voice(s): {sorted(p.stem for p in manifests)}", flush=True)
PYEOF
}
_ensure_setup

# ── Server mode — one process, N ports ────────────────────────────────────────
echo "[FlowTTS] Starting gateway: ${N_PORTS} port(s) from ${BASE_PORT}  backend=${BACKEND_URL:-<default>}"

LOG_FILE="${SCRIPT_DIR}/llm.log"
> "${LOG_FILE}"

EXTRA_ARGS=()
[[ -n "${SAVE_AUDIO}" ]] && EXTRA_ARGS+=(--save-audio "${SAVE_AUDIO}")
[[ -n "${CTRL_PORT}"  ]] && EXTRA_ARGS+=(--ctrl-port  "${CTRL_PORT}")

[[ -n "${BACKEND_URL}" ]] && export FLOWTTS_FISH__BACKEND_URL="${BACKEND_URL}"

RESTART_DELAY=5

while true; do
    [[ -n "${CTRL_PORT}" ]] && fuser -k "${CTRL_PORT}/tcp" 2>/dev/null || true
    fuser -k "${BASE_PORT}/tcp" 2>/dev/null || true

    "$PYTHON" -m flowtts.server --ports "${N_PORTS}" --base-port "${BASE_PORT}" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[FlowTTS] Clean exit. Stopping."
        break
    fi
    echo "[FlowTTS] Server exited with code ${EXIT_CODE}. Restarting in ${RESTART_DELAY}s..."
    sleep "${RESTART_DELAY}"
done
