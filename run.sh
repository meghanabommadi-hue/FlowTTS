#!/usr/bin/env bash
# FlowTTS (OmniVoice) launcher — single process, one model load, N WebSocket ports.
#
# Usage:
#   ./run.sh                                   # 1 port at 8080, defaults
#   ./run.sh --ports 100                       # 100 ports: 8080…8179
#   ./run.sh --ports 3 --port 9000             # ports 9000, 9001, 9002
#   ./run.sh --ctrl-port 8764                  # enable HTTP control API
#   ./run.sh --num-step 12 --max-batch 48      # engine tuning (speed/throughput)
#   ./run.sh --test --ports 3                  # quick smoke test against running server
#
# Engine flags → forwarded as FLOWTTS_OMNIVOICE__* env vars to pydantic-settings:
#   --num-step N          diffusion steps (dominant latency knob; 16 default, try 8-12)
#   --max-batch N         max requests per batched generate()   (default 32)
#   --batch-timeout-ms N  batch collection window in ms          (default 8)
#   --compile             torch.compile the model (+CUDA graphs) — first run is slow

set -uo pipefail

VENV="${VIRTUAL_ENV:-${HOME}/FlowTTS/.venv}"
PYTHON="${VENV}/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export PATH="${VENV}/bin:${PATH}"

# torch bundles its CUDA libs; cudnn wheel dir added for completeness.
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/torch/lib:${VENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
# Faster HF downloads (Xet high-performance transfer).
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
# Treat an empty HF_TOKEN as unset (avoids an illegal "Bearer " auth header).
[ -n "${HF_TOKEN:-}" ] || unset HF_TOKEN 2>/dev/null || true

# ── Parse args ────────────────────────────────────────────────────────────────
BASE_PORT=8080
N_PORTS=1
TEST=0
TEST_HOST="localhost"
SAVE_AUDIO=""
CTRL_PORT=""
OV_NUM_STEP=""
OV_MAX_BATCH=""
OV_BATCH_TIMEOUT_MS=""
OV_COMPILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)             TEST=1; shift ;;
        --port)             BASE_PORT="$2"; shift 2 ;;
        --ports)            N_PORTS="$2"; shift 2 ;;
        --host)             TEST_HOST="$2"; shift 2 ;;
        --save-audio)       SAVE_AUDIO="$2"; shift 2 ;;
        --ctrl-port)        CTRL_PORT="$2"; shift 2 ;;
        --num-step)         OV_NUM_STEP="$2"; shift 2 ;;
        --max-batch)        OV_MAX_BATCH="$2"; shift 2 ;;
        --batch-timeout-ms) OV_BATCH_TIMEOUT_MS="$2"; shift 2 ;;
        --compile)          OV_COMPILE="true"; shift ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--ports N] [--port BASE] [--ctrl-port PORT] [--save-audio DIR]"
            echo "          [--num-step N] [--max-batch N] [--batch-timeout-ms N] [--compile]"
            echo "          [--test [--host H]]"
            exit 1 ;;
    esac
done

# ── Test mode — quick smoke test against a running server ─────────────────────
if [[ $TEST -eq 1 ]]; then
    echo "[FlowTTS] Smoke test: ${N_PORTS} port(s) from ${BASE_PORT} on ${TEST_HOST}"
    "$PYTHON" - <<PYEOF
import asyncio, json, time
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
                # drain until audio_done
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

# ── First-run setup: ensure model + at least one voice npz exist ──────────────
_ensure_setup() {
    "$PYTHON" - <<'PYEOF'
from pathlib import Path
from flowtts.core.config import settings
vdir = Path(settings.voices.voices_dir)
npzs = list(vdir.glob("*.npz")) if vdir.is_dir() else []
if not npzs:
    print(f"[FlowTTS] No voice npz in {vdir}.", flush=True)
    print("[FlowTTS] Build voices first, e.g.:", flush=True)
    print("  python -m flowtts.voices.clone --build-all", flush=True)
    print(f"[FlowTTS] Server will fall back to OmniVoice auto-voice until voices exist.", flush=True)
else:
    print(f"[FlowTTS] {len(npzs)} voice(s): {sorted(p.stem for p in npzs)}", flush=True)
PYEOF
}
_ensure_setup

# ── Server mode — one process, one model, N ports ────────────────────────────
echo "[FlowTTS] Starting OmniVoice server: ${N_PORTS} port(s) from ${BASE_PORT}..."

LOG_FILE="${SCRIPT_DIR}/llm.log"
> "${LOG_FILE}"

EXTRA_ARGS=()
[[ -n "${SAVE_AUDIO}" ]] && EXTRA_ARGS+=(--save-audio "${SAVE_AUDIO}")
[[ -n "${CTRL_PORT}"  ]] && EXTRA_ARGS+=(--ctrl-port  "${CTRL_PORT}")

# Engine tuning via pydantic-settings env vars (FLOWTTS_OMNIVOICE__<FIELD>)
[[ -n "${OV_NUM_STEP}"          ]] && export FLOWTTS_OMNIVOICE__NUM_STEP="${OV_NUM_STEP}"
[[ -n "${OV_MAX_BATCH}"         ]] && export FLOWTTS_OMNIVOICE__MAX_BATCH="${OV_MAX_BATCH}"
[[ -n "${OV_BATCH_TIMEOUT_MS}"  ]] && export FLOWTTS_OMNIVOICE__BATCH_TIMEOUT_MS="${OV_BATCH_TIMEOUT_MS}"
[[ -n "${OV_COMPILE}"           ]] && export FLOWTTS_OMNIVOICE__COMPILE_MODEL="${OV_COMPILE}"

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
