#!/usr/bin/env bash
# FlowTTS launcher: starts the FastAPI gateway(s) + TTS worker.
#
# Usage:
#   ./run.sh                        # gateway on 8765 + worker
#   ./run.sh gateway                # gateway only, port 8765
#   ./run.sh worker                 # worker only
#   ./run.sh --ports 3              # 3 gateways (8765,8766,8767) + worker
#   ./run.sh --ports 3 --port 9000  # 3 gateways (9000,9001,9002) + worker
#   ./run.sh --test                 # run test client against default port
#   ./run.sh --test --ports 3       # parallel test against 3 ports
#   ./run.sh --test --ports 3 --port 9000
#
# The venv at /root/CleanTTSData/.venv has all required packages.

set -euo pipefail

VENV="/root/CleanTTSData/.venv"
PYTHON="${VENV}/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ── Parse args ────────────────────────────────────────────────────────────────
MODE="both"
BASE_PORT=8765
N_PORTS=1
TEST=0
TEST_HOST="localhost"

while [[ $# -gt 0 ]]; do
    case "$1" in
        gateway|worker|both)
            MODE="$1"; shift ;;
        --test)
            TEST=1; shift ;;
        --port)
            BASE_PORT="$2"; shift 2 ;;
        --ports)
            N_PORTS="$2"; shift 2 ;;
        --host)
            TEST_HOST="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [gateway|worker|both] [--ports N] [--port BASE] [--test [--host H]]"
            exit 1 ;;
    esac
done

# ── Test mode — just run the client and exit ──────────────────────────────────
if [[ $TEST -eq 1 ]]; then
    echo "[FlowTTS] Running parallel test: ${N_PORTS} port(s) from ${BASE_PORT} on ${TEST_HOST}"
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
                import json as _json
                out.write_text(_json.dumps({"text": text, "llm_s": llm_s, "total_s": t1-t0, "audio_tokens": tokens}, ensure_ascii=False, indent=2))
                print(f"[port {port}][{i}] OK  {t1-t0:.2f}s  llm={llm_s}s  tokens={n_tok}  -> {out.name}", flush=True)
    except Exception as e:
        print(f"[port {port}] FAILED: {e}", flush=True)

async def main():
    ports = [BASE + i for i in range(N)]
    t0 = time.perf_counter()
    await asyncio.gather(*[test_port(p) for p in ports])
    elapsed = time.perf_counter() - t0
    wavs = sorted(OUT_DIR.glob("*.wav"))
    print(f"\nDone in {elapsed:.2f}s. {len(wavs)} WAV(s) in {OUT_DIR}/", flush=True)

asyncio.run(main())
PYEOF
    exit 0
fi

# ── Server mode ───────────────────────────────────────────────────────────────
# Build comma-separated list of all gateway ports for /ports endpoint
KNOWN_PORTS_CSV=""
for i in $(seq 0 $((N_PORTS - 1))); do
    p=$((BASE_PORT + i))
    KNOWN_PORTS_CSV="${KNOWN_PORTS_CSV:+${KNOWN_PORTS_CSV},}${p}"
done

GATEWAY_PIDS=()

cleanup() {
    echo "[FlowTTS] Shutting down..."
    for pid in "${GATEWAY_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_worker() {
    echo "[FlowTTS] Starting TTS worker..."
    "$PYTHON" -m flowtts.worker
}

run_gateway() {
    local port="$1"
    echo "[FlowTTS] Starting gateway on port ${port}..."
    FLOWTTS_WS__PORT="${port}" \
    FLOWTTS_KNOWN_PORTS="${KNOWN_PORTS_CSV}" \
    "$PYTHON" -m flowtts.main &
    GATEWAY_PIDS+=($!)
}

case "$MODE" in
    worker)
        run_worker
        ;;

    gateway)
        for i in $(seq 0 $((N_PORTS - 1))); do
            run_gateway $((BASE_PORT + i))
        done
        # Wait for all gateways (foreground)
        wait "${GATEWAY_PIDS[@]}"
        ;;

    both)
        # Start worker in background
        run_worker &
        WORKER_PID=$!

        # Start all gateways in background
        for i in $(seq 0 $((N_PORTS - 1))); do
            run_gateway $((BASE_PORT + i))
        done

        echo "[FlowTTS] ${N_PORTS} gateway(s) + worker running. Ctrl+C to stop."

        # Wait for any process to die, then clean up
        wait -n 2>/dev/null || wait
        ;;
esac
