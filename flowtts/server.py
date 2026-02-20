#!/usr/bin/env python3
"""
FlowTTS single-process server.

Loads the model ONCE, then binds N WebSocket ports in the same asyncio
event loop — identical pattern to CleanTTSData/websocket_new/run_ws_server.py.

No Redis, no separate worker, no uvicorn per port. One process, one GPU load.

Usage (via run.sh):
    ./run.sh --ports 3              # ports 8765, 8766, 8767
    ./run.sh --ports 3 --port 9000  # ports 9000, 9001, 9002

Direct:
    python -m flowtts.server --ports 3
    python -m flowtts.server --ports 3 --base-port 9000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid

import logging

import websockets
from websockets.exceptions import WebSocketException

from flowtts.core.config import settings
from flowtts.synthesis.models import FlowTtsSynthesizer

# Silence websockets' own logger — we do our own prints
logging.getLogger("websockets").setLevel(logging.CRITICAL)

_synthesizer: FlowTtsSynthesizer | None = None


def _ts() -> str:
    return time.strftime("%H:%M:%S")


async def _get_synthesizer() -> FlowTtsSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = FlowTtsSynthesizer()
        print(f"[{_ts()}] loading model...", flush=True)
        await _synthesizer.initialize()
        print(f"[{_ts()}] model ready", flush=True)
    return _synthesizer


async def handle_connection(ws: websockets.ServerConnection, port: int) -> None:
    """Handle one persistent WebSocket connection (one call = one socket)."""
    peer = ws.remote_address
    print(f"[{_ts()}] :{port} connected  peer={peer[0]}:{peer[1]}", flush=True)

    try:
        async for raw in ws:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
                continue

            text = (data.get("text") or "").strip()
            call_id = data.get("call_id") or f"{peer[0]}:{peer[1]}"
            text_id = data.get("text_id") or str(uuid.uuid4())

            if not text:
                await ws.send(json.dumps({
                    "type": "error", "call_id": call_id, "text_id": text_id,
                    "error": "Missing text",
                }))
                continue

            print(f"[{_ts()}] :{port} {call_id}  req  {text[:60]!r}", flush=True)

            try:
                synth = await _get_synthesizer()
                t0 = time.perf_counter()
                audio_tokens = await synth.synthesize(text)
                llm_ms = round((time.perf_counter() - t0) * 1000)
                llm_s = round(time.perf_counter() - t0, 4)

                token_count = audio_tokens.count("<|speech_token_")

                await ws.send(json.dumps({
                    "type": "audio",
                    "call_id": call_id,
                    "text_id": text_id,
                    "audio_tokens": audio_tokens,
                    "is_final": True,
                    "llm_s": llm_s,
                }))

                print(
                    f"[{_ts()}] :{port} {call_id}  done  llm={llm_ms}ms  tokens={token_count}",
                    flush=True,
                )

            except Exception as e:
                print(f"[{_ts()}] :{port} {call_id}  ERROR: {e}", flush=True)
                await ws.send(json.dumps({
                    "type": "error", "call_id": call_id, "text_id": text_id,
                    "error": str(e),
                }))

    except WebSocketException:
        pass  # client disconnected — normal
    except Exception as e:
        print(f"[{_ts()}] :{port} connection error: {e}", flush=True)
    finally:
        print(f"[{_ts()}] :{port} disconnected  peer={peer[0]}:{peer[1]}", flush=True)


async def run_server(base_port: int, n_ports: int) -> None:
    # Load model once before binding ports
    await _get_synthesizer()

    ports = [base_port + i for i in range(n_ports)]

    for p in ports:
        async def handler(ws: websockets.ServerConnection, port: int = p) -> None:
            await handle_connection(ws, port)

        await websockets.serve(
            handler,
            "0.0.0.0",
            p,
            ping_interval=30,
            ping_timeout=30,
            max_size=100 * 1024 * 1024,
        )

    print(f"\n[{_ts()}] FlowTTS  {n_ports} port(s) ready:", flush=True)
    for p in ports:
        print(f"  ws://0.0.0.0:{p}", flush=True)
    print(flush=True)

    await asyncio.Future()  # run forever


def main() -> None:
    parser = argparse.ArgumentParser(description="FlowTTS single-process WebSocket server")
    parser.add_argument("--base-port", type=int, default=settings.ws.port,
                        help=f"First port to bind (default: {settings.ws.port})")
    parser.add_argument("--ports", type=int, default=1,
                        help="Number of WebSocket ports to open (default: 1)")
    args = parser.parse_args()

    try:
        asyncio.run(run_server(args.base_port, args.ports))
    except KeyboardInterrupt:
        print("\n[FlowTTS] Stopped.")


if __name__ == "__main__":
    main()
