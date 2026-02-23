#!/usr/bin/env python3
"""
Pipeline position: SINGLE-PROCESS GATEWAY (primary production entry point via run.sh).

Role in pipeline:
  Self-contained TTS server — no Redis, no worker process, no uvicorn per port.
  Loads sglang + ncodec once, then handles all WebSocket ports in one asyncio
  event loop. This is the recommended way to run FlowTTS in production.

  Client
    │  WebSocket (text) on port 8765…8765+N
    ▼
  server.py  (one process, one GPU load)
    │  synthesis_service.synthesize(text)  [sglang in-process]
    │  → audio_tokens string
    ▼
  Client  (audio_tokens JSON — no decode in this path)

Compared to main.py (Redis-backed):
  Simpler:   no Redis, no worker, no inter-process coordination.
  Faster:    no queue latency, inference starts immediately.
  Less flexible: all ports share one sglang Engine, no horizontal scaling
                 across machines without running multiple server.py instances.

Port model:
  --ports N opens N consecutive ports starting at --base-port.
  All ports share the same synthesis_service singleton (one model load).
  Concurrent requests from different ports are handled by asyncio concurrency
  — sglang's async_generate serialises GPU work internally.

Warmup:
  On startup, one warmup sentence is synthesized to prime the GPU and JIT
  caches before real traffic arrives.

Usage (preferred):
    ./run.sh --ports 100              # 100 ports: 8765…8864
    ./run.sh --ports 3 --port 9000   # ports 9000, 9001, 9002

Direct:
    python -m flowtts.server --ports 3
    python -m flowtts.server --ports 100 --base-port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import uuid
from pathlib import Path

import logging

import websockets
from websockets.exceptions import WebSocketException

from flowtts.core.config import settings
from flowtts.decoder.decoder import AudioDecoder
from flowtts.synthesis.models import FlowTtsSynthesizer

# Silence websockets' own logger — we do our own prints
logging.getLogger("websockets").setLevel(logging.CRITICAL)

_synthesizer: FlowTtsSynthesizer | None = None
_decoder: AudioDecoder | None = None
_audio_out_dir: Path | None = None


def _get_decoder() -> AudioDecoder:
    global _decoder
    if _decoder is None:
        _decoder = AudioDecoder()
    return _decoder


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
                llm_s = round(time.perf_counter() - t0, 4)
                llm_ms = round(llm_s * 1000)

                token_count = audio_tokens.count("<|speech_token_")

                # Decode tokens → WAV in a thread so the event loop stays free
                dec = _get_decoder()
                loop = asyncio.get_running_loop()
                td = time.perf_counter()
                decoded = await loop.run_in_executor(
                    None,
                    lambda: dec.decode_to_wav(audio_tokens, to_wav=True),
                )
                decode_s = round(time.perf_counter() - td, 4)
                audio_b64 = base64.b64encode(decoded.wav_bytes).decode("ascii")

                if _audio_out_dir is not None:
                    wav_file = _audio_out_dir / f"{text_id}.wav"
                    wav_file.write_bytes(decoded.wav_bytes)
                    print(f"[{_ts()}] :{port}  saved → {wav_file}", flush=True)

                await ws.send(json.dumps({
                    "type": "audio",
                    "call_id": call_id,
                    "text_id": text_id,
                    "audio_tokens": audio_tokens,
                    "audio_base64": audio_b64,
                    "sample_rate": decoded.sample_rate,
                    "is_final": True,
                    "llm_s": llm_s,
                    "decode_s": decode_s,
                }))

                print(
                    f"[{_ts()}] :{port} {call_id}  done  llm={llm_ms}ms  tokens={token_count}"
                    f"  decode={round(decode_s*1000)}ms  wav={len(decoded.wav_bytes)}B",
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


async def _warmup(synth: FlowTtsSynthesizer) -> None:
    sentence = settings.tts_model.warmup_sentence
    if not sentence:
        return
    print(f"[{_ts()}] warmup: running '{sentence}'...", flush=True)
    try:
        t0 = time.perf_counter()
        await synth.synthesize(sentence)
        print(f"[{_ts()}] warmup done  ({(time.perf_counter()-t0)*1000:.0f}ms)", flush=True)
    except Exception as e:
        print(f"[{_ts()}] warmup failed: {e}", flush=True)


async def run_server(base_port: int, n_ports: int) -> None:
    # Load model once before binding ports
    synth = await _get_synthesizer()
    await _warmup(synth)

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
    global _audio_out_dir

    parser = argparse.ArgumentParser(description="FlowTTS single-process WebSocket server")
    parser.add_argument("--base-port", type=int, default=settings.ws.port,
                        help=f"First port to bind (default: {settings.ws.port})")
    parser.add_argument("--ports", type=int, default=1,
                        help="Number of WebSocket ports to open (default: 1)")
    parser.add_argument("--save-audio", type=str, default=None, metavar="DIR",
                        help="Directory to save decoded WAV files (one per request)")
    args = parser.parse_args()

    if args.save_audio:
        _audio_out_dir = Path(args.save_audio)
        _audio_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[FlowTTS] Saving audio to {_audio_out_dir}/", flush=True)

    try:
        asyncio.run(run_server(args.base_port, args.ports))
    except KeyboardInterrupt:
        print("\n[FlowTTS] Stopped.")


if __name__ == "__main__":
    main()
