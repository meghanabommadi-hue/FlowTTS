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
from aiohttp import web

import websockets
from websockets.exceptions import WebSocketException

from flowtts.core.config import settings
from flowtts.decoder.decoder import AudioDecoder
from flowtts.synthesis.models import FlowTtsSynthesizer

# Silence websockets' own logger — we do our own prints
logging.getLogger("websockets").setLevel(logging.CRITICAL)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

_synthesizer: FlowTtsSynthesizer | None = None
_decoder: AudioDecoder | None = None
_audio_out_dir: Path | None = None
_open_ports: set[int] = set()  # tracks all bound WS ports
_skip_decoder: bool = False  # set via --skip-decoder CLI flag

# Limit concurrent decoder (ONNX+FASR) calls to avoid GPU OOM.
# The LLM already serialises itself via sglang; the decoder does not.
# 2 concurrent decodes is safe even with 88 ports; raise cautiously.
_DECODE_CONCURRENCY = 100
_decode_sem: asyncio.Semaphore | None = None  # initialised in run_server


def _get_decode_sem() -> asyncio.Semaphore:
    global _decode_sem
    if _decode_sem is None:
        _decode_sem = asyncio.Semaphore(_DECODE_CONCURRENCY)
    return _decode_sem


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

                if _skip_decoder:
                    await ws.send(json.dumps({
                        "type": "audio",
                        "call_id": call_id,
                        "text_id": text_id,
                        "audio_tokens": audio_tokens,
                        "audio_base64": "",
                        "sample_rate": 0,
                        "is_final": True,
                        "llm_s": llm_s,
                        "decode_s": 0,
                    }))
                    print(
                        f"[{_ts()}] :{port} {call_id}  done  llm={llm_ms}ms  tokens={token_count}  (decoder skipped)",
                        flush=True,
                    )
                else:
                    # Decode tokens → WAV in a thread so the event loop stays free.
                    # Semaphore limits concurrent GPU allocations in the decoder.
                    dec = _get_decoder()
                    loop = asyncio.get_running_loop()
                    td = time.perf_counter()
                    async with _get_decode_sem():
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


async def _warmup_port(port: int, sentence: str) -> None:
    """Send one synthesize request through the WS handler on *port* to prime it."""
    url = f"ws://127.0.0.1:{port}/ws/warmup"
    try:
        async with websockets.connect(
            url,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
            open_timeout=10,
        ) as ws:
            await ws.send(json.dumps({
                "type": "synthesize",
                "call_id": "warmup",
                "text_id": "warmup",
                "text": sentence,
            }))
            await ws.recv()
    except Exception as e:
        print(f"[{_ts()}] warmup port {port} failed: {e}", flush=True)


async def _warmup_all_ports(ports: list[int]) -> None:
    """Warm up every bound WS port in batches to avoid decoder GPU OOM.

    Batch size matches _DECODE_CONCURRENCY so we never exceed the semaphore
    limit — which also means each batch completes before the next starts,
    keeping peak GPU pressure predictable regardless of port count.
    """
    sentence = settings.tts_model.warmup_sentence
    if not sentence or not ports:
        return
    print(f"[{_ts()}] warming up {len(ports)} port(s) (batch={_DECODE_CONCURRENCY})...", flush=True)
    t0 = time.perf_counter()
    for i in range(0, len(ports), _DECODE_CONCURRENCY):
        batch = ports[i:i + _DECODE_CONCURRENCY]
        await asyncio.gather(*[_warmup_port(p, sentence) for p in batch])
    print(f"[{_ts()}] all ports warmed up  ({(time.perf_counter()-t0)*1000:.0f}ms)", flush=True)


async def _bind_ws_port(port: int) -> bool:
    """Bind a new WebSocket listener on *port*. Returns False if already bound."""
    if port in _open_ports:
        return False
    async def handler(ws: websockets.ServerConnection, p: int = port) -> None:
        await handle_connection(ws, p)
    await websockets.serve(
        handler, "0.0.0.0", port,
        ping_interval=30, ping_timeout=30,
        max_size=100 * 1024 * 1024,
    )
    _open_ports.add(port)
    print(f"[{_ts()}] opened ws://0.0.0.0:{port}", flush=True)
    return True


# ---------------------------------------------------------------------------
# HTTP control API  (aiohttp)
# ---------------------------------------------------------------------------
async def _http_add_port(req: web.Request) -> web.Response:
    """POST /ports/add?port=N  — bind a new WS port while the server is running."""
    try:
        port = int(req.rel_url.query["port"])
    except (KeyError, ValueError):
        return web.Response(status=400, text="missing or invalid ?port=N")
    if port < 1024 or port > 65535:
        return web.Response(status=400, text="port out of range")
    already = port in _open_ports
    if not already:
        await _bind_ws_port(port)
    return web.json_response({"port": port, "opened": not already})


async def _http_list_ports(req: web.Request) -> web.Response:
    """GET /ports  — list all currently open WS ports."""
    return web.json_response({"ports": sorted(_open_ports)})


async def _http_ready(req: web.Request) -> web.Response:
    """GET /ready  — 200 once the model is loaded."""
    if _synthesizer is None:
        return web.Response(status=503, text="loading")
    return web.json_response({"ready": True, "ports": sorted(_open_ports)})


async def _run_control_api(ctrl_port: int) -> None:
    app = web.Application()
    app.router.add_post("/ports/add", _http_add_port)
    app.router.add_get("/ports",      _http_list_ports)
    app.router.add_get("/ready",      _http_ready)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", ctrl_port)
    await site.start()
    print(f"[{_ts()}] control API  http://127.0.0.1:{ctrl_port}", flush=True)


async def run_server(base_port: int, n_ports: int, ctrl_port: int | None = None) -> None:
    # Load model once before binding ports
    synth = await _get_synthesizer()
    await _warmup(synth)

    # Start HTTP control API if requested
    if ctrl_port:
        await _run_control_api(ctrl_port)

    # Bind initial WS ports
    initial_ports = [base_port + i for i in range(n_ports)]
    for p in initial_ports:
        await _bind_ws_port(p)

    # Warmup each port — primes connection handling concurrently
    await _warmup_all_ports(initial_ports)

    print(f"\n[{_ts()}] FlowTTS  {len(_open_ports)} port(s) ready:", flush=True)
    for p in sorted(_open_ports):
        print(f"  ws://0.0.0.0:{p}", flush=True)
    print(flush=True)

    await asyncio.Future()  # run forever


def main() -> None:
    global _audio_out_dir, _skip_decoder

    parser = argparse.ArgumentParser(description="FlowTTS single-process WebSocket server")
    parser.add_argument("--base-port", type=int, default=settings.ws.port,
                        help=f"First port to bind (default: {settings.ws.port})")
    parser.add_argument("--ports", type=int, default=1,
                        help="Number of WebSocket ports to open (default: 1)")
    parser.add_argument("--save-audio", type=str, default=None, metavar="DIR",
                        help="Directory to save decoded WAV files (one per request)")
    parser.add_argument("--ctrl-port", type=int, default=None, metavar="PORT",
                        help="HTTP control API port for on-demand WS port binding (e.g. 8764)")
    parser.add_argument("--skip-decoder", action="store_true", default=False,
                        help="Skip decoder: return audio_tokens only, no WAV (faster, LLM-only mode)")
    args = parser.parse_args()

    if args.skip_decoder:
        _skip_decoder = True
        print("[FlowTTS] Decoder disabled — returning audio_tokens only", flush=True)

    if args.save_audio:
        _audio_out_dir = Path(args.save_audio)
        _audio_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[FlowTTS] Saving audio to {_audio_out_dir}/", flush=True)

    try:
        asyncio.run(run_server(args.base_port, args.ports, args.ctrl_port))
    except KeyboardInterrupt:
        print("\n[FlowTTS] Stopped.")


if __name__ == "__main__":
    main()
