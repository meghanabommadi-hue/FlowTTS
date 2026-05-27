#!/usr/bin/env python3
"""
Pipeline position: SINGLE-PROCESS GATEWAY (primary production entry point).

Role in pipeline:
  Self-contained TTS server — no Redis, no worker process.
  Loads the configured TTS model once, then serves all WebSocket ports from
  one asyncio event loop.

  Client  ──WebSocket──▶  server.py  ──get_synthesizer()──▶  BaseSynthesizer
                                                              ├─ MiraSynthesizer   (model_type=mira)
                                                              └─ VoxCpmSynthesizer (model_type=voxcpm)

Model selection:
  FLOWTTS_MODEL_TYPE=mira    (default)
  FLOWTTS_MODEL_TYPE=voxcpm

Wire protocol (identical for every model):
  Non-streaming request:
    Client  →  JSON { type:"synthesize", call_id, text_id, text }
    Server  →  JSON { type:"audio", sample_rate, wav_bytes, llm_s, decode_s, … }
    Server  →  bytes  (full WAV)

  Streaming request:
    Client  →  JSON { …, streaming:true }
    Server  →  (for each chunk)
               JSON { type:"audio_chunk", chunk_index, is_final, sample_rate, … }
               bytes  (chunk WAV)
    Server  →  JSON { type:"audio_done", … }

Port model:
  --ports N opens N consecutive WS ports from --base-port.
  --ctrl-port exposes an HTTP control API for on-demand port binding.

Usage:
    python -m flowtts.server --ports 3 --ctrl-port 8764
    FLOWTTS_MODEL_TYPE=voxcpm python -m flowtts.server --ports 3 --ctrl-port 8764
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import datetime
import json
import time
import uuid
from pathlib import Path

import logging
from aiohttp import web
import websockets
from websockets.exceptions import WebSocketException

from flowtts.core.config import settings
from flowtts.monitoring.metrics import (
    record_call, record_ws_connection_open, record_ws_connection_close,
    record_ws_error, record_ws_done, ws_log_snapshot, record_port_change,
)
from flowtts.processing.text_normalize import normalize_text
from flowtts.synthesis.base import BaseSynthesizer
from flowtts.synthesis.engine import get_synthesizer

logging.getLogger("websockets").setLevel(logging.CRITICAL)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

# ── process-level state ─────────────────────────────────────────────────────
_wav_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="wav_enc")
_audio_out_dir: Path | None = None
_open_ports: set[int] = set()
_rtf_count: int = 0
_rtf_sum:   float = 0.0

_llm_log: Path = Path(__file__).parents[1] / "llm.log"
_llm_log_file = None   # opened once in main()


# ── helpers ─────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _tsms() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log(line: str) -> None:
    if _llm_log_file is not None:
        _llm_log_file.write(line + "\n")
        _llm_log_file.flush()


def _record_rtf(total_s: float, n_tokens: int, sample_rate: int) -> float:
    """Record RTF for one request; return running average.

    For Mira  (discrete tokens): 1 token = 320 samples @ 16 kHz.
    For VoxCPM (continuous PCM): n_tokens == 0, skip RTF tracking.
    """
    global _rtf_count, _rtf_sum
    if n_tokens <= 0:
        return 0.0
    audio_s = n_tokens * 320 / sample_rate
    rtf = total_s / audio_s
    _rtf_count += 1
    _rtf_sum   += rtf
    return _rtf_sum / _rtf_count


# ── streaming handler ────────────────────────────────────────────────────────

async def _handle_streaming(
    ws: websockets.ServerConnection,
    synth: BaseSynthesizer,
    text: str,
    call_id: str,
    text_id: str,
    port: int,
) -> None:
    """Stream audio chunks to the client as they arrive from the synthesizer.

    Works identically for every BaseSynthesizer implementation:
      Mira    — each chunk is a window of decoded speech tokens
      VoxCPM  — each chunk is a VAE output frame
    Both yield SynthChunk with a complete WAV per chunk.
    """
    t0 = time.perf_counter()
    _log(f"{_tsms()}  IN   port={port}  text_id={text_id}  call_id={call_id}  text={text}")

    chunk_index = 0
    total_tokens = 0
    total_wav_b  = 0
    first_chunk_sent = False
    decode_total = 0.0

    try:
        async for sc in synth.synthesize_stream(text):
            ts_chunk = _tsms()

            if not first_chunk_sent:
                ttft = round((time.perf_counter() - t0) * 1000)
                print(f"[{ts_chunk}] :{port} {call_id}  first_chunk  ttft={ttft}ms"
                      f"  bytes={len(sc.wav_bytes)}", flush=True)
                first_chunk_sent = True

            total_tokens += sc.n_tokens
            total_wav_b  += len(sc.wav_bytes)
            if sc.meta.get("decode_s"):
                decode_total = sc.meta["decode_s"]

            await ws.send(json.dumps({
                "type":        "audio_chunk",
                "call_id":     call_id,
                "text_id":     text_id,
                "chunk_index": chunk_index,
                "sample_rate": sc.sample_rate,
                "wav_bytes":   len(sc.wav_bytes),
                "tokens":      sc.n_tokens,
                "is_final":    sc.is_final,
            }))
            await ws.send(sc.wav_bytes)
            chunk_index += 1

        total_s = time.perf_counter() - t0
        ts_done = _tsms()
        _log(f"{ts_done}  OUT  port={port}  text_id={text_id}  call_id={call_id}"
             f"  total_ms={round(total_s*1000)}")

        avg_rtf = _record_rtf(total_s, total_tokens, synth.sample_rate)
        audio_s = total_tokens * 320 / synth.sample_rate if total_tokens else 0
        rtf     = total_s / audio_s if audio_s > 0 else 0.0

        await ws.send(json.dumps({
            "type":            "audio_done",
            "call_id":         call_id,
            "text_id":         text_id,
            "text":            text,
            "chunks":          chunk_index,
            "total_tokens":    total_tokens,
            "total_wav_bytes": total_wav_b,
            "sample_rate":     synth.sample_rate,
            "llm_s":           round(total_s, 4),
            "decode_s":        round(decode_total, 4),
            "rtf":             round(rtf, 3),
            "avg_rtf":         round(avg_rtf, 3),
        }))

        print(
            f"[{ts_done}] :{port} {call_id}  stream_done"
            f"  chunks={chunk_index}  tokens={total_tokens}"
            f"  total={round(total_s*1000)}ms  wav={total_wav_b}B  rtf={rtf:.3f}",
            flush=True,
        )

    except Exception as e:
        ts_err = _tsms()
        print(f"[{ts_err}] :{port} {call_id}  STREAM ERROR: {e}", flush=True)
        record_ws_error(call_id, port=port, text_id=text_id, error=str(e))
        try:
            await ws.send(json.dumps({
                "type": "error", "call_id": call_id, "text_id": text_id, "error": str(e),
            }))
        except Exception:
            pass


# ── non-streaming handler ────────────────────────────────────────────────────

async def _handle_request(
    ws: websockets.ServerConnection,
    synth: BaseSynthesizer,
    text: str,
    call_id: str,
    text_id: str,
    port: int,
    ts_text_recv: str,
) -> None:
    """Full-response (non-streaming) handler — accumulate everything then send."""
    t0 = time.perf_counter()
    ts_llm_start = _tsms()
    _log(f"{ts_llm_start}  IN   port={port}  text_id={text_id}  call_id={call_id}  text={text}")

    try:
        result = await asyncio.wait_for(synth.synthesize(text), timeout=60.0)
        total_s = time.perf_counter() - t0
        ts_done = _tsms()
        _log(f"{ts_done}  OUT  port={port}  text_id={text_id}  call_id={call_id}"
             f"  llm_ms={round(result.llm_s*1000)}  decode_ms={round(result.decode_s*1000)}")

        if _audio_out_dir is not None:
            wav_file = _audio_out_dir / f"{text_id}.wav"
            wav_file.write_bytes(result.wav_bytes)

        avg_rtf = _record_rtf(total_s, result.n_tokens, result.sample_rate)
        audio_s = result.n_tokens * 320 / result.sample_rate if result.n_tokens else 0
        rtf     = total_s / audio_s if audio_s > 0 else 0.0

        await ws.send(json.dumps({
            "type":       "audio",
            "call_id":    call_id,
            "text_id":    text_id,
            "text":       text,
            "sample_rate": result.sample_rate,
            "wav_bytes":  len(result.wav_bytes),
            "is_final":   True,
            "llm_s":      result.llm_s,
            "decode_s":   result.decode_s,
            "rtf":        round(rtf, 3),
        }))
        await ws.send(result.wav_bytes)

        record_call(
            call_id=call_id, text_id=text_id, port=port, text=text,
            token_count=result.n_tokens, llm_s=result.llm_s, decode_s=result.decode_s,
            wav_bytes=len(result.wav_bytes), ts=ts_done,
        )
        record_ws_done(
            call_id, port=port, text_id=text_id, token_count=result.n_tokens,
            llm_ms=round(result.llm_s * 1000), decode_ms=round(result.decode_s * 1000),
            total_ms=round(total_s * 1000), wav_bytes=len(result.wav_bytes),
            ts_text_recv=ts_text_recv, ts_llm_start=ts_llm_start,
            ts_tokens_ready=ts_done, ts_audio_sent=ts_done,
        )

        print(
            f"[{ts_done}] :{port} {call_id}  done"
            f"  llm={round(result.llm_s*1000)}ms"
            f"  decode={round(result.decode_s*1000)}ms"
            f"  total={round(total_s*1000)}ms"
            f"  tokens={result.n_tokens}"
            f"  wav={len(result.wav_bytes)}B"
            f"  rtf={rtf:.3f}  avg_rtf={avg_rtf:.3f}",
            flush=True,
        )

    except Exception as e:
        ts_err = _tsms()
        print(f"[{ts_err}] :{port} {call_id}  ERROR: {e}", flush=True)
        record_ws_error(call_id, port=port, text_id=text_id, error=str(e))
        await ws.send(json.dumps({
            "type": "error", "call_id": call_id, "text_id": text_id, "error": str(e),
        }))


# ── connection handler ───────────────────────────────────────────────────────

async def handle_connection(ws: websockets.ServerConnection, port: int) -> None:
    peer     = ws.remote_address
    conn_id  = f"{peer[0]}:{peer[1]}"
    print(f"[{_ts()}] :{port} connected  peer={conn_id}", flush=True)
    record_ws_connection_open(conn_id, port=port)

    try:
        async for raw in ws:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
                continue

            text    = (data.get("text") or "").strip()
            call_id = data.get("call_id") or conn_id
            text_id = data.get("text_id") or str(uuid.uuid4())

            if not text:
                await ws.send(json.dumps({
                    "type": "error", "call_id": call_id, "text_id": text_id,
                    "error": "Missing text",
                }))
                continue

            text      = normalize_text(text)
            streaming = bool(data.get("streaming", settings.streaming.enabled))

            ts_text_recv = _tsms()
            _log(f"{ts_text_recv}  RECV port={port}  text_id={text_id}"
                 f"  call_id={call_id}  streaming={streaming}  text={text[:60]!r}")
            print(f"[{ts_text_recv}] :{port} {call_id}"
                  f"  {'stream' if streaming else 'req'}  {text[:60]!r}", flush=True)

            synth = await get_synthesizer()

            if streaming:
                await _handle_streaming(ws, synth, text, call_id, text_id, port)
            else:
                await _handle_request(ws, synth, text, call_id, text_id, port, ts_text_recv)

    except WebSocketException:
        pass
    except Exception as e:
        print(f"[{_ts()}] :{port} connection error: {e}", flush=True)
    finally:
        record_ws_connection_close(conn_id, port=port)
        print(f"[{_ts()}] :{port} disconnected  peer={conn_id}", flush=True)


# ── warmup ───────────────────────────────────────────────────────────────────

_WARMUP_SENTENCES = [
    "नमस्ते, मैं आपकी कैसे मदद कर सकती हूं?",
    "क्या आप अपना नाम बता सकते हैं?",
    "कृपया थोड़ा इंतज़ार करें.",
    "आपकी समस्या हल हो गई है.",
    "हम जल्द ही आपसे संपर्क करेंगे.",
    "आपका भुगतान सफलतापूर्वक हो गया है.",
    "नमस्ते. मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से. क्या मैं customer name से बात कर रही हूं?",
    "आपके loan की किस्त ₹३,७५० अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "आपकी loan application approve हो गई है और Rs. 50000 सीधे आपके bank account 7890123456 में transfer कर दिए जाएंगे.",
    "हमारी company की policy के अनुसार अगर payment 30 दिनों के अंदर नहीं होती तो आपके credit score पर असर पड़ सकता है.",
]


async def _warmup(synth: BaseSynthesizer) -> None:
    """Synthesize a batch of sentences concurrently to prime GPU + CUDA graphs."""
    batch_size = 40
    sentences  = [_WARMUP_SENTENCES[i % len(_WARMUP_SENTENCES)] for i in range(batch_size)]
    print(f"[{_ts()}] warmup: {batch_size} sentences concurrent...", flush=True)
    t0 = time.perf_counter()

    async def _one(s: str) -> bool:
        try:
            await synth.synthesize(normalize_text(s))
            return True
        except Exception as e:
            print(f"[{_ts()}] warmup sentence failed: {e}", flush=True)
            return False

    ok = sum(await asyncio.gather(*[_one(s) for s in sentences]))
    print(f"[{_ts()}] warmup done  {ok}/{batch_size} ok  ({(time.perf_counter()-t0)*1000:.0f}ms)", flush=True)


async def _warmup_port(port: int, synth: BaseSynthesizer) -> None:
    sentence = normalize_text(_WARMUP_SENTENCES[0])
    url = f"ws://127.0.0.1:{port}/ws/warmup"
    try:
        async with websockets.connect(url, ping_interval=None,
                                      max_size=100 * 1024 * 1024, open_timeout=10) as ws:
            await ws.send(json.dumps({"call_id": "warmup", "text_id": "warmup",
                                      "text": sentence, "streaming": False}))
            await ws.recv(); await ws.recv()   # JSON metadata + WAV bytes
    except Exception as e:
        print(f"[{_ts()}] warmup port {port} failed: {e}", flush=True)


async def _warmup_all_ports(ports: list[int], synth: BaseSynthesizer) -> None:
    if not ports:
        return
    print(f"[{_ts()}] warming up {len(ports)} port(s)...", flush=True)
    t0 = time.perf_counter()
    await asyncio.gather(*[_warmup_port(p, synth) for p in ports])
    print(f"[{_ts()}] all ports warmed up  ({(time.perf_counter()-t0)*1000:.0f}ms)", flush=True)


# ── port management ──────────────────────────────────────────────────────────

async def _bind_ws_port(port: int) -> bool:
    if port in _open_ports:
        return False
    async def handler(ws: websockets.ServerConnection, p: int = port) -> None:
        await handle_connection(ws, p)
    await websockets.serve(handler, "0.0.0.0", port,
                           ping_interval=30, ping_timeout=30,
                           max_size=100 * 1024 * 1024)
    _open_ports.add(port)
    record_port_change(_open_ports)
    print(f"[{_ts()}] opened ws://0.0.0.0:{port}", flush=True)
    return True


# ── HTTP control API ─────────────────────────────────────────────────────────

async def _http_add_port(req: web.Request) -> web.Response:
    try:
        port = int(req.rel_url.query["port"])
    except (KeyError, ValueError):
        return web.Response(status=400, text="missing or invalid ?port=N")
    if not (1024 <= port <= 65535):
        return web.Response(status=400, text="port out of range")
    already = port in _open_ports
    if not already:
        await _bind_ws_port(port)
    return web.json_response({"port": port, "opened": not already})


async def _http_list_ports(req: web.Request) -> web.Response:
    return web.json_response({"ports": sorted(_open_ports)})


async def _http_ready(req: web.Request) -> web.Response:
    from flowtts.synthesis import engine as _eng  # noqa: PLC0415
    ready = _eng._synthesizer is not None
    if not ready:
        return web.Response(status=503, text="loading")
    return web.json_response({"ready": True, "ports": sorted(_open_ports)})


async def _http_ws_log(req: web.Request) -> web.Response:
    return web.json_response(ws_log_snapshot())


async def _http_metrics(req: web.Request) -> web.Response:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # noqa: PLC0415
    ct = CONTENT_TYPE_LATEST.split(";")[0].strip()
    return web.Response(body=generate_latest(), content_type=ct)


async def _run_control_api(ctrl_port: int) -> None:
    app = web.Application()
    app.router.add_post("/ports/add", _http_add_port)
    app.router.add_get("/ports",      _http_list_ports)
    app.router.add_get("/ready",      _http_ready)
    app.router.add_get("/metrics",    _http_metrics)
    app.router.add_get("/ws/log",     _http_ws_log)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", ctrl_port)
    await site.start()
    print(f"[{_ts()}] control API  http://127.0.0.1:{ctrl_port}", flush=True)


# ── main server coroutine ─────────────────────────────────────────────────────

async def run_server(base_port: int, n_ports: int, ctrl_port: int | None = None) -> None:
    print(f"[{_ts()}] model_type={settings.model_type}", flush=True)

    synth = await get_synthesizer()

    # Mira needs a concurrent warmup batch to prime sglang + CUDA graphs.
    # VoxCPM already warmed up inside initialize().
    if settings.model_type == "mira":
        await _warmup(synth)

    if ctrl_port:
        await _run_control_api(ctrl_port)

    initial_ports = [base_port + i for i in range(n_ports)]
    for p in initial_ports:
        await _bind_ws_port(p)

    await _warmup_all_ports(initial_ports, synth)

    print(f"\n[{_ts()}] FlowTTS [{settings.model_type}]  {len(_open_ports)} port(s) ready:", flush=True)
    for p in sorted(_open_ports):
        print(f"  ws://0.0.0.0:{p}", flush=True)
    print(flush=True)

    await asyncio.Future()   # run forever


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    global _audio_out_dir, _llm_log_file

    parser = argparse.ArgumentParser(description="FlowTTS single-process WebSocket server")
    parser.add_argument("--base-port", type=int, default=settings.ws.port)
    parser.add_argument("--ports",     type=int, default=1,
                        help="Number of WS ports to open (default: 1)")
    parser.add_argument("--save-audio", type=str, default=None, metavar="DIR")
    parser.add_argument("--ctrl-port",  type=int, default=None, metavar="PORT")
    args = parser.parse_args()

    _llm_log_file = open(_llm_log, "w", buffering=1)
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
