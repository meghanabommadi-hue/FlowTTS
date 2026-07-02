#!/usr/bin/env python3
"""
Pipeline position: SINGLE-PROCESS GATEWAY (primary production entry point via run.sh).

Role in pipeline:
  Self-contained TTS server — no Redis, no worker process, no uvicorn per port.
  Loads k2-fsa/OmniVoice once, then handles all WebSocket ports in one asyncio
  event loop. This is the recommended way to run FlowTTS in production.

  Client
    │  WebSocket (text) on port 8080…8080+N
    ▼
  server.py  (one process, one GPU load)
    │  synthesizer.synthesize_stream(text, voice_id)  [OmniVoice, dynamic batching]
    │  → 24 kHz waveform chunks → int16 PCM
    ▼
  Client  (audio_chunk binary frames … audio_done JSON)

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
    ./run.sh --ports 100              # 100 ports: 8080…8864
    ./run.sh --ports 3 --port 9000   # ports 9000, 9001, 9002

Direct:
    python -m flowtts.server --ports 3
    python -m flowtts.server --ports 100 --base-port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import datetime
import hashlib
import base64
import json
import gc
import re
import sys
import tempfile
import time

import numpy as np
import uuid
from pathlib import Path

import logging
from aiohttp import web

import websockets
from websockets.exceptions import WebSocketException, ConnectionClosed
from websockets.http11 import Response as WsResponse, Headers as WsHeaders

from flowtts.core.config import settings
from flowtts.decoder.decoder import tensor_to_wav, pcm_to_int16_bytes, SAMPLE_RATE
from flowtts.monitoring.metrics import (
    record_call, record_ws_connection_open, record_ws_connection_close,
    record_ws_error, record_ws_done, ws_log_snapshot, record_port_change,
    register_gpu_info, snapshot_metrics,
)
from flowtts.processing.audio_processing import crossfade, fade_in, fade_out, resample_audio

from flowtts.synthesis.models import OmniVoiceSynthesizer

# Silence websockets' own logger — we do our own prints
logging.getLogger("websockets").setLevel(logging.CRITICAL)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

_synthesizer: OmniVoiceSynthesizer | None = None
_wav_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="wav_enc")

# OOM recovery state
_oom_recovery_active: bool = False
_restarting: bool = False
_OOM_RECOVERY_WINDOW_S: float = 5.0

# Concurrency limiter — queue requests beyond this threshold instead of running them immediately
_MAX_ACTIVE_REQUESTS: int = 200
_active_requests: int = 0
_request_semaphore: asyncio.Semaphore | None = None  # initialised in run_server()

# Stale connection reaping — close WS connections idle for longer than this
_WS_IDLE_TIMEOUT_S = 300  # 10 minutes
_ws_last_activity: dict[str, float] = {}   # conn_id → time.monotonic() of last request
_ws_connections: dict[str, "websockets.ServerConnection"] = {}  # conn_id → live ws object
_ws_cancel_events: dict[str, asyncio.Event] = {}  # conn_id → set when client sends cancel
_CROSSFADE_SAMPLES   = settings.streaming.crossfade_samples
_FADE_OUT_SAMPLES    = settings.streaming.fade_out_samples
_OUTPUT_SR           = settings.output.sample_rate
_NATIVE_SR           = SAMPLE_RATE   # updated to engine.sampling_rate after model load
_audio_out_dir: Path | None = None
_wav_cache_dir: Path | None = None
_wav_cache_base: Path = Path.home() / "FlowTTS"

# Explicit per-voice cache directory map. Keys match voice_id sent by clients.
_VOICE_CACHE_MAP: dict[str, str] = {
    "tara":    "cached_data_tara_cache_full",
    "simran":  "cached_data_simran_june",
    "vikram":       "cached_data_vikram",
    "daya":         "cached_data_daya",
    "vanita":       "cached_data_vanita",
    "sunita":       "cached_data_sunita",
    "anika": "cached_data_anika",
    "anika2": "cached_data_anika2",
    "zara": "cached_data_zara",
    "saavi": "cached_data_saavi",
    "monika": "cached_data_monika",
    "gargi": "cached_data_gargi",
}


def _cache_dir_for_voice(voice_id: str | None) -> Path | None:
    """Return per-voice cache dir if known, else fall back to global _wav_cache_dir."""
    if voice_id and voice_id in _VOICE_CACHE_MAP:
        d = _wav_cache_base / _VOICE_CACHE_MAP[voice_id]
        if d.exists():
            return d
    return _wav_cache_dir
_open_ports: set[int] = set()  # tracks all bound WS ports

# Rolling RTF tracking (thread-safe via GIL for simple int/float ops)
_rtf_count: int   = 0
_rtf_sum:   float = 0.0


def _record_rtf(total_s: float, audio_s: float) -> float:
    """Record RTF for one request, return current avg RTF."""
    global _rtf_count, _rtf_sum
    if audio_s <= 0:
        return 0.0
    rtf = total_s / audio_s
    _rtf_count += 1
    _rtf_sum   += rtf
    return _rtf_sum / _rtf_count


def _wav_to_output_pcm(wav: np.ndarray) -> bytes:
    """Resample a native-rate float32 waveform to the output rate and encode int16."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if _OUTPUT_SR != _NATIVE_SR and wav.size:
        wav = resample_audio(wav, _NATIVE_SR, _OUTPUT_SR)
    return pcm_to_int16_bytes(wav)
_llm_log: Path = Path(__file__).parents[1] / "llm.log"
_llm_log_file = None  # opened once in main()
_llm_out_log: Path = Path(__file__).parents[1] / "monitoring" / "llm_outputs.jsonl"
_llm_out_log_file = None  # opened once in main()


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _tsms() -> str:
    """Current time as HH:MM:SS.mmm (millisecond precision)."""
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log(line: str) -> None:
    """Append a timestamped line to llm.log (no-op if file not open)."""
    if _llm_log_file is not None:
        _llm_log_file.write(line + "\n")
        _llm_log_file.flush()


async def _get_synthesizer() -> OmniVoiceSynthesizer:
    global _synthesizer, _NATIVE_SR
    if _synthesizer is None:
        _synthesizer = OmniVoiceSynthesizer()
        print(f"[{_ts()}] loading OmniVoice...", flush=True)
        await _synthesizer.initialize()
        _NATIVE_SR = _synthesizer.sampling_rate
        print(f"[{_ts()}] model ready  native_sr={_NATIVE_SR}  out_sr={_OUTPUT_SR}", flush=True)
        # Register engine metadata in Prometheus once model is loaded
        ei = _synthesizer.engine_info
        register_gpu_info(
            model_gpu_id=0,
            decoder_gpu_id=0,
            tp_size=1,
            attention_backend=f"omnivoice(num_step={ei.get('num_step')},gs={ei.get('guidance_scale')})",
            mem_weight_gb=str(ei.get("frame_rate", "n/a")),
            mem_kvcache_gb="n/a",
        )
    return _synthesizer


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


async def _handle_oom(exc: BaseException) -> None:
    global _oom_recovery_active, _restarting
    if _restarting:
        return
    ts = _tsms()
    if not _oom_recovery_active:
        # Stage 1: soft recovery — free GPU caches, block new requests briefly
        print(f"[{ts}] OOM detected — clearing GPU cache", flush=True)
        _oom_recovery_active = True
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        async def _reset_oom() -> None:
            await asyncio.sleep(_OOM_RECOVERY_WINDOW_S)
            global _oom_recovery_active
            if not _restarting:
                _oom_recovery_active = False
                print(f"[{_ts()}] OOM recovery window expired — resuming", flush=True)
        asyncio.create_task(_reset_oom())
    else:
        # Stage 2: soft recovery was insufficient — trigger process restart
        print(f"[{ts}] OOM during recovery window — scheduling restart", flush=True)
        _restarting = True

        async def _deferred_exit() -> None:
            await asyncio.sleep(1.0)  # allow in-flight ws.send() to flush
            print(f"[{_ts()}] Restarting process (OOM recovery failed)", flush=True)
            sys.exit(1)
        asyncio.create_task(_deferred_exit())


async def _handle_streaming_request(
    ws: websockets.ServerConnection,
    synth: "OmniVoiceSynthesizer",
    text: str,
    call_id: str,
    text_id: str,
    port: int,
    ts_text_recv: str,
    voice_id: str | None = None,
    speed: float | None = None,
    language: str | None = None,
    cancel_event: asyncio.Event | None = None,
) -> None:
    """Stream audio chunks to the client as OmniVoice completes each text chunk.

    OmniVoice is non-autoregressive so it cannot emit audio token-by-token. The
    synthesizer splits the text into chunks (short first chunk for low TTFB),
    dispatches them all to the dynamic batch queue at once, and yields each
    chunk's waveform IN ORDER as it finishes. We resample to the output rate,
    smooth the boundary, and send each as one binary frame:
      - JSON header { type:"audio_chunk", chunk_index, call_id, text_id, is_final }
      - raw int16 PCM bytes appended to the same frame
    A final { type:"audio_done", ... } JSON frame follows.
    """
    t0           = time.perf_counter()
    ts_gen_start = _tsms()
    _log(f"{ts_gen_start}  IN   port={port}  text_id={text_id}  call_id={call_id}  text={text}")

    _XFADE       = _CROSSFADE_SAMPLES   # boundary fade-in length (samples @ native rate)
    _FADEOUT     = _FADE_OUT_SAMPLES    # final-chunk fade-out length
    chunk_index  = 0
    total_samples = 0                   # samples at OUTPUT rate
    total_wav_b  = 0
    enc_total    = 0.0
    first_chunk_sent = False
    gen_ttft_ms: int | None = None      # ms from t0 to first audio chunk sent

    loop = asyncio.get_event_loop()

    def _prep_pcm(wav: np.ndarray, is_final: bool, first: bool) -> bytes:
        """Fade the boundary (native rate), resample to output rate, encode int16."""
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if wav.size:
            wav = wav - wav.mean()  # DC-offset removal avoids boundary clicks
        if (not first) and _XFADE > 0 and wav.size:
            wav = fade_in(wav.copy(), min(_XFADE, len(wav)))
        if is_final and _FADEOUT > 0 and len(wav) > _FADEOUT:
            wav = fade_out(wav.copy(), _FADEOUT)
        return _wav_to_output_pcm(wav)

    try:
        async for idx, wav, is_final in synth.synthesize_stream(
            text, voice_id=voice_id, speed=speed, language=language
        ):
            if cancel_event and cancel_event.is_set():
                print(f"[{_ts()}] :{port} {call_id}  cancelled  text_id={text_id}", flush=True)
                await ws.send(json.dumps({"type": "cancelled", "call_id": call_id, "text_id": text_id}))
                await ws.send(json.dumps({"type": "audio_done", "call_id": call_id, "text_id": text_id,
                                          "total_tokens": total_samples,
                                          "total_wav_bytes": total_wav_b, "sample_rate": _OUTPUT_SR}))
                return

            first = not first_chunk_sent
            audio_bytes = await loop.run_in_executor(_wav_executor, _prep_pcm, wav, is_final, first)
            n_samp = len(audio_bytes) // 2
            total_samples += n_samp
            total_wav_b += len(audio_bytes)

            ts_chunk = _tsms()
            if first:
                gen_ttft_ms = round((time.perf_counter() - t0) * 1000)
                print(
                    f"[{ts_chunk}] :{port} {call_id}  first_chunk"
                    f"  ttft={gen_ttft_ms}ms  samples={n_samp}",
                    flush=True,
                )
                first_chunk_sent = True

            await ws.send(
                json.dumps({
                    "type":        "audio_chunk",
                    "call_id":     call_id,
                    "text_id":     text_id,
                    "chunk_index": chunk_index,
                    "sample_rate": _OUTPUT_SR,
                    "encoding":    "pcm_int16",
                    "wav_bytes":   len(audio_bytes),
                    "tokens":      n_samp,
                    "is_final":    is_final,
                    "cache_hit":   False,
                }).encode() + audio_bytes
            )
            chunk_index += 1

        total_s = round(time.perf_counter() - t0, 4)
        gen_ms  = round(total_s * 1000)
        ts_done = _tsms()
        _log(f"{ts_done}  OUT  port={port}  text_id={text_id}  call_id={call_id}  gen_ms={gen_ms}")

        audio_s = total_samples / _OUTPUT_SR if _OUTPUT_SR else 0.0
        avg_rtf = _record_rtf(total_s, audio_s)
        rtf     = total_s / audio_s if audio_s > 0 else 0.0

        await ws.send(json.dumps({
            "type":            "audio_done",
            "call_id":         call_id,
            "text_id":         text_id,
            "text":            text,
            "chunks":          chunk_index,
            "total_tokens":    total_samples,
            "total_wav_bytes": total_wav_b,
            "sample_rate":     _OUTPUT_SR,
            "llm_s":           total_s,     # wall-clock generation time (batched/pipelined)
            "decode_s":        0.0,         # codec decode is inside generate(); not separable
            "llm_ttft_ms":     gen_ttft_ms,
            "decoder_ttft_ms": gen_ttft_ms,
            "rtf":             round(rtf, 3),
            "avg_rtf":         round(avg_rtf, 3),
        }))

        record_call(
            call_id=call_id,
            text_id=text_id,
            port=port,
            text=text,
            # token_count kept as a 50 fps audio-frame proxy so the metrics
            # short-audio alarm (audio_s = tokens*320/16000) stays meaningful.
            token_count=int(round(audio_s * 50)),
            llm_s=total_s,
            decode_s=0.0,
            wav_bytes=total_wav_b,
            ts=ts_done,
            voice_id=voice_id,
            cache_hit=False,
        )

        print(
            f"[{ts_done}] :{port} {call_id}  stream_done"
            f"  chunks={chunk_index}"
            f"  audio_s={audio_s:.2f}"
            f"  ttft={gen_ttft_ms}ms"
            f"  gen={gen_ms}ms"
            f"  wav={total_wav_b}B"
            f"  rtf={rtf:.3f}",
            flush=True,
        )

    except ConnectionClosed:
        # Client hung up (e.g. code 1000 OK) mid-stream — a normal disconnect, not an error.
        print(f"[{_ts()}] :{port} {call_id}  client closed  text_id={text_id}", flush=True)
        return
    except Exception as e:
        ts_err = _tsms()
        print(f"[{ts_err}] :{port} {call_id}  STREAM ERROR: {e}", flush=True)
        record_ws_error(call_id, port=port, text_id=text_id, error=str(e), voice_id=voice_id)
        if _is_oom(e):
            await _handle_oom(e)
        try:
            await ws.send(json.dumps({
                "type": "error", "call_id": call_id, "text_id": text_id, "error": str(e),
            }))
        except Exception:
            pass


async def handle_connection(ws: websockets.ServerConnection, port: int) -> None:
    """Handle one persistent WebSocket connection (one call = one socket)."""
    peer = ws.remote_address
    conn_id = f"{peer[0]}:{peer[1]}"
    print(f"[{_ts()}] :{port} connected  peer={conn_id}", flush=True)
    record_ws_connection_open(conn_id, port=port)
    _ws_last_activity[conn_id] = time.monotonic()
    _ws_connections[conn_id] = ws
    _ws_cancel_events[conn_id] = asyncio.Event()

    try:
        async for raw in ws:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
                continue

            if data.get("type") == "cancel":
                cancel_ev = _ws_cancel_events.get(conn_id)
                if cancel_ev:
                    cancel_ev.set()
                print(f"[{_ts()}] :{port} {conn_id}  cancel  text_id={data.get('text_id')}", flush=True)
                continue

            text = (data.get("text") or "").strip()
            call_id  = data.get("call_id") or f"{peer[0]}:{peer[1]}"
            text_id  = data.get("text_id") or str(uuid.uuid4())
            voice_id = data.get("voice_id") or None
            speed    = data.get("speed")
            language = data.get("language") or None
            try:
                speed = float(speed) if speed is not None else None
            except (TypeError, ValueError):
                speed = None
            streaming = bool(data.get("streaming", settings.streaming.enabled))
            if not text:
                await ws.send(json.dumps({
                    "type": "error", "call_id": call_id, "text_id": text_id,
                    "error": "Missing text",
                }))
                continue

            # Cache lookup: per-voice cache dir first, then global fallback.
            # Bypass LLM entirely — send cached WAV immediately.
            _vc_dir = _cache_dir_for_voice(voice_id)
            if _vc_dir is not None:
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                cached_wav = _vc_dir / f"{text_hash}.wav"
                if cached_wav.exists():
                    wav_bytes = cached_wav.read_bytes()
                    print(f"[{_ts()}] :{port} call_id:{call_id}  cache_hit:True  voice:{voice_id or 'default'}  {text[:60]!r}", flush=True)
                    await ws.send(
                        json.dumps({
                            "type":        "audio_chunk",
                            "call_id":     call_id,
                            "text_id":     text_id,
                            "chunk_index": 0,
                            "sample_rate": SAMPLE_RATE,
                            "wav_bytes":   len(wav_bytes),
                            "tokens":      0,
                            "is_final":    True,
                            "cache_hit":   True,
                        }).encode() + wav_bytes
                    )
                    await ws.send(json.dumps({
                        "type":            "audio_done",
                        "call_id":         call_id,
                        "text_id":         text_id,
                        "text":            text,
                        "chunks":          1,
                        "total_tokens":    0,
                        "total_wav_bytes": len(wav_bytes),
                        "sample_rate":     SAMPLE_RATE,
                        "llm_s":           0.0,
                        "decode_s":        0.0,
                        "cache_hit":       True,
                    }))
                    record_call(
                        call_id=call_id,
                        text_id=text_id,
                        port=port,
                        text=text,
                        token_count=0,
                        llm_s=0.0,
                        decode_s=0.0,
                        wav_bytes=len(wav_bytes),
                        ts=_tsms(),
                        voice_id=voice_id,
                        cache_hit=True,
                    )
                    continue

            _ws_last_activity[conn_id] = time.monotonic()
            ts_text_recv = _tsms()
            _log(f"{ts_text_recv}  RECV port={port}  text_id={text_id}  call_id={call_id}  streaming={streaming}  text={text[:60]!r}")
            print(f"[{ts_text_recv}] :{port} {call_id}  {'stream' if streaming else 'req'}  {text[:60]!r}", flush=True)

            synth = await _get_synthesizer()

            # Reject new GPU work while OOM recovery or restart is in progress
            if _restarting or _oom_recovery_active:
                reason = "server restarting, try again shortly" if _restarting else "GPU memory recovery in progress, try again shortly"
                await ws.send(json.dumps({"type": "error", "call_id": call_id, "text_id": text_id, "error": reason}))
                if _restarting:
                    await ws.close(1001, "server restarting")
                continue

            # Reset cancel event before starting a new request
            cancel_ev = _ws_cancel_events.get(conn_id)
            if cancel_ev:
                cancel_ev.clear()

            # Queue request if at concurrency limit — waits here until a slot is free
            if _request_semaphore is not None:
                queue_depth = _MAX_ACTIVE_REQUESTS - _request_semaphore._value
                if queue_depth >= _MAX_ACTIVE_REQUESTS:
                    print(f"[{_tsms()}] :{port} {call_id}  queued (active={queue_depth})", flush=True)
                await _request_semaphore.acquire()

            if streaming:
                try:
                    await _handle_streaming_request(
                        ws, synth, text, call_id, text_id, port, ts_text_recv,
                        voice_id=voice_id, speed=speed, language=language, cancel_event=cancel_ev,
                    )
                finally:
                    if _request_semaphore is not None:
                        _request_semaphore.release()
                continue

            try:
                t0 = time.perf_counter()
                ts_gen_start = _tsms()
                _log(f"{ts_gen_start}  IN   port={port}  text_id={text_id}  call_id={call_id}  text={text}")
                # One batched generate() through the dynamic queue → full waveform.
                wav = await asyncio.wait_for(
                    synth.synthesize(text, voice_id=voice_id, speed=speed, language=language),
                    timeout=60.0,
                )
                gen_s = round(time.perf_counter() - t0, 4)
                gen_ms = round(gen_s * 1000)
                ts_ready = _tsms()
                _log(f"{ts_ready}  OUT  port={port}  text_id={text_id}  call_id={call_id}  gen_ms={gen_ms}")

                tw = time.perf_counter()
                audio_bytes = _wav_to_output_pcm(wav)
                enc_s = round(time.perf_counter() - tw, 4)
                n_samp = len(audio_bytes) // 2
                audio_s = n_samp / _OUTPUT_SR if _OUTPUT_SR else 0.0

                record_call(
                    call_id=call_id,
                    text_id=text_id,
                    port=port,
                    text=text,
                    token_count=int(round(audio_s * 50)),  # 50 fps audio-frame proxy for metrics
                    llm_s=gen_s,
                    decode_s=0.0,
                    wav_bytes=len(audio_bytes),
                    ts=ts_ready,
                    voice_id=voice_id,
                    cache_hit=False,
                )

                if _audio_out_dir is not None:
                    wav_file = _audio_out_dir / f"{text_id}.wav"
                    wav_file.write_bytes(tensor_to_wav(wav, sample_rate=_OUTPUT_SR).wav_bytes)
                    print(f"[{_ts()}] :{port}  saved → {wav_file}", flush=True)

                await ws.send(
                    json.dumps({
                        "type":        "audio_chunk",
                        "call_id":     call_id,
                        "text_id":     text_id,
                        "chunk_index": 0,
                        "sample_rate": _OUTPUT_SR,
                        "encoding":    "pcm_int16",
                        "wav_bytes":   len(audio_bytes),
                        "tokens":      n_samp,
                        "is_final":    True,
                        "cache_hit":   False,
                    }).encode() + audio_bytes
                )
                await ws.send(json.dumps({
                    "type":            "audio_done",
                    "call_id":         call_id,
                    "text_id":         text_id,
                    "text":            text,
                    "chunks":          1,
                    "total_tokens":    n_samp,
                    "total_wav_bytes": len(audio_bytes),
                    "sample_rate":     _OUTPUT_SR,
                    "llm_s":           gen_s,
                    "decode_s":        0.0,
                    "cache_hit":       False,
                }))
                ts_audio_sent = _tsms()

                total_s = gen_s + enc_s
                avg_rtf = _record_rtf(total_s, audio_s)
                rtf     = total_s / audio_s if audio_s > 0 else 0.0
                print(
                    f"[{ts_audio_sent}] :{port} {call_id}  done"
                    f"  gen={gen_ms}ms"
                    f"  enc={round(enc_s*1000)}ms"
                    f"  total={round(total_s*1000)}ms"
                    f"  audio_s={audio_s:.2f}"
                    f"  wav={len(audio_bytes)}B"
                    f"  rtf={rtf:.3f}"
                    f"  avg_rtf={avg_rtf:.3f}",
                    flush=True,
                )

                record_ws_done(
                    call_id,
                    port=port,
                    text_id=text_id,
                    token_count=int(round(audio_s * 50)),
                    llm_ms=gen_ms,
                    decode_ms=0,
                    total_ms=round(total_s * 1000),
                    wav_bytes=len(audio_bytes),
                    ts_text_recv=ts_text_recv,
                    ts_llm_start=ts_gen_start,
                    ts_tokens_ready=ts_ready,
                    ts_audio_sent=ts_audio_sent,
                )

            except Exception as e:
                ts_err = _tsms()
                print(f"[{ts_err}] :{port} {call_id}  ERROR: {e}", flush=True)
                record_ws_error(call_id, port=port, text_id=text_id, error=str(e), voice_id=voice_id)
                if _is_oom(e):
                    await _handle_oom(e)
                try:
                    await ws.send(json.dumps({
                        "type": "error", "call_id": call_id, "text_id": text_id,
                        "error": str(e),
                    }))
                except Exception as send_err:
                    print(f"[{_ts()}] :{port} {call_id}  error send failed (client gone): {send_err}", flush=True)

            finally:
                if _request_semaphore is not None:
                    _request_semaphore.release()

    except WebSocketException as e:
        print(f"[{_ts()}] :{port} {conn_id}  ws closed: {e}", flush=True)
    except Exception as e:
        print(f"[{_ts()}] :{port} connection error: {e}", flush=True)
    finally:
        _ws_last_activity.pop(conn_id, None)
        _ws_connections.pop(conn_id, None)
        _ws_cancel_events.pop(conn_id, None)
        record_ws_connection_close(conn_id, port=port)
        print(f"[{_ts()}] :{port} disconnected  peer={conn_id}", flush=True)


_WARMUP_SENTENCES = [
    # short
    "नमस्ते, मैं आपकी कैसे मदद कर सकती हूं?",
    "क्या आप अपना नाम बता सकते हैं?",
    "कृपया थोड़ा इंतज़ार करें.",
    "आपकी समस्या हल हो गई है.",
    "हम जल्द ही आपसे संपर्क करेंगे.",
    "आपका भुगतान सफलतापूर्वक हो गया है.",
    "क्या आप मुझे और जानकारी दे सकते हैं?",
    "आपका account number क्या है?",
    "क्या यह सही समय है बात करने का?",
    "मैं आपकी पूरी मदद करने के लिए यहाँ हूं.",
    # short with numbers
    "आपका खाता नंबर ९८७६५४३२१० है, कृपया confirm करें.",
    "आपका बकाया Rs. 2500 है, कृपया आज ही जमा करें.",
    "आपकी EMI Rs. 3750 हर महीने देय है.",
    "आपका बकाया ₹२,५०० है, कृपया आज ही जमा करें.",
    "आपका भुगतान ₹१०,००० सफलतापूर्वक हो गया है.",
    # medium
    "नमस्ते. मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से. क्या मैं customer name से बात कर रही हूं?",
    "आपके loan की किस्त अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "हमारे रिकॉर्ड के अनुसार आपका बकाया amount ₹५,००० है, कृपया जल्द से जल्द इसे जमा करें.",
    "आपकी EMI की due date निकल चुकी है, late charge से बचने के लिए आज ही payment करें.",
    "क्या आप अपनी personal details verify कर सकते हैं ताकि हम आपके account की जानकारी दे सकें?",
    "आपके loan की किस्त ₹३,७५० अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "आपकी EMI की due date ३० अप्रैल निकल चुकी है, late charge से बचने के लिए आज ही payment करें.",
    "आपके account नंबर ४५६७८९०१२३ पर ₹१५,००० का loan approve हुआ है, क्या आप details verify करेंगे?",
    "आपके loan की किस्त Rs. 3750 अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "आपके account number 4567890123 पर Rs. 15000 का loan approve हुआ है, क्या आप details verify करेंगे?",
    # medium — extra variety
    "आपकी loan application हमें मिल गई है और हम उसे process कर रहे हैं.",
    "क्या आप अपना registered mobile number confirm कर सकते हैं?",
    "आपकी next EMI की date 15 तारीख है, कृपया समय पर payment करें.",
    "हम आपको एक OTP भेज रहे हैं, कृपया उसे verify करें.",
    "आपका loan account successfully activate हो गया है.",
    # long
    "आपकी loan application approve हो गई है और ₹५०,००० सीधे आपके bank account ७८९०१२३४५६ में transfer कर दिए जाएंगे, जिसमें २ से ३ कार्य दिवस लग सकते हैं.",
    "हमारी company की policy के अनुसार अगर payment ३० दिनों के अंदर नहीं होती तो आपके credit score पर असर पड़ सकता है, इसलिए कृपया समय पर ₹८,२५० का भुगतान करें.",
    "आप हमारे mobile app के माध्यम से अपनी ₹४,५०० की EMI pay कर सकते हैं, इसके अलावा NEFT, IMPS, या UPI का भी उपयोग किया जा सकता है.",
    "आपकी loan application approve हो गई है और Rs. 50000 सीधे आपके bank account 7890123456 में transfer कर दिए जाएंगे, जिसमें 2 से 3 कार्य दिवस लग सकते हैं.",
    "हमारी company की policy के अनुसार अगर payment 30 दिनों के अंदर नहीं होती तो आपके credit score पर असर पड़ सकता है, इसलिए कृपया समय पर Rs. 8250 का भुगतान करें.",
    # long — extra variety
    "आपकी loan application approve हो गई है और loan amount सीधे आपके bank account में transfer कर दी जाएगी, जिसमें दो से तीन कार्य दिवस लग सकते हैं.",
    "आप हमारे mobile app के माध्यम से अपनी EMI pay कर सकते हैं, इसके अलावा NEFT, IMPS, या UPI का भी उपयोग किया जा सकता है, और किसी भी समस्या के लिए हमारी customer care team हमेशा available है.",
    "हमारे records के अनुसार आपका loan account number 1234567890 है और आपकी monthly EMI Rs. 4500 है जो हर महीने की 5 तारीख को deduct होती है.",
    "आपकी payment successfully receive हो गई है और आपका account अब up to date है, अगर आपको कोई और जानकारी चाहिए तो हमें call करें.",
]


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
    """Warm up every bound WS port concurrently.

    All warmup requests are coalesced by the OmniVoice engine's batch queue,
    so firing all ports simultaneously is safe.
    """
    sentence = settings.omnivoice.warmup_sentence
    if not sentence or not ports:
        return
    print(f"[{_ts()}] warming up {len(ports)} port(s) concurrently...", flush=True)
    t0 = time.perf_counter()
    await asyncio.gather(*[_warmup_port(p, sentence) for p in ports])
    print(f"[{_ts()}] all ports warmed up  ({(time.perf_counter()-t0)*1000:.0f}ms)", flush=True)


async def _bind_ws_port(port: int) -> bool:
    """Bind a new WebSocket listener on *port*. Returns False if already bound."""
    if port in _open_ports:
        return False
    async def handler(ws: websockets.ServerConnection, p: int = port) -> None:
        await handle_connection(ws, p)

    async def process_request(connection, request):
        if request.path == "/health":
            if _restarting:
                body = json.dumps({"status": "error", "reason": "server restarting"}).encode()
                status, phrase = 503, "Service Unavailable"
            elif _oom_recovery_active:
                body = json.dumps({"status": "error", "reason": "GPU OOM recovery in progress"}).encode()
                status, phrase = 503, "Service Unavailable"
            elif _synthesizer is None:
                body = json.dumps({"status": "error", "reason": "model loading"}).encode()
                status, phrase = 503, "Service Unavailable"
            else:
                body = json.dumps({"status": "ok", "ready": True}).encode()
                status, phrase = 200, "OK"
            headers = WsHeaders([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
            return WsResponse(status, phrase, headers, body)
        if request.path == "/metrics":
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            body = generate_latest()
            ct = CONTENT_TYPE_LATEST.split(";")[0].strip()
            headers = WsHeaders([("Content-Type", ct), ("Content-Length", str(len(body)))])
            return WsResponse(200, "OK", headers, body)

    await websockets.serve(
        handler, "0.0.0.0", port,
        ping_interval=30, ping_timeout=30,
        max_size=100 * 1024 * 1024,
        process_request=process_request,
    )
    _open_ports.add(port)
    record_port_change(_open_ports)
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
    if _restarting:
        return web.Response(status=503, text="restarting")
    return web.json_response({"ready": True, "ports": sorted(_open_ports), "oom_recovery": _oom_recovery_active})


async def _http_ws_log(req: web.Request) -> web.Response:
    """GET /ws/log  — last 20 WS events (open/done/error/close)."""
    return web.json_response(ws_log_snapshot())


async def _http_ws_active(req: web.Request) -> web.Response:
    """GET /ws/active  — currently active WebSocket connections."""
    snap = snapshot_metrics()
    return web.json_response(snap["ws"])


async def _http_metrics(req: web.Request) -> web.Response:
    """GET /metrics  — Prometheus scrape endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    ct = CONTENT_TYPE_LATEST.split(";")[0].strip()
    return web.Response(body=generate_latest(), content_type=ct)


async def _http_health(req: web.Request) -> web.Response:
    """GET /health  — liveness check. Returns 200 when ready, 503 otherwise."""
    if _restarting:
        return web.json_response({"status": "error", "reason": "server restarting"}, status=503)
    if _oom_recovery_active:
        return web.json_response({"status": "error", "reason": "GPU OOM recovery in progress"}, status=503)
    if _synthesizer is None:
        return web.json_response({"status": "error", "reason": "model loading"}, status=503)
    return web.json_response({"status": "ok", "ready": True})


_VOICE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


async def _http_clone_voice(req: web.Request) -> web.Response:
    """POST /voices — clone a voice from an uploaded reference clip (in-process; live).

    Accepts either:
      • multipart/form-data:  audio=<file>, voice_id=..., ref_text=..., preferred_lang=...
      • application/json:      {"voice_id","ref_text","preferred_lang","audio_base64"}

    ref_text is REQUIRED (no ASR). Returns JSON clone status; the voice is usable
    immediately (no restart) via the WebSocket `voice_id` field.
    """
    voice_id = ref_text = preferred_lang = None
    audio_bytes: bytes | None = None
    audio_name = "ref.wav"
    ctype = req.headers.get("Content-Type", "")
    try:
        if ctype.startswith("multipart/"):
            reader = await req.multipart()
            async for part in reader:
                if part.name == "audio":
                    audio_name = part.filename or audio_name
                    audio_bytes = await part.read(decode=False)
                elif part.name == "voice_id":
                    voice_id = (await part.read()).decode("utf-8").strip()
                elif part.name == "ref_text":
                    ref_text = (await part.read()).decode("utf-8")
                elif part.name in ("preferred_lang", "language"):
                    preferred_lang = (await part.read()).decode("utf-8").strip()
        else:
            data = await req.json()
            voice_id = (data.get("voice_id") or "").strip()
            ref_text = data.get("ref_text")
            preferred_lang = (data.get("preferred_lang") or data.get("language") or "").strip()
            b64 = data.get("audio_base64") or data.get("audio")
            if b64:
                audio_bytes = base64.b64decode(b64)
    except Exception as e:  # noqa: BLE001
        return web.json_response({"status": "error", "error": f"bad request: {e}"}, status=400)

    # ── validate ──
    if not voice_id or not _VOICE_ID_RE.match(voice_id):
        return web.json_response(
            {"status": "error", "error": "voice_id required and must match [A-Za-z0-9_-]{1,64}"}, status=400)
    if not ref_text or not ref_text.strip():
        return web.json_response({"status": "error", "error": "ref_text is required (no auto-transcription)"}, status=400)
    if not audio_bytes:
        return web.json_response(
            {"status": "error", "error": "audio required (multipart field 'audio' or json 'audio_base64')"}, status=400)
    if _restarting or _oom_recovery_active:
        return web.json_response({"status": "error", "error": "server busy (GPU recovery/restart)"}, status=503)

    suffix = Path(audio_name).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(prefix=f"clone_{voice_id}_", suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        synth = await _get_synthesizer()
        result = await synth.engine.create_voice(voice_id, tmp.name, ref_text, language=preferred_lang or None)
        print(f"[{_ts()}] voice cloned: {voice_id}  tokens={result['tokens']}  lang={preferred_lang or '-'}", flush=True)
        return web.json_response({"status": "ok", **result})
    except Exception as e:  # noqa: BLE001
        if _is_oom(e):
            await _handle_oom(e)
        print(f"[{_ts()}] voice clone FAILED {voice_id}: {e}", flush=True)
        return web.json_response({"status": "error", "voice_id": voice_id, "error": str(e)}, status=500)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


async def _http_list_voices(req: web.Request) -> web.Response:
    """GET /voices — list the aliases currently loaded."""
    if _synthesizer is None:
        return web.json_response({"status": "error", "error": "model loading"}, status=503)
    return web.json_response({"status": "ok", "voices": _synthesizer.engine.registry.aliases()})


async def _run_control_api(ctrl_port: int) -> None:
    app = web.Application(client_max_size=64 * 1024 * 1024)  # allow up to 64MB audio uploads
    app.router.add_post("/ports/add", _http_add_port)
    app.router.add_get("/ports",      _http_list_ports)
    app.router.add_get("/ready",      _http_ready)
    app.router.add_get("/health",     _http_health)
    app.router.add_get("/metrics",    _http_metrics)
    app.router.add_get("/ws/log",     _http_ws_log)
    app.router.add_get("/ws/active",  _http_ws_active)
    app.router.add_post("/voices",    _http_clone_voice)
    app.router.add_get("/voices",     _http_list_voices)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", ctrl_port)
    await site.start()
    print(f"[{_ts()}] control API  http://0.0.0.0:{ctrl_port}  (metrics: http://<public-ip>:{ctrl_port}/metrics)", flush=True)


async def run_server(base_port: int, n_ports: int, ctrl_port: int | None = None) -> None:
    global _request_semaphore
    _request_semaphore = asyncio.Semaphore(_MAX_ACTIVE_REQUESTS)

    # Start HTTP control API first so the port is claimed before warmup begins
    if ctrl_port:
        await _run_control_api(ctrl_port)

    # Load model once before binding ports (the engine warms itself in initialize())
    synth = await _get_synthesizer()

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

    asyncio.create_task(_stale_connection_reaper())

    await asyncio.Future()  # run forever


async def _stale_connection_reaper() -> None:
    """Close WebSocket connections that have been idle for _WS_IDLE_TIMEOUT_S seconds."""
    while True:
        await asyncio.sleep(60)  # check every minute
        now = time.monotonic()
        stale = [
            conn_id for conn_id, last in list(_ws_last_activity.items())
            if now - last > _WS_IDLE_TIMEOUT_S
        ]
        for conn_id in stale:
            ws = _ws_connections.get(conn_id)
            if ws is not None:
                print(f"[{_ts()}] stale connection closed  peer={conn_id}  idle={_WS_IDLE_TIMEOUT_S}s", flush=True)
                await ws.close(1001, "idle timeout")


def main() -> None:
    global _audio_out_dir

    parser = argparse.ArgumentParser(description="FlowTTS single-process WebSocket server")
    parser.add_argument("--base-port", type=int, default=settings.ws.port,
                        help=f"First port to bind (default: {settings.ws.port})")
    parser.add_argument("--ports", type=int, default=1,
                        help="Number of WebSocket ports to open (default: 1)")
    parser.add_argument("--save-audio", type=str, default=None, metavar="DIR",
                        help="Directory to save decoded WAV files (one per request)")
    parser.add_argument("--ctrl-port", type=int, default=None, metavar="PORT",
                        help="HTTP control API port for on-demand WS port binding (e.g. 8764)")
    args = parser.parse_args()

    global _llm_log_file, _llm_out_log_file
    _llm_log_file = open(_llm_log, "w", buffering=1)  # line-buffered, overwrites each run
    _llm_out_log.parent.mkdir(parents=True, exist_ok=True)
    _llm_out_log_file = open(_llm_out_log, "a", buffering=1)  # line-buffered, append across restarts

    if args.save_audio:
        _audio_out_dir = Path(args.save_audio)
        _audio_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[FlowTTS] Saving audio to {_audio_out_dir}/", flush=True)

    global _wav_cache_dir
    if settings.wav_cache_dir:
        _wav_cache_dir = Path(settings.wav_cache_dir)
        print(f"[FlowTTS] WAV cache: {_wav_cache_dir}/", flush=True)

    try:
        asyncio.run(run_server(args.base_port, args.ports, args.ctrl_port))
    except KeyboardInterrupt:
        print("\n[FlowTTS] Stopped.")


if __name__ == "__main__":
    main()
