"""Pipeline position: WEBSOCKET GATEWAY — the production wire protocol.

Role in pipeline:
  Serves the EXACT protocol `flowtts/server.py` speaks, so the existing voice-bot
  fleet can be pointed at DhVaani without touching a single client. Only the
  engine behind it changes.

Wire format (unchanged from flowtts/server.py):
  client -> server   {"text": ..., "call_id": ..., "text_id": ..., "voice_id": ...,
                      "streaming": true}
                     {"type": "cancel", "text_id": ...}
                     {"type": "cache_hit", "source": "kv_store", ...}
  server -> client   ONE frame per chunk: JSON header bytes immediately followed
                     by the raw audio bytes --
                       json.dumps({...}).encode() + audio
                     then a terminal {"type": "audio_done", ...} JSON frame.
  HTTP on the same port: GET /health, GET /metrics.

Field mapping
-------------
The old protocol's names come from the autoregressive LLM + codec pipeline. They
are kept verbatim so dashboards and clients keep parsing, with these meanings:

    llm_s            -> seconds spent in the flow decoder (the ODE)
    decode_s         -> seconds spent in the vocoder
    tokens /
    total_tokens     -> mel frames generated (93.75 per second of audio)
    llm_ttft_ms      -> ms to the first span's mel
    decoder_ttft_ms  -> ms to the first PCM bytes (the real TTFB)
    chunks           -> number of audio frames sent

The WAV cache fast path from the legacy server is preserved: a sha256-named hit
under `cached_data_<voice>/` bypasses the GPU entirely, which for repetitive IVR
prompts is the cheapest request there is.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

import structlog
import websockets
from websockets.exceptions import WebSocketException
from websockets.http11 import Headers as WsHeaders
from websockets.http11 import Response as WsResponse

from flowtts.dhvaani.config import dhv_settings
from flowtts.dhvaani.types import DhvaaniError, SynthParams, new_request_id

logger = structlog.get_logger(__name__)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


class WebSocketGateway:
    """Multi-port WebSocket server speaking the legacy FlowTTS protocol."""

    def __init__(self, engine, settings=None, cache_base: Path | None = None,
                 voice_cache_map: dict[str, str] | None = None):
        self._s = settings or dhv_settings
        self._engine = engine
        self.open_ports: set[int] = set()
        self._last_activity: dict[str, float] = {}
        self._conns: dict[str, object] = {}
        self._cancels: dict[str, asyncio.Event] = {}
        self._reaper: asyncio.Task | None = None

        # Legacy per-voice WAV cache. Kept configurable rather than hardcoded as
        # it was in flowtts/server.py.
        self._cache_base = cache_base or (Path.home() / "FlowTTS")
        self._voice_cache_map = voice_cache_map or {}

        self._rtf_n = 0
        self._rtf_sum = 0.0
        self._active = 0

    # -- cache ---------------------------------------------------------------
    def _cache_dir(self, voice_id: str | None) -> Path | None:
        if voice_id and voice_id in self._voice_cache_map:
            d = self._cache_base / self._voice_cache_map[voice_id]
            if d.exists():
                return d
        return None

    def _cache_lookup(self, text: str, voice_id: str | None) -> bytes | None:
        d = self._cache_dir(voice_id)
        if d is None:
            return None
        f = d / f"{hashlib.sha256(text.encode()).hexdigest()}.wav"
        try:
            return f.read_bytes() if f.exists() else None
        except OSError:
            return None

    def _record_rtf(self, total_s: float, audio_s: float) -> float:
        if audio_s <= 0:
            return self._rtf_sum / self._rtf_n if self._rtf_n else 0.0
        self._rtf_n += 1
        self._rtf_sum += total_s / audio_s
        return self._rtf_sum / self._rtf_n

    # -- request handling ----------------------------------------------------
    async def _handle_synthesize(self, ws, data: dict, port: int, conn_id: str) -> None:
        from flowtts.monitoring.metrics import record_call, record_ws_error

        text = (data.get("text") or "").strip()
        call_id = data.get("call_id") or conn_id
        text_id = data.get("text_id") or str(uuid.uuid4())
        voice_id = data.get("voice_id") or None

        if not text:
            await ws.send(json.dumps({
                "type": "error", "call_id": call_id, "text_id": text_id,
                "error": "Missing text",
            }))
            return

        # --- WAV cache fast path: no GPU work at all ---
        cached = self._cache_lookup(text, voice_id)
        if cached is not None:
            await ws.send(
                json.dumps({
                    "type": "audio_chunk", "call_id": call_id, "text_id": text_id,
                    "chunk_index": 0, "sample_rate": self._s.audio.output_sample_rate,
                    "encoding": "pcm_int16", "wav_bytes": len(cached), "tokens": 0,
                    "is_final": True, "cache_hit": True,
                }).encode() + cached
            )
            await ws.send(json.dumps({
                "type": "audio_done", "call_id": call_id, "text_id": text_id,
                "text": text, "chunks": 1, "total_tokens": 0,
                "total_wav_bytes": len(cached),
                "sample_rate": self._s.audio.output_sample_rate,
                "llm_s": 0.0, "decode_s": 0.0, "cache_hit": True,
            }))
            logger.info("ws_cache_hit", call_id=call_id, voice=voice_id or "default")
            return

        cancel = self._cancels.setdefault(conn_id, asyncio.Event())
        cancel.clear()

        params = SynthParams.from_settings(
            self._s,
            speed=data.get("speed"),
            num_step=data.get("num_step"),
            guidance_scale=data.get("guidance_scale"),
            output_sample_rate=data.get("sample_rate"),
        )

        t0 = time.perf_counter()
        chunk_index = 0
        total_bytes = 0
        first_pcm_ms = None
        metrics = None
        self._active += 1
        try:
            async for chunk in self._engine.synthesize_stream(
                text, voice_id, data.get("language"), params,
                request_id=new_request_id(), cancel_event=cancel,
            ):
                if cancel.is_set():
                    await ws.send(json.dumps({
                        "type": "cancelled", "call_id": call_id, "text_id": text_id,
                    }))
                    break
                if not chunk.audio and not chunk.is_final:
                    continue
                if first_pcm_ms is None and chunk.audio:
                    first_pcm_ms = round((time.perf_counter() - t0) * 1000)
                total_bytes += len(chunk.audio)
                frames = len(chunk.audio) // 2  # int16 mono samples
                await ws.send(
                    json.dumps({
                        "type": "audio_chunk", "call_id": call_id, "text_id": text_id,
                        "chunk_index": chunk_index,
                        "sample_rate": chunk.sample_rate,
                        "encoding": chunk.encoding,
                        "wav_bytes": len(chunk.audio),
                        "tokens": frames,
                        "is_final": chunk.is_final,
                        "cache_hit": False,
                    }).encode() + chunk.audio
                )
                chunk_index += 1
                m = chunk.meta.get("metrics")
                if m is not None:
                    metrics = m

            total_s = time.perf_counter() - t0
            sr = params.output_sample_rate
            audio_s = (total_bytes / 2) / sr if sr else 0.0
            mel_frames = int(audio_s * 93.75)
            flow_s = round((metrics or {}).get("flow_ms", 0.0) / 1000.0, 4)
            vocode_s = round((metrics or {}).get("vocode_ms", 0.0) / 1000.0, 4)
            avg_rtf = self._record_rtf(total_s, audio_s)
            rtf = total_s / audio_s if audio_s > 0 else 0.0

            await ws.send(json.dumps({
                "type": "audio_done", "call_id": call_id, "text_id": text_id,
                "text": text, "chunks": chunk_index,
                "total_tokens": mel_frames, "total_wav_bytes": total_bytes,
                "sample_rate": sr,
                "llm_s": flow_s, "decode_s": vocode_s,
                "llm_ttft_ms": round((metrics or {}).get("ttfb_ms", 0.0)),
                "decoder_ttft_ms": first_pcm_ms,
                "rtf": round(rtf, 3), "avg_rtf": round(avg_rtf, 3),
                "cache_hit": False,
            }))

            record_call(
                call_id=call_id, text_id=text_id, port=port, text=text,
                token_count=mel_frames, llm_s=flow_s, decode_s=vocode_s,
                wav_bytes=total_bytes, ts=_ts(), voice_id=voice_id, cache_hit=False,
            )
            logger.info(
                "ws_stream_done", call_id=call_id, port=port, chunks=chunk_index,
                ttfb_ms=first_pcm_ms, total_ms=round(total_s * 1000),
                audio_s=round(audio_s, 2), rtf=round(rtf, 3),
            )
        except DhvaaniError as e:
            record_ws_error(call_id, port=port, text_id=text_id, error=str(e), voice_id=voice_id)
            await self._send_error(ws, call_id, text_id, str(e))
        except Exception as e:
            logger.exception("ws_stream_failed", call_id=call_id, error=str(e))
            record_ws_error(call_id, port=port, text_id=text_id, error=str(e), voice_id=voice_id)
            await self._send_error(ws, call_id, text_id, str(e))
        finally:
            self._active -= 1

    @staticmethod
    async def _send_error(ws, call_id, text_id, error: str) -> None:
        try:
            await ws.send(json.dumps({
                "type": "error", "call_id": call_id, "text_id": text_id, "error": error,
            }))
        except Exception:
            pass  # client already gone

    # -- connection ----------------------------------------------------------
    async def handle_connection(self, ws, port: int) -> None:
        from flowtts.monitoring.metrics import (
            record_db_cache_hit, record_ws_connection_close, record_ws_connection_open,
        )

        peer = ws.remote_address
        conn_id = f"{peer[0]}:{peer[1]}" if peer else uuid.uuid4().hex[:12]
        record_ws_connection_open(conn_id, port=port)
        self._last_activity[conn_id] = time.monotonic()
        self._conns[conn_id] = ws
        self._cancels[conn_id] = asyncio.Event()
        logger.info("ws_connected", peer=conn_id, port=port)

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
                    continue

                mtype = data.get("type")
                if mtype == "cancel":
                    ev = self._cancels.get(conn_id)
                    if ev:
                        ev.set()
                    continue
                if mtype == "cache_hit" and data.get("source") == "kv_store":
                    record_db_cache_hit(
                        call_id=data.get("call_id") or conn_id,
                        text_id=data.get("text_id") or str(uuid.uuid4()),
                        port=port,
                        voice_id=data.get("voice_id") or None,
                    )
                    continue

                self._last_activity[conn_id] = time.monotonic()
                await self._handle_synthesize(ws, data, port, conn_id)

        except WebSocketException as e:
            logger.info("ws_closed", peer=conn_id, port=port, reason=str(e)[:120])
        except Exception as e:
            logger.warning("ws_connection_error", peer=conn_id, error=str(e)[:200])
        finally:
            self._last_activity.pop(conn_id, None)
            self._conns.pop(conn_id, None)
            self._cancels.pop(conn_id, None)
            record_ws_connection_close(conn_id, port=port)
            logger.info("ws_disconnected", peer=conn_id, port=port)

    # -- HTTP on the WS port -------------------------------------------------
    def _process_request(self, connection, request):
        path = getattr(request, "path", "")
        if path == "/health":
            ready = bool(self._engine and self._engine.ready)
            body = json.dumps(
                {"status": "ok", "ready": True} if ready
                else {"status": "error", "reason": "engine not ready"}
            ).encode()
            status, phrase = (200, "OK") if ready else (503, "Service Unavailable")
            headers = WsHeaders([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ])
            return WsResponse(status, phrase, headers, body)
        if path == "/metrics":
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            body = generate_latest()
            headers = WsHeaders([
                ("Content-Type", CONTENT_TYPE_LATEST.split(";")[0].strip()),
                ("Content-Length", str(len(body))),
            ])
            return WsResponse(200, "OK", headers, body)
        return None

    # -- ports ---------------------------------------------------------------
    async def bind_port(self, port: int) -> bool:
        from flowtts.monitoring.metrics import record_port_change

        if port in self.open_ports:
            return False

        async def handler(ws, p: int = port):
            await self.handle_connection(ws, p)

        await websockets.serve(
            handler, self._s.server.host, port,
            ping_interval=self._s.server.ws_ping_interval_s,
            ping_timeout=self._s.server.ws_ping_interval_s,
            max_size=self._s.server.ws_max_message_bytes,
            process_request=self._process_request,
        )
        self.open_ports.add(port)
        record_port_change(self.open_ports)
        logger.info("ws_port_open", port=port)
        return True

    async def serve(self, ports: list[int]) -> None:
        for p in ports:
            await self.bind_port(p)
        if self._reaper is None:
            self._reaper = asyncio.create_task(self._reap(), name="dhvaani-ws-reaper")

    async def _reap(self) -> None:
        """Close connections idle past server.ws_idle_timeout_s.

        A voice bot that dies mid-call leaves a socket open forever otherwise,
        and at fleet scale those accumulate into real file-descriptor pressure.
        """
        timeout = self._s.server.ws_idle_timeout_s
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            for conn_id, last in list(self._last_activity.items()):
                if now - last > timeout:
                    ws = self._conns.get(conn_id)
                    if ws is not None:
                        logger.info("ws_idle_close", peer=conn_id, idle_s=round(now - last))
                        try:
                            await ws.close(1001, "idle timeout")
                        except Exception:
                            pass

    def stats(self) -> dict:
        return {
            "open_ports": sorted(self.open_ports),
            "connections": len(self._conns),
            "active_requests": self._active,
            "avg_rtf": round(self._rtf_sum / self._rtf_n, 3) if self._rtf_n else 0.0,
        }
