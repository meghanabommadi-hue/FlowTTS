"""Pipeline position: REST API — OpenAI-compatible speech endpoint.

Role in pipeline:
  HTTP front door. Exists alongside (not instead of) the legacy WebSocket
  gateway: the WS path is what the existing voice-bot fleet speaks, and this is
  for load testing, for OpenAI-SDK clients, and for anything that wants a plain
  HTTP stream.

      POST /v1/audio/speech  ->  engine.synthesize_stream  ->  chunked PCM/WAV
      GET  /v1/models        ->  the one model we serve
      GET  /v1/languages     ->  27 languages + normalisation tier
      GET  /v1/stats, /metrics, /healthz, /readyz

Streaming WAV headers
---------------------
A streaming response cannot know its length up front, so the RIFF/data sizes are
written as 0xFFFFFFFF. Every streaming player accepts this (it is what ffmpeg
and most TTS APIs emit); the non-streaming path writes a correct header because
it can.

Client disconnects
------------------
At 200 RPS an abandoned stream that keeps rendering is pure waste. Every
streaming response watches for disconnect and fires the request's cancel event,
which propagates into `FlowScheduler.cancel()` and frees the arena slots
immediately.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import time

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from flowtts.dhvaani.api.models import (
    HealthResponse,
    LanguageInfo,
    LanguageListResponse,
    ModelCard,
    ModelListResponse,
    SpeechRequest,
)
from flowtts.dhvaani.config import dhv_settings
from flowtts.dhvaani.text import lang as langmod
from flowtts.dhvaani.types import DhvaaniError, SynthParams, new_request_id

logger = structlog.get_logger(__name__)

router = APIRouter()

_MEDIA = {
    "pcm": "audio/L16",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
}
_FFMPEG_FORMATS = {"mp3": ("mp3", "libmp3lame"), "opus": ("ogg", "libopus"),
                   "aac": ("adts", "aac"), "flac": ("flac", "flac")}


def wav_header(sample_rate: int, data_bytes: int | None, bits: int = 16, channels: int = 1) -> bytes:
    """44-byte RIFF header. `data_bytes=None` emits streaming placeholders."""
    streaming = data_bytes is None
    data_size = 0xFFFFFFFF if streaming else data_bytes
    riff_size = 0xFFFFFFFF if streaming else data_bytes + 36
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", data_size)
    )


def _engine(request: Request):
    eng = getattr(request.app.state, "engine", None)
    if eng is None or not eng.ready:
        raise HTTPException(status_code=503, detail="engine is not ready")
    return eng


async def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    keys = dhv_settings.server.api_keys
    if not keys:
        return
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    token = token or x_api_key
    if token not in keys:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _params(req: SpeechRequest) -> SynthParams:
    return SynthParams.from_settings(
        dhv_settings,
        speed=req.speed,
        num_step=req.num_step,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        output_sample_rate=req.sample_rate,
    )


async def _ffmpeg_encode(pcm: bytes, sample_rate: int, fmt: str) -> bytes:
    """Transcode raw s16le through ffmpeg. Returns b"" when ffmpeg is absent."""
    container, codec = _FFMPEG_FORMATS[fmt]
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-c:a", codec, "-f", container, "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return b""
    out, err = await proc.communicate(pcm)
    if proc.returncode != 0:
        raise DhvaaniError(f"ffmpeg failed encoding {fmt}: {err.decode()[:300]}")
    return out


# ---------------------------------------------------------------------------
# POST /v1/audio/speech
# ---------------------------------------------------------------------------
@router.post("/v1/audio/speech", dependencies=[Depends(require_api_key)])
async def create_speech(req: SpeechRequest, request: Request):
    eng = _engine(request)
    rid = new_request_id()
    params = _params(req)
    sr = params.output_sample_rate
    fmt = req.response_format
    accept = (request.headers.get("accept") or "").lower()
    sse = req.stream_format == "sse" or "text/event-stream" in accept
    streaming = req.stream or sse

    if fmt in _FFMPEG_FORMATS and streaming:
        raise HTTPException(
            status_code=400,
            detail=f"response_format={fmt} cannot be streamed; use pcm or wav, "
                   "or request stream=false",
        )

    cancel = asyncio.Event()

    async def _watch_disconnect():
        # An abandoned HTTP stream would otherwise keep occupying arena slots
        # and burning GPU for audio nobody will hear.
        try:
            while not cancel.is_set():
                if await request.is_disconnected():
                    cancel.set()
                    logger.info("client_disconnected", request_id=rid)
                    return
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            return

    headers = {"X-Request-Id": rid, "X-Sample-Rate": str(sr)}

    # ---- non-streaming --------------------------------------------------
    if not streaming:
        try:
            pcm, metrics = await eng.synthesize(
                req.input, req.voice, req.language, params, rid
            )
        except DhvaaniError as e:
            raise HTTPException(status_code=e.status_code, detail=str(e))
        headers["X-TTFB-Ms"] = str(round(metrics.ttfb_ms, 1))
        headers["X-Total-Ms"] = str(round(metrics.total_ms, 1))
        headers["X-Audio-Seconds"] = str(round(metrics.audio_s, 3))
        headers["X-RTF"] = str(round(metrics.rtf, 4))

        if fmt == "pcm":
            return Response(pcm, media_type=f"audio/L16; rate={sr}", headers=headers)
        if fmt == "wav":
            return Response(
                wav_header(sr, len(pcm)) + pcm, media_type="audio/wav", headers=headers
            )
        encoded = await _ffmpeg_encode(pcm, sr, fmt)
        if not encoded:
            raise HTTPException(
                status_code=400,
                detail=f"response_format={fmt} needs ffmpeg on PATH; it was not found. "
                       "Use pcm or wav.",
            )
        return Response(encoded, media_type=_MEDIA[fmt], headers=headers)

    # ---- streaming ------------------------------------------------------
    watcher = asyncio.create_task(_watch_disconnect())

    async def gen_audio():
        try:
            if fmt == "wav":
                yield wav_header(sr, None)
            async for chunk in eng.synthesize_stream(
                req.input, req.voice, req.language, params, rid, cancel
            ):
                if chunk.audio:
                    yield chunk.audio
        except DhvaaniError as e:
            logger.warning("stream_aborted", request_id=rid, error=str(e))
        except Exception as e:
            logger.exception("stream_failed", request_id=rid, error=str(e))
        finally:
            cancel.set()
            watcher.cancel()

    async def gen_sse():
        """OpenAI's SSE speech shape: base64 deltas, then a done event."""
        t0 = time.perf_counter()
        total = 0
        try:
            async for chunk in eng.synthesize_stream(
                req.input, req.voice, req.language, params, rid, cancel
            ):
                if not chunk.audio:
                    continue
                total += len(chunk.audio)
                payload = {
                    "type": "speech.audio.delta",
                    "audio": base64.b64encode(chunk.audio).decode("ascii"),
                }
                yield f"data: {json.dumps(payload)}\n\n"
            done = {
                "type": "speech.audio.done",
                "usage": {"input_characters": len(req.input)},
                "audio_bytes": total,
                "sample_rate": sr,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            cancel.set()
            watcher.cancel()

    if sse:
        headers["Cache-Control"] = "no-cache"
        headers["X-Accel-Buffering"] = "no"
        return StreamingResponse(gen_sse(), media_type="text/event-stream", headers=headers)

    media = "audio/wav" if fmt == "wav" else f"audio/L16; rate={sr}"
    headers["X-Accel-Buffering"] = "no"
    return StreamingResponse(gen_audio(), media_type=media, headers=headers)


# ---------------------------------------------------------------------------
# Metadata / ops
# ---------------------------------------------------------------------------
@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    return ModelListResponse(
        data=[ModelCard(id="dhvaani-0.5", created=int(time.time()))]
    )


@router.get("/v1/languages", response_model=LanguageListResponse)
async def list_languages():
    return LanguageListResponse(data=[LanguageInfo(**d) for d in langmod.describe_all()])


@router.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request):
    eng = getattr(request.app.state, "engine", None)
    if eng is None:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(status="error", ready=False, reason="no engine").model_dump(),
        )
    if not eng.ready:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(status="error", ready=False, reason="starting").model_dump(),
        )
    return HealthResponse(status="ok", ready=True, backend=eng.stats().get("backend"))


@router.get("/readyz", response_model=HealthResponse)
async def readyz(request: Request):
    return await healthz(request)


@router.get("/v1/stats")
async def stats(request: Request):
    eng = _engine(request)
    return eng.stats()


@router.get("/metrics")
async def metrics(request: Request):
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    from flowtts.dhvaani.monitoring.metrics import update_from_stats

    eng = getattr(request.app.state, "engine", None)
    if eng is not None and eng.ready:
        try:
            update_from_stats(eng.stats())
        except Exception:
            pass
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST.split(";")[0].strip())
