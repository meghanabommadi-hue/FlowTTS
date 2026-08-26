"""Pipeline position: HTTP + WebSocket GATEWAY (primary production surface).

Role in pipeline:
  One FastAPI app carrying every transport, so nginx proxies a single port:

      POST /v1/tts               JSON in, audio out (or base64 JSON)
      POST /v1/tts/stream        chunked audio, first bytes as soon as chunk 0 lands
      POST /v1/audio/speech      OpenAI-compatible (stream=true supported)
      GET  /v1/voices            list cloned voices
      POST /v1/voices            clone a voice (multipart or JSON), usable immediately
      DEL  /v1/voices/{id}       remove a voice
      POST /v1/voices/preview    one-shot clone + synthesize, nothing persisted
      GET  /v1/languages         languages, with which have full number support
      POST /v1/normalize         run the text preprocessor alone (debugging)
      GET  /v1/stats             engine + latency counters
      GET  /healthz /readyz      liveness / readiness
      GET  /metrics              Prometheus
      WS   /ws , /ws/{call_id}   the FlowTTS binary streaming protocol

  Every route funnels into the same ``service.synthesizer``, so all of them
  share one model load, one batch queue and one voice registry.

The streaming routes are the reason this server exists: they write the first
audio bytes the moment chunk 0 comes off the GPU, while later chunks are still
generating. Anything that buffers the response — including nginx's default
proxy_buffering — turns time-to-first-byte into time-to-last-byte, which is why
deploy/nginx/omnivoice.conf turns it off.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import tempfile
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import numpy as np
import structlog
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from flowtts.api import audio_io
from flowtts.api.models import (
    SpeechRequest,
    SynthesisMetadata,
    SynthesisRequest,
    SynthesisResponse,
    VoiceCloneRequest,
    VoiceInfo,
    VoiceListResponse,
)
from flowtts.api.service import service
from flowtts.core.config import settings
from flowtts.synthesis.models import StreamChunk
from flowtts.synthesis.omnivoice_engine import GenParams, is_silent

logger = structlog.get_logger(__name__)

MAX_REFERENCE_BYTES = 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
async def require_api_key(authorization: Optional[str] = Header(None),
                          x_api_key: Optional[str] = Header(None)) -> None:
    """No-op unless FLOWTTS_API_KEYS is set, then Bearer or X-API-Key is required."""
    if not settings.api_keys:
        return
    token = x_api_key
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    if token not in settings.api_keys:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


# ---------------------------------------------------------------------------
# Shared request handling
# ---------------------------------------------------------------------------
def _resolve_format(requested: str | None) -> str:
    return requested or settings.output.default_format


def _resolve_rate(requested: int | None) -> int:
    return requested or settings.output.sample_rate


def _out(wav: np.ndarray, target_rate: int) -> np.ndarray:
    """Resample an engine-rate waveform to the caller's requested rate."""
    return audio_io.resample(wav, service.sample_rate, target_rate)


async def _decode_reference(b64: str) -> Path:
    """Write a base64 reference clip to a temp file for the codec to read."""
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"reference_audio is not valid base64: {exc}")
    if not raw:
        raise HTTPException(status_code=400, detail="reference_audio is empty")
    if len(raw) > MAX_REFERENCE_BYTES:
        raise HTTPException(status_code=413, detail="reference_audio too large")
    handle = tempfile.NamedTemporaryFile(prefix="flowtts_ref_", suffix=".wav", delete=False)
    handle.write(raw)
    handle.close()
    return Path(handle.name)


async def _build_prompt(req: SynthesisRequest) -> tuple[Any, Path | None]:
    """Build an inline VoiceClonePrompt when the request carries reference audio."""
    if not req.reference_audio:
        return None, None
    if not req.reference_text or not req.reference_text.strip():
        raise HTTPException(
            status_code=400,
            detail="reference_text is required with reference_audio (this server does no ASR)",
        )
    path = await _decode_reference(req.reference_audio)
    try:
        prompt = await service.require_ready().engine.create_prompt(str(path), req.reference_text)
        return prompt, path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _synth_kwargs(req: SynthesisRequest, prompt: Any) -> dict:
    return {
        "voice_id": req.voice_id,
        "language": req.language,
        "instruct": req.instruct,
        "prompt": prompt,
        "params": GenParams.build(req.generation_overrides()),
        "normalizer_overrides": req.normalizer_overrides(),
    }


def _http_error(exc: Exception) -> HTTPException:
    """Map an internal failure onto a status code the caller can act on."""
    if isinstance(exc, HTTPException):
        return exc
    message = str(exc)
    if service.is_oom(exc):
        asyncio.create_task(service.handle_oom())
        return HTTPException(status_code=503, detail="GPU out of memory; retry shortly")
    if "still loading" in message or "restarting" in message or "recovery" in message:
        return HTTPException(status_code=503, detail=message)
    if isinstance(exc, (ValueError, KeyError)):
        return HTTPException(status_code=400, detail=message)
    logger.exception("request_failed")
    return HTTPException(status_code=500, detail=message)


# ---------------------------------------------------------------------------
# Synthesis routes
# ---------------------------------------------------------------------------
router = APIRouter()


async def _synthesize_whole(req: SynthesisRequest) -> tuple[bytes, str, str, SynthesisMetadata]:
    """Non-streaming path, with the WAV cache in front of it."""
    synthesizer = service.require_ready()
    fmt = _resolve_format(req.format)
    rate = _resolve_rate(req.sample_rate)
    overrides = req.generation_overrides()
    started = time.perf_counter()

    clean, language = synthesizer.prepare(
        req.text, req.language, normalizer_overrides=req.normalizer_overrides()
    )

    # Cache lookup only for the plain path: an inline reference clip or a
    # voice-design instruct is not addressable by text + voice_id alone.
    cache_key = None
    if not req.reference_audio and not req.instruct:
        cache_key = service.cache_key(clean, req.voice_id, language, overrides)
        cached = service.cache_lookup(cache_key)
        if cached is None and not overrides:
            cached = service.cache_lookup_legacy(req.text.strip())
        if cached is not None:
            # The cache stores WAV at the engine's native rate. Re-encode it to
            # whatever this caller asked for rather than handing back the stored
            # bytes: otherwise a request for 8 kHz PCM silently receives a 24 kHz
            # WAV whenever the text happens to be cached, which a telephony
            # pipeline downstream will happily play at the wrong speed.
            try:
                wav, cached_rate = audio_io.decode_wav(cached)
                wav = audio_io.resample(wav, cached_rate, rate)
                body, actual_fmt, content_type = audio_io.encode_audio(wav, rate, fmt)
            except Exception as exc:  # noqa: BLE001 — a corrupt entry must not fail the request
                logger.warning("wav_cache_decode_failed", error=str(exc), key=cache_key)
                body = None
            if body is not None:
                elapsed = round((time.perf_counter() - started) * 1000)
                meta = SynthesisMetadata(
                    sample_rate=rate, format=actual_fmt,
                    duration_seconds=round(len(wav) / rate, 3) if rate else 0.0,
                    chunks=1, language=language, voice_id=req.voice_id,
                    normalized_text=clean, total_ms=elapsed, ttfb_ms=elapsed,
                    cache_hit=True,
                )
                return body, actual_fmt, content_type, meta

    prompt, temp_ref = await _build_prompt(req)
    try:
        wav = await asyncio.wait_for(
            synthesizer.synthesize(req.text, chunked=req.chunked, **_synth_kwargs(req, prompt)),
            timeout=settings.server.request_timeout_s,
        )
    finally:
        if temp_ref is not None:
            temp_ref.unlink(missing_ok=True)

    if is_silent(wav):
        raise RuntimeError("synthesis produced no audible output for this text")

    wav = _out(wav, rate)
    body, actual_fmt, content_type = audio_io.encode_audio(wav, rate, fmt)

    total_ms = round((time.perf_counter() - started) * 1000)
    duration = len(wav) / rate if rate else 0.0
    if duration > 0:
        service.record_rtf(total_ms / 1000 / duration)

    if cache_key is not None and actual_fmt == "wav" and rate == service.sample_rate:
        service.cache_store(cache_key, body)

    meta = SynthesisMetadata(
        sample_rate=rate, format=actual_fmt, duration_seconds=round(duration, 3),
        chunks=len(synthesizer.chunk(clean)) if clean else 0,
        language=language, voice_id=req.voice_id, normalized_text=clean,
        total_ms=total_ms, ttfb_ms=total_ms,
        real_time_factor=round(total_ms / 1000 / duration, 3) if duration > 0 else None,
    )
    return body, actual_fmt, content_type, meta


async def _stream_audio(req: SynthesisRequest, started: float,
                        prompt: Any, temp_ref: Path | None,
                        first: "StreamChunk", rest) -> AsyncIterator[bytes]:
    """Chunked streaming body: emit chunk 0, then the rest as they land.

    Chunk 0 is produced by the caller before the response starts (see
    :func:`_open_stream`), so a failure there can still be reported as an HTTP
    error rather than as a 200 with an empty body.
    """
    fmt = _resolve_format(req.format)
    rate = _resolve_rate(req.sample_rate)

    if fmt != "pcm":
        yield audio_io.streaming_wav_header(rate)

    total_samples = 0
    try:
        audio = _out(first.audio, rate)
        total_samples += audio.size
        ttfb = (time.perf_counter() - started) * 1000
        service.record_ttfb(ttfb)
        logger.info("stream_ttfb", ms=round(ttfb), voice=req.voice_id,
                    chars=len(req.text))
        yield audio_io.to_pcm16(audio)

        async for chunk in rest:
            if not chunk.audio.size:
                continue
            audio = _out(chunk.audio, rate)
            total_samples += audio.size
            yield audio_io.to_pcm16(audio)
    except Exception as exc:  # noqa: BLE001
        service.counters["errors"] += 1
        if service.is_oom(exc):
            await service.handle_oom()
        # Chunk 0 already went out, so the status code is long gone; ending the
        # stream is the only signal left. Chunk-0 failures never reach here.
        logger.error("stream_failed_midway", error=str(exc), chars=len(req.text),
                     sent_samples=total_samples, exc_info=True)
        return
    finally:
        if temp_ref is not None:
            temp_ref.unlink(missing_ok=True)

    duration = total_samples / rate if rate else 0.0
    if duration > 0:
        service.record_rtf((time.perf_counter() - started) / duration)


async def _open_stream(req: SynthesisRequest):
    """Generate chunk 0, then hand back everything needed to stream the rest.

    Producing the first chunk before the response begins is what lets a failed
    synthesis return a real status code. It costs no latency: there is nothing
    to send until chunk 0 exists either way, so the client receives the headers
    and the first audio in the same breath.
    """
    synthesizer = service.require_ready()
    started = time.perf_counter()
    prompt, temp_ref = await _build_prompt(req)
    try:
        stream = synthesizer.synthesize_stream(req.text, **_synth_kwargs(req, prompt))
        first = None
        async for chunk in stream:
            # Audible content, not merely a non-zero length: trimming a silent
            # clip leaves a short non-empty array, and streaming that looks like
            # success to the caller while sounding like a dropped call.
            if not is_silent(chunk.audio):
                first = chunk
                break
        if first is None:
            raise RuntimeError(
                "synthesis produced no audible output for this text"
            )
        return started, prompt, temp_ref, first, stream
    except Exception:
        if temp_ref is not None:
            temp_ref.unlink(missing_ok=True)
        raise


@router.post("/v1/tts", dependencies=[Depends(require_api_key)],
             summary="Synthesize speech")
async def tts(req: SynthesisRequest, request: Request) -> Response:
    """Synthesize *text*. Returns audio bytes, or JSON+base64 if the client asks.

    Set ``stream: true`` (or POST to /v1/tts/stream) for chunked delivery.
    """
    if req.stream:
        return await tts_stream(req)

    service.counters["requests"] += 1
    try:
        async with service.slot():
            body, fmt, content_type, meta = await _synthesize_whole(req)
    except Exception as exc:  # noqa: BLE001
        service.counters["errors"] += 1
        raise _http_error(exc) from exc

    accept = (request.headers.get("accept") or "").lower()
    if "application/json" in accept:
        return JSONResponse(SynthesisResponse(
            audio_base64=base64.b64encode(body).decode("ascii"), metadata=meta,
        ).model_dump())

    return Response(
        content=body,
        media_type=content_type,
        headers={
            "X-Sample-Rate": str(meta.sample_rate),
            "X-Audio-Format": fmt,
            "X-Duration-Seconds": str(meta.duration_seconds),
            "X-Total-Ms": str(meta.total_ms),
            "X-Cache-Hit": "1" if meta.cache_hit else "0",
            "X-Language": meta.language or "",
            "Content-Disposition": f'inline; filename="speech.{fmt}"',
        },
    )


@router.post("/v1/tts/stream", dependencies=[Depends(require_api_key)],
             summary="Synthesize speech, streamed")
async def tts_stream(req: SynthesisRequest) -> StreamingResponse:
    """Stream audio as it is generated. First bytes ship when chunk 0 completes."""
    service.counters["requests"] += 1
    service.counters["streamed"] += 1
    try:
        service.require_ready()
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc) from exc

    fmt = _resolve_format(req.format)
    rate = _resolve_rate(req.sample_rate)

    # Hold an admission slot across both phases: the generator below continues
    # to use the GPU, so releasing after chunk 0 would let the limiter under-count.
    slot = service.slot()
    await slot.__aenter__()
    try:
        started, prompt, temp_ref, first, rest = await _open_stream(req)
    except Exception as exc:  # noqa: BLE001
        await slot.__aexit__(None, None, None)
        service.counters["errors"] += 1
        raise _http_error(exc) from exc

    async def _guarded() -> AsyncIterator[bytes]:
        try:
            async for piece in _stream_audio(req, started, prompt, temp_ref, first, rest):
                yield piece
        finally:
            await slot.__aexit__(None, None, None)

    return StreamingResponse(
        _guarded(),
        media_type=audio_io.streaming_content_type(fmt),
        headers={
            "X-Sample-Rate": str(rate),
            "X-Audio-Format": "pcm" if fmt == "pcm" else "wav",
            "Cache-Control": "no-cache",
            # Belt and braces for proxies that ignore proxy_buffering off.
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/v1/audio/speech", dependencies=[Depends(require_api_key)],
             summary="OpenAI-compatible speech endpoint")
async def openai_speech(req: SpeechRequest, request: Request) -> Response:
    """Drop-in for OpenAI's ``/v1/audio/speech``; also accepts this server's fields."""
    synthesis = req.to_synthesis_request()
    if synthesis.stream:
        return await tts_stream(synthesis)
    return await tts(synthesis, request)


# ---------------------------------------------------------------------------
# Voice routes
# ---------------------------------------------------------------------------
@router.get("/v1/voices", response_model=VoiceListResponse, summary="List voices")
async def list_voices() -> VoiceListResponse:
    synthesizer = service.require_ready()
    return VoiceListResponse(
        voices=[VoiceInfo(**info) for info in synthesizer.registry.describe()],
        default_voice=settings.voices.default_voice,
    )


async def _clone(voice_id: str, ref_text: str, audio: bytes, language: str | None,
                 overwrite: bool) -> dict:
    synthesizer = service.require_ready()
    if synthesizer.registry.has(voice_id) and not overwrite:
        raise HTTPException(status_code=409,
                            detail=f"voice '{voice_id}' exists; pass overwrite=true to replace")
    if not audio:
        raise HTTPException(status_code=400, detail="reference audio is required")
    if len(audio) > MAX_REFERENCE_BYTES:
        raise HTTPException(status_code=413, detail="reference audio too large")

    handle = tempfile.NamedTemporaryFile(prefix=f"clone_{voice_id}_", suffix=".wav", delete=False)
    handle.write(audio)
    handle.close()
    try:
        async with service.slot():
            return await synthesizer.engine.create_voice(
                voice_id, handle.name, ref_text, language=language
            )
    finally:
        Path(handle.name).unlink(missing_ok=True)


@router.post("/v1/voices", dependencies=[Depends(require_api_key)], summary="Clone a voice")
async def clone_voice(
    request: Request,
    audio: Optional[UploadFile] = File(None, description="Reference clip (wav/mp3)"),
    voice_id: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    overwrite: bool = Form(False),
) -> JSONResponse:
    """Clone a voice from a reference clip. Usable immediately — no restart.

    Accepts multipart/form-data (``audio`` file plus fields) or JSON
    (``audio_base64``). ``reference_text`` is required either way: this server
    runs no ASR, and a wrong transcript degrades every future synthesis with
    that voice.
    """
    content_type = (request.headers.get("content-type") or "").lower()

    if content_type.startswith("multipart/"):
        if not voice_id or not reference_text or audio is None:
            raise HTTPException(
                status_code=400,
                detail="multipart clone needs voice_id, reference_text and an audio file",
            )
        payload = VoiceCloneRequest(voice_id=voice_id, reference_text=reference_text,
                                    language=language, overwrite=overwrite)
        blob = await audio.read()
    else:
        try:
            body = VoiceCloneRequest(**(await request.json()))
        except ValidationError as exc:
            # Without this the raw pydantic error escapes as a 500; the caller
            # needs to be told which field they left out.
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except (json.JSONDecodeError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        payload = body
        if not body.audio_base64:
            raise HTTPException(status_code=400, detail="audio_base64 is required for a JSON clone")
        try:
            blob = base64.b64decode(body.audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"audio_base64 is not valid base64: {exc}")

    try:
        result = await _clone(payload.voice_id, payload.reference_text, blob,
                              payload.language, payload.overwrite)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc) from exc
    return JSONResponse({"status": "ok", **result})


@router.delete("/v1/voices/{voice_id}", dependencies=[Depends(require_api_key)],
               summary="Delete a voice")
async def delete_voice(voice_id: str) -> JSONResponse:
    synthesizer = service.require_ready()
    if not synthesizer.engine.delete_voice(voice_id):
        raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found")
    return JSONResponse({"status": "ok", "voice_id": voice_id})


@router.post("/v1/voices/preview", dependencies=[Depends(require_api_key)],
             summary="Clone and speak in one call, without saving the voice")
async def preview_voice(
    audio: UploadFile = File(..., description="Reference clip (wav/mp3)"),
    reference_text: str = Form(..., description="Transcript of the reference clip"),
    text: str = Form(..., description="Text to speak in the cloned voice"),
    language: Optional[str] = Form(None),
    speed: Optional[float] = Form(None),
    format: Optional[str] = Form(None),
    sample_rate: Optional[int] = Form(None),
    num_step: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
) -> Response:
    """Hear a voice before deciding to keep it. Nothing is written to disk."""
    blob = await audio.read()
    request = SynthesisRequest(
        text=text,
        reference_audio=base64.b64encode(blob).decode("ascii"),
        reference_text=reference_text,
        language=language, speed=speed, format=format, sample_rate=sample_rate,
        generation={"num_step": num_step, "guidance_scale": guidance_scale},
    )
    service.counters["requests"] += 1
    try:
        async with service.slot():
            body, fmt, content_type, meta = await _synthesize_whole(request)
    except Exception as exc:  # noqa: BLE001
        service.counters["errors"] += 1
        raise _http_error(exc) from exc
    return Response(content=body, media_type=content_type,
                    headers={"X-Sample-Rate": str(meta.sample_rate),
                             "X-Audio-Format": fmt,
                             "X-Duration-Seconds": str(meta.duration_seconds)})


# ---------------------------------------------------------------------------
# Introspection routes
# ---------------------------------------------------------------------------
@router.get("/v1/languages", summary="Supported languages")
async def languages() -> JSONResponse:
    """Languages this server has text-normalization tables for.

    OmniVoice itself supports 600+ languages; anything not listed here still
    synthesizes, it just receives lighter text preprocessing.
    """
    from flowtts.text.languages import (
        DIGIT_FALLBACK_LANGUAGES,
        SUPPORTED_LANGUAGES,
        _PROFILES,
    )

    return JSONResponse({
        "languages": [
            {
                "code": code,
                "name": profile.name,
                "script": profile.script or "latn",
                "omnivoice_code": profile.omnivoice_code,
                "numbers": "cardinal" if code not in DIGIT_FALLBACK_LANGUAGES else "digit-by-digit",
            }
            for code, profile in sorted(_PROFILES.items())
        ],
        "count": len(SUPPORTED_LANGUAGES),
        "note": "OmniVoice supports 600+ languages; these are the ones with "
                "number/date/abbreviation normalization tables.",
    })


@router.post("/v1/normalize", summary="Run the text preprocessor alone")
async def normalize(payload: dict) -> JSONResponse:
    """Preview what the model will actually be asked to say.

    Body: ``{"text": "...", "language": "hi", "normalizer": {...}}``. Useful for
    debugging a mispronunciation without spending a GPU call on it.
    """
    from flowtts.synthesis.chunker import estimate_duration
    from flowtts.text import NormalizerConfig, detect_language, normalize_for_tts

    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    language = payload.get("language")
    config_data = settings.text.model_dump()
    config_data.pop("default_language", None)
    config_data.update({k: v for k, v in (payload.get("normalizer") or {}).items()
                        if v is not None})

    clean, resolved = normalize_for_tts(text, language, NormalizerConfig.from_dict(config_data))

    from flowtts.synthesis.chunker import split_for_streaming
    chunks = split_for_streaming(
        clean,
        first_chunk_seconds=settings.streaming.first_chunk_seconds,
        first_chunk_max_seconds=settings.streaming.first_chunk_max_seconds,
        second_chunk_seconds=settings.streaming.second_chunk_seconds,
        chunk_seconds=settings.streaming.chunk_seconds,
        min_chunk_seconds=settings.streaming.min_chunk_seconds,
    )
    from flowtts.text import omnivoice_lang, resolve_language

    from flowtts.synthesis.models import for_model

    return JSONResponse({
        "original": text,
        "normalized": for_model(clean),
        # Three views of the language, because this endpoint exists to show what
        # the server decided: what the caller sent, what the normalizer tables
        # were selected by, and what OmniVoice will actually be given (it keys
        # several Indic languages by ISO 639-3, so "or" reaches it as "ory").
        "language": resolved,
        "resolved_language": resolve_language(resolved) if resolved else None,
        "omnivoice_language": omnivoice_lang(resolved),
        "detected_language": detect_language(text),
        "chunks": [
            {"index": i, "text": for_model(chunk),
             "estimated_seconds": round(estimate_duration(chunk), 2)}
            for i, chunk in enumerate(chunks)
        ],
        "estimated_seconds": round(estimate_duration(clean), 2),
    })


@router.get("/v1/stats", summary="Engine and latency counters")
async def stats() -> JSONResponse:
    return JSONResponse(service.stats())


@router.get("/healthz", summary="Liveness")
@router.get("/health", include_in_schema=False)
async def healthz() -> JSONResponse:
    if service.restarting:
        return JSONResponse({"status": "error", "reason": "restarting"}, status_code=503)
    if not service.ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok", "ready": True})


@router.get("/readyz", summary="Readiness")
@router.get("/ready", include_in_schema=False)
async def readyz() -> JSONResponse:
    if not service.ready or service.restarting or service.oom_recovery:
        return JSONResponse({"ready": False, "oom_recovery": service.oom_recovery},
                            status_code=503)
    return JSONResponse({
        "ready": True,
        "voices": service.synthesizer.registry.aliases(),
        "sample_rate": service.sample_rate,
        "formats": audio_io.available_formats(),
    })


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(),
                    media_type=CONTENT_TYPE_LATEST.split(";")[0].strip())


# ---------------------------------------------------------------------------
# WebSocket — the FlowTTS binary streaming protocol
# ---------------------------------------------------------------------------
async def _ws_session(ws: WebSocket, call_id: str) -> None:
    """One connection = one call. Text in as JSON, audio out as binary frames.

    Each audio frame is a JSON header immediately followed by raw little-endian
    int16 PCM, which lets a client demultiplex without a second channel. This is
    the protocol the existing FlowTTS clients already speak, unchanged; the new
    per-request parameters are additive fields on the synthesize message.
    """
    await ws.accept()
    cancel_events: dict[str, asyncio.Event] = {}
    logger.info("ws_connected", call_id=call_id)

    try:
        while True:
            raw = await ws.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "error": "invalid JSON"}))
                continue

            kind = message.get("type", "synthesize")
            text_id = message.get("text_id") or str(uuid.uuid4())

            if kind == "cancel":
                event = cancel_events.get(message.get("text_id") or "")
                if event:
                    event.set()
                continue
            if kind == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
                continue

            try:
                req = SynthesisRequest(
                    text=message.get("text") or "",
                    voice_id=message.get("voice_id"),
                    language=message.get("language"),
                    instruct=message.get("instruct"),
                    speed=message.get("speed"),
                    duration=message.get("duration"),
                    sample_rate=message.get("sample_rate"),
                    generation=message.get("generation"),
                    normalizer=message.get("normalizer"),
                    normalize=message.get("normalize"),
                )
            except Exception as exc:  # noqa: BLE001 — validation error
                await ws.send_text(json.dumps({
                    "type": "error", "call_id": call_id, "text_id": text_id,
                    "error": str(exc),
                }))
                continue

            event = asyncio.Event()
            cancel_events[text_id] = event
            try:
                await _ws_synthesize(ws, req, message.get("call_id") or call_id,
                                     text_id, event)
            finally:
                cancel_events.pop(text_id, None)

    except WebSocketDisconnect:
        logger.info("ws_disconnected", call_id=call_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ws_error", call_id=call_id, error=str(exc))


async def _ws_synthesize(ws: WebSocket, req: SynthesisRequest, call_id: str,
                         text_id: str, cancel: asyncio.Event) -> None:
    started = time.perf_counter()
    rate = _resolve_rate(req.sample_rate)
    service.counters["requests"] += 1
    service.counters["streamed"] += 1

    try:
        synthesizer = service.require_ready()
    except Exception as exc:  # noqa: BLE001
        await ws.send_text(json.dumps({"type": "error", "call_id": call_id,
                                       "text_id": text_id, "error": str(exc)}))
        return

    index = 0
    total_bytes = 0
    total_samples = 0
    ttfb_ms: int | None = None

    try:
        async with service.slot():
            async for chunk in synthesizer.synthesize_stream(
                req.text, cancel_event=cancel, **_synth_kwargs(req, None)
            ):
                if not chunk.audio.size and not chunk.is_final:
                    continue
                audio = _out(chunk.audio, rate)
                pcm = audio_io.to_pcm16(audio)
                total_bytes += len(pcm)
                total_samples += audio.size

                if ttfb_ms is None:
                    ttfb_ms = round((time.perf_counter() - started) * 1000)
                    service.record_ttfb(ttfb_ms)

                header = json.dumps({
                    "type": "audio_chunk",
                    "call_id": call_id,
                    "text_id": text_id,
                    "chunk_index": index,
                    "sample_rate": rate,
                    "encoding": "pcm_int16",
                    "wav_bytes": len(pcm),
                    "tokens": len(pcm) // 2,
                    "is_final": chunk.is_final,
                    "cache_hit": False,
                }).encode()
                await ws.send_bytes(header + pcm)
                index += 1

        elapsed = time.perf_counter() - started
        audio_seconds = total_samples / rate if rate else 0.0
        rtf = elapsed / audio_seconds if audio_seconds > 0 else 0.0
        if audio_seconds > 0:
            service.record_rtf(rtf)

        await ws.send_text(json.dumps({
            "type": "audio_done",
            "call_id": call_id,
            "text_id": text_id,
            "chunks": index,
            "total_tokens": total_samples,
            "total_wav_bytes": total_bytes,
            "sample_rate": rate,
            "llm_ttft_ms": ttfb_ms,
            "decoder_ttft_ms": ttfb_ms,
            "llm_s": round(elapsed, 4),
            "decode_s": 0.0,
            "rtf": round(rtf, 3),
            "cancelled": cancel.is_set(),
        }))
        logger.info("ws_done", call_id=call_id, chunks=index, ttfb_ms=ttfb_ms,
                    audio_s=round(audio_seconds, 2), rtf=round(rtf, 3))

    except Exception as exc:  # noqa: BLE001
        service.counters["errors"] += 1
        if service.is_oom(exc):
            await service.handle_oom()
        logger.error("ws_synthesis_failed", call_id=call_id, error=str(exc))
        try:
            await ws.send_text(json.dumps({"type": "error", "call_id": call_id,
                                           "text_id": text_id, "error": str(exc)}))
        except Exception:  # noqa: BLE001 — client already gone
            pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load the model in the background so the port binds immediately.

    Binding first means a health check gets "loading" instead of a refused
    connection, which is the difference between a rolling restart and an outage
    as far as a load balancer is concerned.
    """
    task = asyncio.create_task(service.initialize())
    try:
        yield
    finally:
        task.cancel()


def create_app(*, load_on_startup: bool = True) -> FastAPI:
    app = FastAPI(
        lifespan=_lifespan if load_on_startup else None,
        title="FlowTTS — OmniVoice",
        description=(
            "Low-latency multilingual TTS on k2-fsa/OmniVoice, with a TensorRT / "
            "TensorRT-LLM accelerated Qwen3 backbone, voice cloning, streaming "
            "synthesis and Indic text normalization."
        ),
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Sample-Rate", "X-Audio-Format", "X-Duration-Seconds",
                        "X-Total-Ms", "X-Cache-Hit", "X-Language"],
    )
    app.include_router(router)

    @app.websocket("/ws")
    async def ws_root(websocket: WebSocket) -> None:
        await _ws_session(websocket, str(uuid.uuid4()))

    @app.websocket("/ws/{call_id}")
    async def ws_call(websocket: WebSocket, call_id: str) -> None:
        await _ws_session(websocket, call_id)

    return app


app = create_app()
