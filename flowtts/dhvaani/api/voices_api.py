"""Pipeline position: VOICE API — create / list / inspect / delete voice clones.

Role in pipeline:
  CRUD over `VoiceStore`. Creating a voice runs the whole reference-audio
  preparation pipeline once (decode, silence removal, trim, RMS normalise, mel)
  and persists the result, so synthesis never touches audio files.

The transcript field is required and load-bearing -- DhVaani derives its
speaking rate from prompt_frames / prompt_tokens. See `voices/store.py`.
"""

from __future__ import annotations

from urllib.parse import quote

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from flowtts.dhvaani.api.models import VoiceListResponse, VoiceResponse
from flowtts.dhvaani.api.rest import require_api_key, wav_header
from flowtts.dhvaani.config import dhv_settings
from flowtts.dhvaani.types import DhvaaniError, SynthParams, new_request_id

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/voices", tags=["voices"])

# Short, natural sentences used by the preview endpoint, per language.
_PREVIEW = {
    "hi": "नमस्ते, यह आपकी आवाज़ का नमूना है।",
    "en": "Hello, this is a preview of your cloned voice.",
    "bn": "নমস্কার, এটি আপনার কণ্ঠের নমুনা।",
    "ta": "வணக்கம், இது உங்கள் குரலின் மாதிரி.",
    "te": "నమస్కారం, ఇది మీ గొంతు నమూనా.",
    "kn": "ನಮಸ್ಕಾರ, ಇದು ನಿಮ್ಮ ಧ್ವನಿಯ ಮಾದರಿ.",
    "ml": "നമസ്കാരം, ഇത് നിങ്ങളുടെ ശബ്ദത്തിന്റെ മാതൃകയാണ്.",
    "mr": "नमस्कार, हा तुमच्या आवाजाचा नमुना आहे.",
    "gu": "નમસ્તે, આ તમારા અવાજનો નમૂનો છે.",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਇਹ ਤੁਹਾਡੀ ਆਵਾਜ਼ ਦਾ ਨਮੂਨਾ ਹੈ।",
    "or": "ନମସ୍କାର, ଏହା ଆପଣଙ୍କ ସ୍ୱରର ନମୁନା।",
    "as": "নমস্কাৰ, এইটো আপোনাৰ কণ্ঠৰ নমুনা।",
    "ur": "السلام علیکم، یہ آپ کی آواز کا نمونہ ہے۔",
}


def _store(request: Request):
    eng = getattr(request.app.state, "engine", None)
    if eng is None or not eng.ready or eng.voices is None:
        raise HTTPException(status_code=503, detail="engine is not ready")
    return eng


def _to_response(meta: dict) -> VoiceResponse:
    return VoiceResponse(**{k: v for k, v in meta.items() if k in VoiceResponse.model_fields})


async def _read_upload(file: UploadFile, limit: int) -> bytes:
    """Stream the upload, aborting past `limit` instead of buffering it all."""
    chunks: list[bytes] = []
    total = 0
    while True:
        block = await file.read(1 << 20)
        if not block:
            break
        total += len(block)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"reference audio exceeds {limit} bytes "
                       f"(voice.max_upload_bytes)",
            )
        chunks.append(block)
    if not chunks:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    return b"".join(chunks)


@router.post("", response_model=VoiceResponse, dependencies=[Depends(require_api_key)])
async def create_voice(
    request: Request,
    file: UploadFile = File(..., description="Reference clip: wav/flac/ogg/mp3/m4a"),
    voice_id: str = Form(...),
    transcript: str = Form(..., description="Exact transcript of the clip"),
    name: str = Form(default=""),
    description: str = Form(default=""),
    language: str = Form(default=""),
    overwrite: bool = Form(default=False),
):
    eng = _store(request)
    data = await _read_upload(file, dhv_settings.voice.max_upload_bytes)
    try:
        prompt = eng.voices.create(
            voice_id=voice_id,
            audio=data,
            transcript=transcript,
            name=name,
            description=description,
            language=language,
            overwrite=overwrite,
            source_filename=file.filename or "",
        )
    except DhvaaniError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    return _to_response(prompt.to_metadata())


@router.get("", response_model=VoiceListResponse)
async def list_voices(request: Request):
    eng = _store(request)
    return VoiceListResponse(data=[_to_response(m) for m in eng.voices.list()])


@router.get("/{voice_id}", response_model=VoiceResponse)
async def get_voice(voice_id: str, request: Request):
    eng = _store(request)
    try:
        return _to_response(eng.voices.get(voice_id).to_metadata())
    except DhvaaniError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@router.delete("/{voice_id}", dependencies=[Depends(require_api_key)])
async def delete_voice(voice_id: str, request: Request):
    eng = _store(request)
    try:
        eng.voices.delete(voice_id)
    except DhvaaniError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    return {"deleted": True, "voice_id": voice_id}


@router.post("/{voice_id}/preview", dependencies=[Depends(require_api_key)])
async def preview_voice(voice_id: str, request: Request, text: str | None = None):
    """Synthesize a short sentence in this voice. The fastest way to check a
    freshly created clone before wiring it into production traffic."""
    eng = _store(request)
    try:
        voice = eng.voices.get(voice_id)
    except DhvaaniError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    sentence = text or _PREVIEW.get(voice.language) or _PREVIEW["en"]
    params = SynthParams.from_settings(dhv_settings)
    try:
        pcm, metrics = await eng.synthesize(
            sentence, voice_id, voice.language, params, new_request_id()
        )
    except DhvaaniError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    sr = params.output_sample_rate
    return Response(
        wav_header(sr, len(pcm)) + pcm,
        media_type="audio/wav",
        headers={
            "X-Voice-Id": voice_id,
            # HTTP header values must be latin-1 encodable (RFC 7230), and every
            # preview sentence here is Devanagari, Tamil, Perso-Arabic ... so the
            # raw text would raise on encode. Percent-encode it instead.
            "X-Text": quote(sentence[:120], safe=""),
            "X-TTFB-Ms": str(round(metrics.ttfb_ms, 1)),
            "X-Total-Ms": str(round(metrics.total_ms, 1)),
        },
    )
