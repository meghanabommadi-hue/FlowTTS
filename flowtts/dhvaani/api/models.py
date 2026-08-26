"""Pipeline position: API CONTRACT — request/response schemas for the HTTP layer.

Role in pipeline:
  Validates and normalises everything arriving over REST before it reaches the
  engine. The WebSocket gateway keeps the legacy FlowTTS wire format instead
  (see `api/ws.py`); these models cover only the new HTTP surface.

`SpeechRequest` mirrors OpenAI's `POST /v1/audio/speech` so existing OpenAI TTS
clients work unchanged, with DhVaani-specific knobs added as optional fields
that OpenAI clients simply never send.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from flowtts.dhvaani.config import dhv_settings

ResponseFormat = Literal["pcm", "wav", "mp3", "opus", "aac", "flac"]
SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000)


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request."""

    # --- OpenAI fields ---
    model: str = Field(default="dhvaani-0.5", description="Ignored; one model is served")
    input: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(default=None, description="Voice id from /v1/voices")
    response_format: ResponseFormat = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: Optional[str] = Field(default=None, description="Accepted and ignored")
    stream_format: Optional[Literal["audio", "sse"]] = None

    # --- DhVaani extensions ---
    stream: bool = Field(default=False, description="Chunked streaming response")
    language: Optional[str] = Field(default=None, description="ISO code; auto-detected if absent")
    sample_rate: Optional[int] = Field(default=None)
    num_step: Optional[int] = Field(default=None, ge=1, le=64)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    seed: Optional[int] = None

    @field_validator("input")
    @classmethod
    def _check_input(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("input must not be empty")
        limit = dhv_settings.server.max_text_chars
        if len(v) > limit:
            raise ValueError(f"input is {len(v)} characters; the limit is {limit}")
        return v

    @field_validator("sample_rate")
    @classmethod
    def _check_sr(cls, v):
        if v is not None and v not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}")
        return v


class VoiceResponse(BaseModel):
    voice_id: str
    name: str = ""
    description: str = ""
    language: str = ""
    transcript: str = ""
    duration_s: float = 0.0
    mel_frames: int = 0
    n_tokens: int = 0
    frames_per_token: float = 0.0
    sample_rate: int = 24000
    created_at: float = 0.0
    source_filename: str = ""
    checksum: str = ""
    prompt_rms: float = 0.0


class VoiceListResponse(BaseModel):
    object: str = "list"
    data: list[VoiceResponse] = []


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "ARTPARK-IISc"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard] = []


class LanguageInfo(BaseModel):
    code: str
    name: str
    native_name: str
    script: str
    normalization: str


class LanguageListResponse(BaseModel):
    object: str = "list"
    data: list[LanguageInfo] = []


class HealthResponse(BaseModel):
    status: str
    ready: bool
    backend: Optional[str] = None
    reason: Optional[str] = None


class ErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
