"""Pydantic models for FlowTTS WebSocket messages."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    SYNTHESIZE = "synthesize"
    AUDIO = "audio"
    ERROR = "error"


class SynthesizeRequest(BaseModel):
    """Client → Server: request to synthesize speech for a text."""

    type: MessageType = Field(default=MessageType.SYNTHESIZE)
    call_id: str = Field(..., description="Logical call/session id")
    text_id: str = Field(..., description="Identifier for this text request")
    text: str = Field(..., description="Text to synthesize")


class AudioMessage(BaseModel):
    """Server → Client: synthesized audio payload."""

    type: MessageType = Field(default=MessageType.AUDIO)
    call_id: str
    text_id: str
    # Raw LLM token output (always present when decode is skipped)
    audio_tokens: Optional[str] = Field(default=None, description="Raw LLM audio token string")
    # Decoded audio (present when decoder is enabled)
    audio_base64: Optional[str] = Field(default=None, description="Base64-encoded WAV PCM data")
    sample_rate: Optional[int] = Field(default=None, description="Sample rate of the WAV data")
    is_final: bool = Field(default=True)
    llm_s: Optional[float] = Field(default=None, description="LLM generation time in seconds")
    decode_s: Optional[float] = Field(default=None, description="Decoder time in seconds")


class ErrorMessage(BaseModel):
    """Server → Client: error payload."""

    type: MessageType = Field(default=MessageType.ERROR)
    call_id: Optional[str] = None
    text_id: Optional[str] = None
    error: str

