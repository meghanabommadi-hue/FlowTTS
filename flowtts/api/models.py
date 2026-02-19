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
    audio_base64: str = Field(..., description="Base64-encoded WAV PCM data")
    sample_rate: int = Field(..., description="Sample rate of the WAV data")
    is_final: bool = Field(default=True)


class ErrorMessage(BaseModel):
    """Server → Client: error payload."""

    type: MessageType = Field(default=MessageType.ERROR)
    call_id: Optional[str] = None
    text_id: Optional[str] = None
    error: str

