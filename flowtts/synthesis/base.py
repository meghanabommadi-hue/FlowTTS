"""Pipeline position: SYNTHESIS — shared interface for all TTS backends.

Every model (Mira, VoxCPM, …) implements BaseSynthesizer.
server.py only imports this interface — it never imports a concrete class.

Adding a new model:
  1. Create flowtts/synthesis/<name>.py  →  class <Name>Synthesizer(BaseSynthesizer)
  2. Register it in flowtts/synthesis/engine.py  →  _REGISTRY dict
  3. Add its Settings block to flowtts/core/config.py
  Done — server.py needs zero changes.

Streaming protocol (yielded by synthesize_stream):
  Each yielded item is a SynthChunk namedtuple:
    .wav_bytes   — complete WAV file for this chunk (ready to send to client)
    .is_final    — True on the last chunk
    .sample_rate — Hz of the WAV (may differ per model: Mira=16k, VoxCPM=48k)
    .n_tokens    — discrete speech tokens in this chunk (0 for diffusion models)
    .meta        — free-form dict (timing etc; non-empty only on final chunk)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, NamedTuple


class SynthChunk(NamedTuple):
    wav_bytes:   bytes
    is_final:    bool
    sample_rate: int
    n_tokens:    int   # 0 for models without discrete tokens (VoxCPM)
    meta:        dict  # timing / debug info — non-empty only on final chunk


class BaseSynthesizer(ABC):
    """Abstract TTS backend.

    Concrete subclasses must implement:
      - initialize()         async, called once at startup
      - synthesize()         async, full-response (accumulates everything)
      - synthesize_stream()  async-generator, yields SynthChunk per chunk
      - sample_rate          property

    Both synthesize() and synthesize_stream() must be safe for concurrent
    async callers (the underlying engine serialises GPU work internally).
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Load model weights.  Safe to call multiple times (no-op after first)."""

    @abstractmethod
    async def synthesize(self, text: str, voice_id: str | None = None) -> "SynthResult":
        """Synthesize text fully.  Returns SynthResult (wav_bytes + metadata)."""

    @abstractmethod
    async def synthesize_stream(self, text: str, voice_id: str | None = None) -> AsyncGenerator[SynthChunk, None]:
        """Async generator — yields SynthChunk for each audio chunk."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""


class SynthResult(NamedTuple):
    """Full-response result from synthesize()."""
    wav_bytes:   bytes
    sample_rate: int
    n_tokens:    int   # 0 for diffusion models
    llm_s:       float # time spent in LLM / generation
    decode_s:    float # time spent in decoder / VAE (0 for diffusion models)
