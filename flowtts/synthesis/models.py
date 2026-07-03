"""Pipeline position: SYNTHESIS — text → 24 kHz waveform (public wrapper).

Role in pipeline:
  Thin, stable facade over FishSpeechEngine used by server.py and the Redis worker.
  Exposes two entry points:

    synthesize(text, ...)         → full utterance as one np.ndarray (non-streaming)
    synthesize_stream(text, ...)  → async generator of (chunk_index, waveform, is_final)

Streaming model:
  Fish Audio S2 Pro is autoregressive, so the sglang backend streams one CONTIGUOUS
  16-bit PCM stream. We forward those PCM fragments straight through as float32
  waveform chunks (no text chunking, no per-chunk crossfade — see `continuous_stream`).
"""

from __future__ import annotations

import numpy as np
import structlog

from flowtts.synthesis.fish_engine import FishSpeechEngine
from flowtts.synthesis.text_chunker import normalize_text

logger = structlog.get_logger(__name__)


class FishSpeechSynthesizer:
    """Connects to the Fish S2 Pro backend once; synthesizes text → waveform, whole or streamed."""

    def __init__(self) -> None:
        self.engine = FishSpeechEngine()

    async def initialize(self) -> None:
        await self.engine.initialize()

    async def close(self) -> None:
        await self.engine.close()

    # Expose engine metadata for server.py / metrics registration.
    @property
    def sampling_rate(self) -> int:
        return self.engine.sampling_rate

    @property
    def engine_info(self) -> dict:
        return self.engine.engine_info

    @property
    def continuous_stream(self) -> bool:
        """True → the backend streams contiguous PCM; callers must not crossfade chunks."""
        return self.engine.continuous_stream

    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        speed: float | None = None,
        language: str | None = None,
    ) -> np.ndarray:
        """Return the full waveform (float32, engine.sampling_rate) for *text*."""
        text = normalize_text(text)
        if not text:
            return np.zeros(0, dtype=np.float32)
        return await self.engine.synthesize(text, voice_id=voice_id, speed=speed, language=language)

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        speed: float | None = None,
        language: str | None = None,
    ):
        """Yield (chunk_index, waveform, is_final) as PCM streams from the backend, in order."""
        text = normalize_text(text)
        if not text:
            return
        async for item in self.engine.synthesize_stream(
            text, voice_id=voice_id, speed=speed, language=language
        ):
            yield item


# Backward-compatible alias: the previous class name is kept so any importer that
# still references OmniVoiceSynthesizer (e.g. the Redis worker path) keeps working.
OmniVoiceSynthesizer = FishSpeechSynthesizer
