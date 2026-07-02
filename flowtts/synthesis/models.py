"""Pipeline position: SYNTHESIS — text → 24 kHz waveform (public wrapper).

Role in pipeline:
  Thin, stable facade over OmniVoiceEngine used by server.py and the Redis worker.
  Exposes two entry points:

    synthesize(text, ...)         → full utterance as one np.ndarray (non-streaming)
    synthesize_stream(text, ...)  → async generator of (chunk_index, waveform, is_final)

Streaming model (OmniVoice is non-autoregressive → cannot emit token-by-token):
  The text is split into chunks (short first chunk for low TTFB). ALL chunks are
  dispatched to the engine's batch queue immediately so they coalesce with other
  requests and with each other; results are yielded IN ORDER so the client plays
  a continuous stream while later chunks are still generating (pipelining).
"""

from __future__ import annotations

import asyncio

import numpy as np
import structlog

from flowtts.core.config import settings
from flowtts.synthesis.omnivoice_engine import OmniVoiceEngine
from flowtts.synthesis.text_chunker import normalize_text, split_for_streaming

logger = structlog.get_logger(__name__)


class OmniVoiceSynthesizer:
    """Loads OmniVoice once; synthesizes text → waveform, whole or streamed."""

    def __init__(self) -> None:
        self.engine = OmniVoiceEngine()

    async def initialize(self) -> None:
        await self.engine.initialize()

    # Expose engine metadata for server.py / metrics registration.
    @property
    def sampling_rate(self) -> int:
        return self.engine.sampling_rate

    @property
    def engine_info(self) -> dict:
        return self.engine.engine_info

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
        """Yield (chunk_index, waveform, is_final) as each chunk completes, in order."""
        text = normalize_text(text)
        st = settings.streaming
        chunks = split_for_streaming(
            text,
            first_chunk_max_chars=st.first_chunk_max_chars,
            chunk_max_chars=st.chunk_max_chars,
            min_chunk_chars=st.min_chunk_chars,
        )
        if not chunks:
            return

        # Dispatch every chunk immediately so they batch together; yield in order.
        tasks = [
            asyncio.create_task(
                self.engine.synthesize(c, voice_id=voice_id, speed=speed, language=language)
            )
            for c in chunks
        ]
        last = len(tasks) - 1
        try:
            for i, task in enumerate(tasks):
                wav = await task
                yield i, wav, (i == last)
        finally:
            # If the consumer stops early (client cancel/disconnect), don't leak tasks.
            for task in tasks:
                if not task.done():
                    task.cancel()
