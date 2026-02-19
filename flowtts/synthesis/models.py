"""Concrete synthesizer implementations for FlowTTS.

This mirrors the structure of ``litranscriber.transcription.models``, but
for a single TTS model that produces audio tokens via ``TtsEngine``.
"""

from __future__ import annotations

from typing import Optional

import structlog

from flowtts.synthesis.engine import TtsEngine


logger = structlog.get_logger(__name__)


class FlowTtsSynthesizer:
    """Simple synthesizer wrapper that exposes a ``synthesize`` coroutine."""

    def __init__(self) -> None:
        self._engine: Optional[TtsEngine] = TtsEngine()

    async def initialize(self) -> None:
        if self._engine is None:
            self._engine = TtsEngine()
        await self._engine.initialize()
        logger.info("flowtts_synthesizer_initialized")

    async def synthesize(self, text: str) -> str:
        """Return audio-token string for the given text."""
        assert self._engine is not None
        return await self._engine.generate_tokens(text)


synthesizer = FlowTtsSynthesizer()

