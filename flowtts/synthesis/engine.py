"""Pipeline position: SYNTHESIS SERVICE — process-wide singleton.

Role in pipeline:
  Provides a single shared synthesizer (synthesis_service) so the model loads
  exactly once per process, whether driven by server.py (single-process) or the
  Redis worker (worker.py).

      wav = await synthesis_service.synthesize(text, voice_id=...)
"""

from __future__ import annotations

import numpy as np
import structlog

from flowtts.synthesis.models import OmniVoiceSynthesizer

logger = structlog.get_logger(__name__)


class SynthesisService:
    """Singleton service that manages OmniVoice synthesis."""

    _instance: "SynthesisService | None" = None
    _initialized: bool = False

    def __new__(cls) -> "SynthesisService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self) -> None:
        if self._initialized:
            return
        logger.info("initializing_synthesis_service")
        self.synthesizer = OmniVoiceSynthesizer()
        await self.synthesizer.initialize()
        self._initialized = True
        logger.info("synthesis_service_initialized")

    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        speed: float | None = None,
        language: str | None = None,
    ) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("SynthesisService not initialized. Call initialize() first.")
        return await self.synthesizer.synthesize(
            text, voice_id=voice_id, speed=speed, language=language
        )

    @property
    def is_initialized(self) -> bool:
        return self._initialized


synthesis_service = SynthesisService()
