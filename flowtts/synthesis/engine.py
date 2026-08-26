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
from flowtts.synthesis.omnivoice_engine import GenParams

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
        instruct: str | None = None,
        **generation,
    ) -> np.ndarray:
        """Synthesize one utterance through the shared engine.

        ``speed`` and any OmniVoice generation-config field may be passed as
        keyword arguments; they are folded into a GenParams for this request
        only, leaving the server defaults untouched.
        """
        if not self._initialized:
            raise RuntimeError("SynthesisService not initialized. Call initialize() first.")
        overrides = {k: v for k, v in generation.items() if v is not None}
        if speed is not None:
            overrides["speed"] = speed
        return await self.synthesizer.synthesize(
            text,
            voice_id=voice_id,
            language=language,
            instruct=instruct,
            params=GenParams.build(overrides) if overrides else None,
        )

    @property
    def is_initialized(self) -> bool:
        return self._initialized


synthesis_service = SynthesisService()
