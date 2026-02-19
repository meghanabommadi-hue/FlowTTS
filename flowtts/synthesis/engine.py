"""Main synthesis service that manages TTS model initialization.

This mirrors ``litranscriber.transcription.engine`` but for a single
FlowTTS model that produces audio tokens.
"""

from __future__ import annotations

import structlog

from flowtts.synthesis.models import FlowTtsSynthesizer


logger = structlog.get_logger(__name__)


class SynthesisService:
    """Singleton service that manages TTS synthesis."""

    _instance: "SynthesisService | None" = None
    _initialized: bool = False

    def __new__(cls) -> "SynthesisService":
        if cls._instance is None:
            cls._instance = super(SynthesisService, cls).__new__(cls)
        return cls._instance

    async def initialize(self) -> None:
        """Initialize the underlying synthesizer."""
        if self._initialized:
            logger.info("synthesis_service_already_initialized")
            return

        logger.info("initializing_synthesis_service")
        try:
            self.synthesizer: FlowTtsSynthesizer = FlowTtsSynthesizer()
            await self.synthesizer.initialize()
            self._initialized = True
            logger.info("synthesis_service_initialized")
        except Exception as e:
            logger.error("synthesis_service_initialization_failed", error=str(e))
            raise

    async def synthesize(self, text: str) -> str:
        """Generate audio token sequence for the given text."""
        if not self._initialized:
            raise RuntimeError(
                "SynthesisService not initialized. "
                "Call initialize() first during application startup."
            )
        return await self.synthesizer.synthesize(text)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


synthesis_service = SynthesisService()

