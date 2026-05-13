"""Pipeline position: SYNTHESIS SERVICE — singleton wrapper around the model.

Role in pipeline:
  Provides a process-wide singleton (synthesis_service) that both the
  worker (worker.py) and the single-process server (server.py) call via:

      audio_tokens = await synthesis_service.synthesize(text)

  Ensures the sglang Engine and ncodec TTSCodec are loaded exactly once per
  process, no matter how many concurrent requests arrive.

Lazy initialisation:
  initialize() is called on first use (worker._process_job) or at server
  startup (server._get_synthesizer). Subsequent calls are no-ops.

Async safety:
  synthesize() delegates to FlowTtsSynthesizer.synthesize() which calls
  sgl.Engine.async_generate() — non-blocking, safe for concurrent coroutines.
  The sglang Engine serialises GPU requests internally.
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

    async def synthesize(self, text: str, language: str | None = None) -> str:
        """Generate audio token sequence for the given text.

        Args:
            text:     Input text to synthesize.
            language: Language tag for LoRA routing (e.g. "hi", "ta").
                      Defaults to settings.tts_model.default_language.
        """
        if not self._initialized:
            raise RuntimeError(
                "SynthesisService not initialized. "
                "Call initialize() first during application startup."
            )
        return await self.synthesizer.synthesize(text, language=language)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


synthesis_service = SynthesisService()

