"""TTS model wrapper: produces audio token sequence from text."""

from __future__ import annotations


class FlowTtsSynthesizer:
    """Wrapper for the FlowTTS model (e.g. Qwen/Mira). Produces audio_tokens string from text."""

    async def initialize(self) -> None:
        """Load model. Override in real implementation."""
        pass

    async def synthesize(self, text: str) -> str:
        """Return full audio token string for the given text. Override in real implementation."""
        # Stub: real impl would load model and run inference.
        return ""
