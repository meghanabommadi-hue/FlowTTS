"""Pipeline position: SYNTHESIS ENGINE — model factory & process-level singleton.

This module is the only place that knows which concrete synthesizer class maps
to which model_type string.  Everything else (server.py, worker.py, …) only
imports BaseSynthesizer and calls get_synthesizer().

Adding a new model:
  1. Create flowtts/synthesis/<name>.py  →  class <Name>Synthesizer(BaseSynthesizer)
  2. Add one line to _REGISTRY below.
  3. Done.  server.py needs zero changes.

Usage:
    from flowtts.synthesis.engine import get_synthesizer

    synth = await get_synthesizer()   # loads once, returns same object on repeat calls
    result = await synth.synthesize(text)
"""

from __future__ import annotations

import structlog

from flowtts.core.config import settings
from flowtts.synthesis.base import BaseSynthesizer

logger = structlog.get_logger(__name__)

# ── Registry: model_type → synthesizer class ──────────────────────────────
# To add a new model: import its class and add one entry here.
def _build_registry() -> dict:
    from flowtts.synthesis.mira      import MiraSynthesizer      # noqa: PLC0415
    from flowtts.synthesis.voxcpm    import VoxCpmSynthesizer    # noqa: PLC0415
    from flowtts.synthesis.omnivoice import OmniVoiceSynthesizer # noqa: PLC0415
    from flowtts.synthesis.miotts    import MiottsSynthesizer    # noqa: PLC0415
    return {
        "mira":      MiraSynthesizer,
        "voxcpm":    VoxCpmSynthesizer,
        "omnivoice": OmniVoiceSynthesizer,
        "miotts":    MiottsSynthesizer,
    }

# Process-level singleton
_synthesizer: BaseSynthesizer | None = None


async def get_synthesizer() -> BaseSynthesizer:
    """Return the initialized synthesizer for settings.model_type.

    Loads the model on the first call; subsequent calls return the cached
    instance immediately (no re-initialization).
    """
    global _synthesizer
    if _synthesizer is not None:
        return _synthesizer

    model_type = settings.model_type
    registry = _build_registry()

    if model_type not in registry:
        raise ValueError(
            f"Unknown model_type {model_type!r}.  "
            f"Available: {sorted(registry)}"
        )

    cls = registry[model_type]
    logger.info("synthesizer_loading", model_type=model_type, cls=cls.__name__)

    synth = cls()
    await synth.initialize()
    _synthesizer = synth

    logger.info("synthesizer_ready", model_type=model_type)
    return _synthesizer
