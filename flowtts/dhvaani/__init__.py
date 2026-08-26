"""DhVaani — zero-shot TTS for 27 Indian languages, served on FlowTTS.

DhVaani-0.5 (ARTPARK-IISc) is a fine-tune of ZipVoice, a 123M-parameter
flow-matching TTS. It is architecturally unlike the MiraTTS path in the rest of
this repo: non-autoregressive, no audio codec, no KV cache. See docs/DHVAANI.md.

Public surface:

    from flowtts.dhvaani import DhvaaniEngine, dhv_settings

    engine = DhvaaniEngine()
    await engine.start()
    async for chunk in engine.synthesize_stream("नमस्ते", voice_id="simran"):
        ...

Imports are lazy so `import flowtts.dhvaani` does not pull in torch. That keeps
the CLI tools, the config and the tests importable on a machine with no GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flowtts.dhvaani.config import PROFILES, apply_profile, dhv_settings
from flowtts.dhvaani.types import (
    AudioChunk,
    DhvaaniError,
    EngineNotReady,
    InvalidReferenceAudio,
    QueueFull,
    RequestCancelled,
    RequestMetrics,
    SpanRequest,
    SynthParams,
    TextTooLong,
    VoiceAlreadyExists,
    VoiceNotFound,
    VoicePrompt,
)

if TYPE_CHECKING:  # pragma: no cover
    from flowtts.dhvaani.engine.engine import DhvaaniEngine

__version__ = "0.5.0"

__all__ = [
    "DhvaaniEngine",
    "dhv_settings",
    "apply_profile",
    "PROFILES",
    "SynthParams",
    "VoicePrompt",
    "SpanRequest",
    "AudioChunk",
    "RequestMetrics",
    "DhvaaniError",
    "VoiceNotFound",
    "VoiceAlreadyExists",
    "InvalidReferenceAudio",
    "TextTooLong",
    "QueueFull",
    "EngineNotReady",
    "RequestCancelled",
    "__version__",
]


def __getattr__(name: str):
    # DhvaaniEngine transitively imports torch; defer it until actually asked for.
    if name == "DhvaaniEngine":
        from flowtts.dhvaani.engine.engine import DhvaaniEngine as _E

        return _E
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
