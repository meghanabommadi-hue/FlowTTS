"""Synthesis: text → audio.

Public API:
  get_synthesizer()  →  BaseSynthesizer   (from flowtts.synthesis.engine)
  BaseSynthesizer, SynthChunk, SynthResult (from flowtts.synthesis.base)
"""

from flowtts.synthesis.base   import BaseSynthesizer, SynthChunk, SynthResult
from flowtts.synthesis.engine import get_synthesizer

__all__ = ["BaseSynthesizer", "SynthChunk", "SynthResult", "get_synthesizer"]
