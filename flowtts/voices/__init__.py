"""Voice-clone system: precomputed npz prompts addressed by alias.

  npz_io.py   — pure-NumPy save/load of the .npz voice-clone format (no torch)
  registry.py — VoiceRegistry: alias → VoiceClonePrompt, loaded at startup
  clone.py    — offline CLI to build <alias>.npz from reference audio
"""

from flowtts.voices.registry import VoiceRegistry

__all__ = ["VoiceRegistry"]
