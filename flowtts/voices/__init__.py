"""Voice-clone system: reference clips + transcripts addressed by alias.

  store.py    — stdlib JSON manifest save/load for the reference-voice format (no torch)
  registry.py — VoiceRegistry: alias → (reference clip path, ref_text), loaded at startup
  clone.py    — offline CLI to build <alias>.wav + <alias>.json from reference audio
  npz_io.py   — DEPRECATED: legacy OmniVoice codec-token .npz format (unused)
"""

from flowtts.voices.registry import VoiceRegistry

__all__ = ["VoiceRegistry"]
