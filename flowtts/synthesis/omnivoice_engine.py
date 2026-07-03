"""Deprecated: OmniVoice was replaced by Fish Audio S2 Pro (served via sglang-omni).

This shim re-exports the new engine under the old name so any lingering import of
``OmniVoiceEngine`` keeps working. Use ``flowtts.synthesis.fish_engine`` directly.
"""

from __future__ import annotations

from flowtts.synthesis.fish_engine import FishSpeechEngine

# Back-compat alias.
OmniVoiceEngine = FishSpeechEngine

__all__ = ["OmniVoiceEngine", "FishSpeechEngine"]
