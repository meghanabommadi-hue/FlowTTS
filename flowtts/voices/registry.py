"""Pipeline position: VOICE REGISTRY — alias → cloned voice, loaded at startup.

Role in pipeline:
  On startup the OmniVoice engine builds one VoiceRegistry over the voices_dir.
  Every WebSocket request's ``voice_id`` is an alias resolved here to a
  ``VoiceClonePrompt`` that is passed to ``model.generate(voice_clone_prompt=...)``,
  which skips the codec encoder + Whisper ASR entirely.

  server.py / engine → registry.prompt(voice_id) → VoiceClonePrompt → generate()

Adding a voice at deploy time = drop `<alias>.npz` into voices_dir (see
voices/clone.py). No code change, no re-encoding on boot.

torch and the omnivoice package are imported lazily inside prompt-building so this
module (and the npz format) stay importable/testable without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from flowtts.voices.npz_io import load_voice_npz

logger = structlog.get_logger(__name__)


class VoiceRegistry:
    """Loads voice-clone npz files and hands out VoiceClonePrompt objects by alias."""

    def __init__(self, voices_dir: str | Path, default_voice: str | None = None) -> None:
        self.voices_dir = Path(voices_dir)
        self.default_voice = default_voice
        self._raw: dict[str, dict[str, Any]] = {}     # alias → npz dict
        self._prompts: dict[str, Any] = {}            # alias → VoiceClonePrompt (built lazily)
        self._load_all()

    # ------------------------------------------------------------------ loading
    def _load_all(self) -> None:
        if not self.voices_dir.is_dir():
            logger.warning("voices_dir_missing", voices_dir=str(self.voices_dir))
            return
        for npz_path in sorted(self.voices_dir.glob("*.npz")):
            try:
                data = load_voice_npz(npz_path)
                alias = data.get("alias") or npz_path.stem
                self._raw[alias] = data
                logger.info(
                    "voice_loaded",
                    alias=alias,
                    tokens=tuple(data["ref_audio_tokens"].shape),
                    ref_text_preview=data["ref_text"][:40],
                )
            except Exception as e:  # noqa: BLE001
                logger.error("voice_load_failed", path=str(npz_path), error=str(e))

        if self.default_voice and self.default_voice not in self._raw:
            logger.warning(
                "default_voice_missing",
                default_voice=self.default_voice,
                available=sorted(self._raw),
            )

    # ------------------------------------------------------------------ public
    def aliases(self) -> list[str]:
        return sorted(self._raw)

    def has(self, alias: str | None) -> bool:
        return bool(alias) and alias in self._raw

    def language(self, voice_id: str | None) -> str | None:
        """Return the resolved voice's preferred language, or None if unset."""
        alias = self.resolve(voice_id)
        if alias is None:
            return None
        lang = self._raw[alias].get("language")
        return lang or None

    def add(self, alias: str, npz_path: str | Path) -> dict[str, Any]:
        """Hot-register a voice from an npz (used by the live clone REST endpoint)."""
        data = load_voice_npz(npz_path)
        self._raw[alias] = data
        self._prompts.pop(alias, None)  # drop any stale cached prompt
        logger.info("voice_registered", alias=alias, tokens=tuple(data["ref_audio_tokens"].shape))
        return data

    def remove(self, alias: str) -> bool:
        """Drop a voice from the live registry (used by DELETE /v1/voices/{id})."""
        existed = self._raw.pop(alias, None) is not None
        self._prompts.pop(alias, None)
        if existed:
            logger.info("voice_unregistered", alias=alias)
        return existed

    def describe(self) -> list[dict]:
        """Metadata for every loaded voice, for GET /v1/voices."""
        return [
            {
                "voice_id": alias,
                "language": data.get("language") or None,
                "reference_frames": int(data["ref_audio_tokens"].shape[-1]),
                "ref_text": data["ref_text"],
                "sample_rate": data.get("sample_rate") or None,
                "is_default": alias == self.default_voice,
            }
            for alias, data in sorted(self._raw.items())
        ]

    def resolve(self, voice_id: str | None) -> str | None:
        """Return the alias to use: the requested one if known, else the default."""
        if voice_id and voice_id in self._raw:
            return voice_id
        if self.default_voice and self.default_voice in self._raw:
            return self.default_voice
        return None

    def prompt(self, voice_id: str | None):
        """Return a VoiceClonePrompt for the resolved alias, or None if unavailable.

        Builds the torch-backed prompt on first use and caches it.
        """
        alias = self.resolve(voice_id)
        if alias is None:
            return None
        if alias in self._prompts:
            return self._prompts[alias]

        prompt = self._build_prompt(self._raw[alias])
        self._prompts[alias] = prompt
        return prompt

    # ------------------------------------------------------------------ internal
    @staticmethod
    def _build_prompt(data: dict[str, Any]):
        """Reconstruct a VoiceClonePrompt from npz data (imports torch + omnivoice)."""
        import torch
        from omnivoice.models.omnivoice import VoiceClonePrompt

        tokens = torch.from_numpy(data["ref_audio_tokens"].astype("int64"))
        return VoiceClonePrompt(
            ref_audio_tokens=tokens,
            ref_text=data["ref_text"],
            ref_rms=float(data["ref_rms"]),
        )
