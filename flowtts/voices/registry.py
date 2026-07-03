"""Pipeline position: VOICE REGISTRY — alias → reference clip, loaded at startup.

Role in pipeline:
  On startup the Fish engine builds one VoiceRegistry over the voices_dir. Every
  WebSocket request's ``voice_id`` is an alias resolved here to a reference
  ``(audio_path, ref_text)`` pair that is sent to the sglang backend as
  ``references=[{"audio_path": ..., "text": ...}]`` for zero-shot voice cloning.
  SGLang encodes the clip into VQ codes and caches the KV via RadixAttention, so
  repeated same-voice requests largely skip the reference prefill.

  server.py / engine → registry.reference(voice_id) → (audio_path, ref_text) → backend

Adding a voice at deploy time = drop `<alias>.json` + its clip into voices_dir
(see voices/clone.py or the live POST /voices endpoint). No code change, no
GPU work, no restart.

This module is torch-free — a voice is just a clip + transcript on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from flowtts.voices.store import load_voice

logger = structlog.get_logger(__name__)


class VoiceRegistry:
    """Loads voice manifests and resolves aliases to reference (audio_path, ref_text)."""

    def __init__(self, voices_dir: str | Path, default_voice: str | None = None) -> None:
        self.voices_dir = Path(voices_dir)
        self.default_voice = default_voice
        self._raw: dict[str, dict[str, Any]] = {}   # alias → manifest dict (+ resolved audio_path)
        self._load_all()

    # ------------------------------------------------------------------ loading
    def _load_all(self) -> None:
        if not self.voices_dir.is_dir():
            logger.warning("voices_dir_missing", voices_dir=str(self.voices_dir))
            return
        for manifest in sorted(self.voices_dir.glob("*.json")):
            try:
                self._register_manifest(manifest)
            except Exception as e:  # noqa: BLE001
                logger.error("voice_load_failed", path=str(manifest), error=str(e))

        if self.default_voice and self.default_voice not in self._raw:
            logger.warning(
                "default_voice_missing",
                default_voice=self.default_voice,
                available=sorted(self._raw),
            )

    def _register_manifest(self, manifest_path: str | Path) -> str:
        data = load_voice(manifest_path)
        alias = data["alias"]
        audio_path = self.voices_dir / data["audio_file"]
        if not audio_path.is_file():
            raise FileNotFoundError(f"reference clip missing: {audio_path}")
        data["audio_path"] = str(audio_path)
        self._raw[alias] = data
        logger.info(
            "voice_loaded",
            alias=alias,
            audio=data["audio_file"],
            language=data["language"] or None,
            ref_text_preview=data["ref_text"][:40],
        )
        return alias

    # ------------------------------------------------------------------ public
    def aliases(self) -> list[str]:
        return sorted(self._raw)

    def has(self, alias: str | None) -> bool:
        return bool(alias) and alias in self._raw

    def resolve(self, voice_id: str | None) -> str | None:
        """Return the alias to use: the requested one if known, else the default."""
        if voice_id and voice_id in self._raw:
            return voice_id
        if self.default_voice and self.default_voice in self._raw:
            return self.default_voice
        return None

    def language(self, voice_id: str | None) -> str | None:
        """Return the resolved voice's preferred language, or None if unset."""
        alias = self.resolve(voice_id)
        if alias is None:
            return None
        return self._raw[alias].get("language") or None

    def reference(self, voice_id: str | None) -> tuple[str, str] | None:
        """Return (audio_path, ref_text) for the resolved alias, or None if unavailable."""
        alias = self.resolve(voice_id)
        if alias is None:
            return None
        data = self._raw[alias]
        return data["audio_path"], data["ref_text"]

    def add(self, alias: str, manifest_path: str | Path) -> dict[str, Any]:
        """Hot-register a voice from a manifest (used by the live clone REST endpoint)."""
        self._register_manifest(manifest_path)
        logger.info("voice_registered", alias=alias)
        return self._raw[alias]
