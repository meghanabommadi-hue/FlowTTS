"""Pipeline position: VOICE-REFERENCE SERIALIZATION (pure stdlib — no GPU/torch/numpy).

Role in pipeline:
  A cloned voice for Fish Audio S2 Pro is simply a **reference clip + its
  transcript** (+ optional preferred language). Unlike the previous OmniVoice
  stack, we do NOT precompute codec tokens — the sglang backend encodes the clip
  into VQ codes on first use and caches the KV states via RadixAttention.

  So a voice is persisted as two files in voices_dir:
    <alias>.<ext>    the reference audio clip (mono wav, written by the cloner)
    <alias>.json     the manifest: {schema_version, alias, ref_text, language, audio_file}

  voices/clone.py    → save_voice(...)                 (offline or live clone)
  voices/registry.py → load_voice(...) → reference()   (serve time)

This module is dependency-free (stdlib json only) so the format stays importable
and unit-testable on any box. See flowtts/test/test_voice_store.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def manifest_path(voices_dir: str | Path, alias: str) -> Path:
    return Path(voices_dir) / f"{alias}.json"


def save_voice(
    voices_dir: str | Path,
    *,
    alias: str,
    ref_text: str,
    audio_file: str,
    language: str | None = None,
) -> Path:
    """Write a voice manifest `<alias>.json` into voices_dir.

    ``audio_file`` is the reference clip's filename (basename) inside voices_dir —
    stored relative so the manifest stays portable across mount points. The clip
    itself must already have been written next to the manifest by the caller.
    """
    if not alias:
        raise ValueError("alias is required")
    if not ref_text or not ref_text.strip():
        raise ValueError("ref_text is required (no auto-transcription)")
    if not audio_file:
        raise ValueError("audio_file is required")

    voices_dir = Path(voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_path(voices_dir, alias)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "alias": alias,
        "ref_text": str(ref_text),
        "language": str(language or ""),
        "audio_file": Path(audio_file).name,  # basename only — resolved against voices_dir
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_voice(path: str | Path) -> dict[str, Any]:
    """Load a voice manifest `<alias>.json` into a plain dict.

    Returns keys: ``alias`` (str), ``ref_text`` (str), ``language`` (str, ""=unset),
    ``audio_file`` (str, basename), ``schema_version`` (int). Validates required
    fields; raises ValueError on a malformed/newer-schema manifest.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: manifest is not a JSON object")

    version = int(data.get("schema_version", 0))
    if version > SCHEMA_VERSION:
        raise ValueError(f"{path.name}: schema_version {version} newer than supported {SCHEMA_VERSION}")

    ref_text = data.get("ref_text")
    audio_file = data.get("audio_file")
    if not ref_text or not str(ref_text).strip():
        raise ValueError(f"{path.name}: missing ref_text")
    if not audio_file:
        raise ValueError(f"{path.name}: missing audio_file")

    return {
        "alias": str(data.get("alias") or path.stem),
        "ref_text": str(ref_text),
        "language": str(data.get("language") or ""),
        "audio_file": Path(str(audio_file)).name,
        "schema_version": version,
    }
