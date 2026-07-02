"""Pipeline position: VOICE-CLONE SERIALIZATION (pure NumPy — no GPU/torch).

Role in pipeline:
  A cloned voice in OmniVoice is a `VoiceClonePrompt(ref_audio_tokens, ref_text,
  ref_rms)` — the Higgs-codec token grid of a reference clip plus its transcript
  and loudness. We precompute it ONCE (offline) and persist it as a tiny .npz so
  the running server can load it instantly by alias, skipping the codec encoder
  and Whisper ASR on every boot.

  voices/clone.py   → save_voice_npz(...)                      (offline, has torch)
  voices/registry.py→ load_voice_npz(...) → VoiceClonePrompt   (serve time, has torch)

This module is dependency-free (NumPy only) and does NOT use pickle
(`allow_pickle=False`), so npz files are safe to load and the format is
unit-testable on any box. See flowtts/test/test_voice_npz.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 2  # v2 adds optional "language"; v1 files still load (language → "")


def save_voice_npz(
    path: str | Path,
    *,
    ref_audio_tokens: np.ndarray,   # (C, T) codec tokens, C=8 for OmniVoice
    ref_text: str,
    ref_rms: float,
    sample_rate: int,
    frame_rate: float,
    alias: str,
    language: str | None = None,
) -> Path:
    """Persist a voice-clone prompt as a compressed .npz (no pickle).

    ``ref_audio_tokens`` is stored as int16 to stay tiny (a few KB). Codec token
    ids are < 1025 so int16 is lossless. ``language`` is the voice's preferred
    synthesis language (optional).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tokens = np.asarray(ref_audio_tokens)
    if tokens.ndim != 2:
        raise ValueError(f"ref_audio_tokens must be 2-D (C, T); got shape {tokens.shape}")
    if tokens.max(initial=0) > np.iinfo(np.int16).max or tokens.min(initial=0) < np.iinfo(np.int16).min:
        raise ValueError("ref_audio_tokens out of int16 range — cannot store losslessly")

    np.savez_compressed(
        path,
        ref_audio_tokens=tokens.astype(np.int16),
        ref_text=np.asarray(str(ref_text)),      # 0-d '<U' array → no pickle needed
        ref_rms=np.asarray(float(ref_rms), dtype=np.float32),
        sample_rate=np.asarray(int(sample_rate), dtype=np.int32),
        frame_rate=np.asarray(float(frame_rate), dtype=np.float32),
        alias=np.asarray(str(alias)),
        language=np.asarray(str(language or "")),
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
    )
    # np.savez_compressed appends .npz if missing — normalize the returned path.
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def load_voice_npz(path: str | Path) -> dict[str, Any]:
    """Load a voice-clone npz into a plain dict (NumPy only, ``allow_pickle=False``).

    Returns keys: ``ref_audio_tokens`` (np.ndarray int16 (C,T)), ``ref_text`` (str),
    ``ref_rms`` (float), ``sample_rate`` (int), ``frame_rate`` (float),
    ``alias`` (str), ``schema_version`` (int).
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = {"ref_audio_tokens", "ref_text", "ref_rms"} - set(data.files)
        if missing:
            raise ValueError(f"{path.name}: not a valid voice npz (missing {sorted(missing)})")
        version = int(data["schema_version"]) if "schema_version" in data.files else 0
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"{path.name}: schema_version {version} newer than supported {SCHEMA_VERSION}"
            )
        return {
            "ref_audio_tokens": np.asarray(data["ref_audio_tokens"]),
            "ref_text": str(data["ref_text"]),
            "ref_rms": float(data["ref_rms"]),
            "sample_rate": int(data["sample_rate"]) if "sample_rate" in data.files else 0,
            "frame_rate": float(data["frame_rate"]) if "frame_rate" in data.files else 0.0,
            "alias": str(data["alias"]) if "alias" in data.files else path.stem,
            "language": str(data["language"]) if "language" in data.files else "",
            "schema_version": version,
        }
