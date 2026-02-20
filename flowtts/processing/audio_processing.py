"""Audio processing utilities for post-decoder PCM audio.

Contains both resampling and crossfade helpers so callers only need to
import from a single module.
"""

from __future__ import annotations

import numpy as np


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample PCM from orig_sr to target_sr (simple linear interpolation stub).

    For production, use scipy.signal.resample or librosa.resample.
    """
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_len, dtype=np.float64)
    return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)


def crossfade(
    chunk_a: np.ndarray,
    chunk_b: np.ndarray,
    fade_samples: int = 0,
) -> np.ndarray:
    """Concatenate two PCM chunks with optional crossfade.

    If fade_samples > 0, overlap the end of chunk_a with the start of chunk_b
    using a linear crossfade. Otherwise concatenate directly.
    """
    if fade_samples <= 0:
        return np.concatenate([chunk_a, chunk_b])
    if fade_samples >= min(len(chunk_a), len(chunk_b)):
        fade_samples = min(len(chunk_a), len(chunk_b)) // 2
    end_a = chunk_a[-fade_samples:]
    start_b = chunk_b[:fade_samples]
    fade = np.linspace(1, 0, fade_samples, dtype=chunk_a.dtype)
    overlapped = end_a * fade + start_b * (1 - fade)
    return np.concatenate([chunk_a[:-fade_samples], overlapped, chunk_b[fade_samples:]])

