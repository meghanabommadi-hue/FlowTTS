"""Pipeline position: POST-DECODE AUDIO PROCESSING — PCM → processed PCM.

Role in pipeline:
  Operates on raw float32 PCM arrays after ncodec decoding.
  Called by processing/pipeline.py, which is invoked by any consumer that
  needs audio at a different sample rate or wants smooth multi-utterance audio.

  ncodec outputs 48 kHz float32 PCM
    → resample_audio(audio, 48000, target_sr)  if target_sr ≠ 48000
    → crossfade(prev_chunk, new_chunk)          for seamless call-centre streams

Functions:
  resample_audio  — linear interpolation; fast but approximate.
                    For higher quality use scipy.signal.resample or librosa.
  crossfade       — appends two PCM chunks with optional linear overlap region.
                    fade_samples=0 → plain concatenation (current default).
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


def fade_out(
    audio: np.ndarray,
    fade_samples: int,
) -> np.ndarray:
    """Apply a linear fade-out to the last *fade_samples* of *audio*.

    Useful for suppressing codec tail noise on non-final streaming chunks.
    If fade_samples >= len(audio) the entire array is faded.
    """
    if fade_samples <= 0 or len(audio) == 0:
        return audio
    fade_samples = min(fade_samples, len(audio))
    out = audio.copy()
    out[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=audio.dtype)
    return out


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

