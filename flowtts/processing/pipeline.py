"""Orchestrate post-decoder audio: resample + crossfade (and future effects)."""

from __future__ import annotations

import numpy as np

from flowtts.processing.audio_processing import resample_audio


def process_audio_pipeline(
    pcm: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = 16000,
) -> np.ndarray:
    """Run resample (and optionally crossfade) on decoded PCM."""
    out = resample_audio(pcm, sample_rate, target_sample_rate)
    # Single-chunk path: no crossfade. Caller can use crossfade() when
    # concatenating multiple decoded chunks.
    return out
