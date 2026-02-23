"""Pipeline position: AUDIO PIPELINE — orchestrates all post-decode steps.

Role in pipeline:
  Single entry point for post-decode audio processing. Called after
  AudioDecoder.decode_to_wav() when the PCM needs further transformation.

  decoder/decoder.py → DecodedAudio.pcm_bytes (float32)
    → process_audio_pipeline(pcm, 48000, target_sr)
        1. resample_audio() if target_sr ≠ sample_rate
        (future: equalisation, noise gate, normalisation, …)
    → processed float32 numpy array

Currently not wired into the live gateway path — the gateway sends WAV
directly from the decoder. This pipeline is available for callers that
need resampled PCM (e.g. telephony at 8 kHz or 16 kHz).
"""

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
