"""Post-decoder audio processing only.

This package handles PCM after the decoder: resampling, crossfading, and
pipeline orchestration. Token buffering lives in flowtts.decoder.buffer
(before the decoder).
"""

from flowtts.processing.audio_processing import resample_audio, crossfade
from flowtts.processing.pipeline import process_audio_pipeline

__all__ = ["resample_audio", "crossfade", "process_audio_pipeline"]
