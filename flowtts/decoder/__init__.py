"""Audio I/O helpers (waveform → PCM/WAV) + per-call lifecycle bookkeeping.

The old token→PCM ncodec stage is gone: OmniVoice.generate() returns a decoded
waveform directly, so this package now only holds output-format helpers and the
(secondary, Redis-path) decoder lifecycle bookkeeping.
"""

from flowtts.decoder.buffer import TokenBufferManager
from flowtts.decoder.decoder import DecodedAudio, pcm_to_int16_bytes, tensor_to_wav, SAMPLE_RATE

__all__ = [
    "TokenBufferManager",
    "DecodedAudio",
    "pcm_to_int16_bytes",
    "tensor_to_wav",
    "SAMPLE_RATE",
]
