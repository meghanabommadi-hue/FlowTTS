"""Pipeline position: AUDIO I/O — OmniVoice waveform (float32) → PCM / WAV bytes.

Role in pipeline:
  OmniVoice.generate() already returns a decoded waveform (there is no separate
  token→PCM stage anymore — the Higgs codec decode happens inside generate()).
  This module only converts that float32 waveform into the wire formats the
  WebSocket stream uses:

    pcm_to_int16_bytes(wav)  → raw little-endian int16 bytes (streaming chunks)
    tensor_to_wav(wav)       → a full .wav container (non-streaming / cache files)

Sample rate: SAMPLE_RATE is the configured OUTPUT rate (settings.output.sample_rate),
24000 by default. Resampling from OmniVoice's native rate (if the output rate is
lower) is done by the caller via processing.audio_processing.resample_audio.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from flowtts.core.config import settings

SAMPLE_RATE = settings.output.sample_rate  # output rate echoed to clients (default 24000)


@dataclass
class DecodedAudio:
    """Decoded audio payload as bytes plus basic metadata."""
    wav_bytes: bytes
    pcm_bytes: bytes
    sample_rate: int
    num_samples: int


def pcm_to_int16_bytes(pcm: np.ndarray) -> bytes:
    """Convert float32 PCM in [-1, 1] to raw int16 little-endian bytes (no WAV header).

    Used for streaming chunks so the client concatenates frames into one
    continuous PCM stream with no header interference.
    """
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    pcm = np.clip(pcm, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


def tensor_to_wav(wav, sample_rate: int = SAMPLE_RATE) -> DecodedAudio:
    """Wrap a float32 waveform (np.ndarray or tensor) in a 16-bit PCM WAV container."""
    wav = np.asarray(wav)
    if wav.dtype == np.float16:
        wav = wav.astype(np.float32)
    wav = wav.reshape(-1)

    pcm_bytes = wav.astype(np.float32).tobytes()

    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=sample_rate, subtype="PCM_16", format="WAV")
    buf.seek(0)
    wav_bytes = buf.read()

    return DecodedAudio(
        wav_bytes=wav_bytes,
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        num_samples=len(wav),
    )
