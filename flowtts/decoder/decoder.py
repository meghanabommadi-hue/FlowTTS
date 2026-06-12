"""Pipeline position: DECODER — audio token string → PCM / WAV bytes.

Role in pipeline:
  Final GPU stage after sglang inference. Converts the speech token string
  produced by the LLM into playable audio using the batched TTSCodec decoder.

  All concurrent decode requests are coalesced by TTSCodec's internal batch
  queue into a single GPU forward pass (ONNX serial + batched PyTorch decoder).

Batch decode API (primary, used by server.py):
  wav_tensor = await codec.decode_async(speech_tokens, context_tokens)
  wav_bytes  = tensor_to_wav(wav_tensor)

Sync decode API (used by warmup / offline tools):
  wav_tensor = codec.decode(speech_tokens, context_tokens)
  wav_bytes  = tensor_to_wav(wav_tensor)

Sample rate: 16000 Hz (batch codec output, no FASR upsampler).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000  # batch codec outputs 16 kHz


@dataclass
class DecodedAudio:
    """Decoded audio payload as bytes plus basic metadata."""
    wav_bytes: bytes
    pcm_bytes: bytes
    sample_rate: int
    num_samples: int


def pcm_to_int16_bytes(pcm: np.ndarray) -> bytes:
    """Convert float32 PCM array to raw int16 bytes (no WAV header).

    Use this for streaming chunks so the client can concatenate frames
    into one continuous PCM stream without header interference.
    """
    pcm = np.clip(pcm, -1.0, 1.0)
    return (pcm * 32767).astype(np.int16).tobytes()


def tensor_to_wav(wav_tensor, sample_rate: int = SAMPLE_RATE) -> DecodedAudio:
    """Convert the tensor returned by TTSCodec.decode / decode_async → DecodedAudio."""
    wav = np.asarray(wav_tensor)
    if wav.dtype == np.float16:
        wav = wav.astype(np.float32)
    wav = wav.squeeze()

    pcm_bytes = wav.tobytes()

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
