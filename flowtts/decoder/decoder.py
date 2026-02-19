"""Audio-token decoder using ``ncodec.TTSCodec``.

This module wraps the sample ``decoder.py`` into a reusable class that
decodes a semantic token string (plus context tokens) into 16‑bit PCM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import io

import numpy as np
import soundfile as sf
from ncodec.codec import TTSCodec


CONTEXT_TOKENS = (
    "<|context_token_3991|><|context_token_1250|><|context_token_2828|>"
    "<|context_token_3303|><|context_token_1187|><|context_token_3021|>"
    "<|context_token_355|><|context_token_3767|><|context_token_3663|>"
    "<|context_token_837|><|context_token_731|><|context_token_3656|>"
    "<|context_token_757|><|context_token_3360|><|context_token_3250|>"
    "<|context_token_3626|><|context_token_1244|><|context_token_526|>"
    "<|context_token_3829|><|context_token_205|><|context_token_1619|>"
    "<|context_token_268|><|context_token_4024|><|context_token_3375|>"
    "<|context_token_3032|><|context_token_2180|><|context_token_3278|>"
    "<|context_token_1609|><|context_token_3685|><|context_token_1359|>"
    "<|context_token_2817|><|context_token_3999|>"
)


@dataclass
class DecodedAudio:
    """Decoded audio payload as bytes plus basic metadata."""

    wav_bytes: bytes
    sample_rate: int


class AudioDecoder:
    """Decode audio tokens to PCM using TTSCodec."""

    def __init__(self, sample_rate: int = 48000) -> None:
        self._codec = TTSCodec()
        self._sample_rate = sample_rate

    def decode_to_wav(
        self, audio_tokens: str, context_tokens: str | None = None
    ) -> DecodedAudio:
        """Decode a semantic token string into WAV bytes."""
        ctx = context_tokens or CONTEXT_TOKENS

        audio = self._codec.decode(audio_tokens, ctx)
        audio = np.asarray(audio)
        if audio.dtype == np.float16:
            audio = audio.astype(np.float32)

        audio = audio.squeeze()

        wav_io = io.BytesIO()
        sf.write(
            wav_io,
            audio,
            samplerate=self._sample_rate,
            subtype="PCM_16",
            format="WAV",
        )
        wav_io.seek(0)

        return DecodedAudio(wav_bytes=wav_io.read(), sample_rate=self._sample_rate)


decoder = AudioDecoder()

