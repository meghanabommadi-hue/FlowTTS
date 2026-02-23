"""Pipeline position: DECODER — audio token string → PCM / WAV bytes.

Role in pipeline:
  Final GPU stage after sglang inference. Converts the speech token string
  produced by the LLM into playable audio.

  Two modes of use:
    1. Inline (gateway decodes):  api/websockets.py calls AudioDecoder directly
       (decoder.enabled=True, no separate decoder process needed).

    2. Standalone worker (DecoderWorker):  a dedicated process subscribes to the
       Redis results channel (flowtts:audio:{call_id}), decodes each job, and
       publishes WAV to flowtts:decoded:{call_id}. The gateway then only listens
       on the decoded channel and forwards audio to the WebSocket client.

  Other callers:
    • test/codec_server.py  (_codec_process, standalone decoder pool)
    • test/decode_json.py   (batch offline decoding from JSON files)

Decode sequence inside AudioDecoder.decode_to_wav():
  1. ncodec.TTSCodec.decode(audio_tokens, context_tokens)
       → runs three ONNX/PyTorch stages inside ncodec:
           processer.onnx   — context + speech token preprocessing
           detokenizer      — token sequence → low-res waveform (PyTorch)
           upsampler (FASR) — 24 kHz → 48 kHz super-resolution
       → returns float32 tensor
  2. float16 → float32 cast if needed
  3. (optional) soundfile.write → WAV bytes  [to_wav=True]
  4. Return DecodedAudio(wav_bytes, pcm_bytes, sample_rate, num_samples)

DecoderWorker Redis channels:
  Subscribes to:  flowtts:audio:*      (pattern — catches all call_ids)
  Publishes to:   flowtts:decoded:{call_id}

Global singleton `decoder`:
  Instantiated at module import time and reused across all calls within
  the same process. For parallel decoding across processes, each subprocess
  creates its own instance (see test/codec_server.py).
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Tuple

import io

import numpy as np
import soundfile as sf
import structlog
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

    wav_bytes: bytes        # WAV-encoded bytes (empty if to_wav=False)
    pcm_bytes: bytes        # raw float32 PCM bytes (always present)
    sample_rate: int
    num_samples: int


class AudioDecoder:
    """Decode audio tokens to PCM using TTSCodec."""

    def __init__(self, sample_rate: int = 48000) -> None:
        self._codec = TTSCodec()
        self._sample_rate = sample_rate

    def decode_to_wav(
        self,
        audio_tokens: str,
        context_tokens: str | None = None,
        to_wav: bool = True,
    ) -> DecodedAudio:
        """Decode a semantic token string into PCM, optionally WAV-encoded.

        Args:
            audio_tokens:   Speech token string from sglang.
            context_tokens: Speaker context tokens. Uses default if None.
            to_wav:         If True, encode PCM as WAV bytes (adds ~1–5ms).
                            If False, wav_bytes is empty — decode_s measures
                            only the ncodec decode time.
        """
        ctx = context_tokens or CONTEXT_TOKENS

        audio = self._codec.decode(audio_tokens, ctx)
        audio = np.asarray(audio)
        if audio.dtype == np.float16:
            audio = audio.astype(np.float32)
        audio = audio.squeeze()

        pcm_bytes = audio.tobytes()  # raw float32 LE bytes

        wav_bytes = b""
        if to_wav:
            wav_io = io.BytesIO()
            sf.write(
                wav_io,
                audio,
                samplerate=self._sample_rate,
                subtype="PCM_16",
                format="WAV",
            )
            wav_io.seek(0)
            wav_bytes = wav_io.read()

        return DecodedAudio(
            wav_bytes=wav_bytes,
            pcm_bytes=pcm_bytes,
            sample_rate=self._sample_rate,
            num_samples=len(audio),
        )


decoder = AudioDecoder()

_logger = structlog.get_logger(__name__)


class DecoderWorker:
    """Standalone Redis-backed decoder worker.

    Subscribes to the LLM results channel (flowtts:audio:*), decodes each
    audio_tokens payload via AudioDecoder, and publishes the WAV result to
    flowtts:decoded:{call_id} for the gateway to forward to the client.

    Usage::

        worker = DecoderWorker()
        await worker.run()   # blocks; call worker.stop() from another task to exit
    """

    def __init__(self) -> None:
        self._audio_decoder = AudioDecoder()
        self._running = False
        self._redis: object | None = None  # redis.asyncio.Redis, typed as object to avoid import at module level

    async def _connect(self) -> None:
        import redis.asyncio as aioredis
        from flowtts.core.config import settings

        cfg = settings.redis
        url = f"redis://{cfg.host}:{cfg.port}/{cfg.db}"
        self._redis = await aioredis.from_url(  # type: ignore[attr-defined]
            url,
            password=cfg.password,
            decode_responses=False,
        )
        _logger.info("decoder_worker_redis_connected", host=cfg.host, port=cfg.port)

    async def run(self) -> None:
        """Main loop: subscribe to LLM results, decode, publish WAV."""
        from flowtts.core.config import settings

        await self._connect()
        self._running = True

        pattern = f"{settings.redis.results_channel_prefix}:*"
        pubsub = self._redis.pubsub()  # type: ignore[union-attr]
        await pubsub.psubscribe(pattern)
        _logger.info("decoder_worker_subscribed", pattern=pattern)

        try:
            async for message in pubsub.listen():
                if not self._running:
                    break
                if message["type"] != "pmessage":
                    continue
                await self._handle_message(message)
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.punsubscribe(pattern)
            await pubsub.aclose()  # type: ignore[func-returns-value]
            _logger.info("decoder_worker_stopped")

    async def _handle_message(self, message: dict) -> None:
        from flowtts.core.config import settings

        try:
            data = json.loads(message["data"])
        except (json.JSONDecodeError, KeyError):
            _logger.warning("decoder_worker_bad_message")
            return

        call_id = data.get("call_id", "")
        text_id = data.get("text_id", "")
        audio_tokens = data.get("audio_tokens", "")
        is_final = data.get("is_final", True)
        llm_s = data.get("llm_s")

        if not audio_tokens:
            _logger.warning("decoder_worker_empty_tokens", call_id=call_id, text_id=text_id)
            return

        # Run decode in executor so the event loop stays unblocked
        loop = asyncio.get_running_loop()
        t0 = time.time()
        decoded = await loop.run_in_executor(
            None,
            lambda: self._audio_decoder.decode_to_wav(audio_tokens, to_wav=True),
        )
        decode_s = round(time.time() - t0, 4)

        audio_b64 = base64.b64encode(decoded.wav_bytes).decode("ascii")

        payload = {
            "call_id": call_id,
            "text_id": text_id,
            "audio_base64": audio_b64,
            "sample_rate": decoded.sample_rate,
            "is_final": is_final,
            "llm_s": llm_s,
            "decode_s": decode_s,
        }

        channel = f"{settings.redis.decoded_channel_prefix}:{call_id}"
        await self._redis.publish(channel, json.dumps(payload))  # type: ignore[union-attr]

        _logger.info(
            "decoder_worker_job_done",
            call_id=call_id,
            text_id=text_id,
            decode_s=decode_s,
        )

    def stop(self) -> None:
        """Signal the worker to stop after the current message."""
        self._running = False


async def run_decoder_worker() -> None:
    """Entry point for running DecoderWorker as a standalone process."""
    worker = DecoderWorker()
    await worker.run()


def main() -> None:
    asyncio.run(run_decoder_worker())


if __name__ == "__main__":
    main()

