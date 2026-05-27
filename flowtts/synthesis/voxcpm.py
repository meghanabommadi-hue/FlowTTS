"""Pipeline position: SYNTHESIS — VoxCPM2 backend.

Implements BaseSynthesizer for the VoxCPM2 model family.

Unlike Mira, VoxCPM2 does NOT produce discrete speech tokens.  Instead it
runs a diffusion ODE (Continuous Flow Matching) at each decode step and feeds
the output directly into an AudioVAE decoder.  PCM frames come out as
float32 numpy arrays at 48 kHz.

Sequence inside synthesize():
  1. server.generate(target_text, prompt_latents, …)
     → async generator of np.ndarray[float32] PCM chunks + timing dict
  2. Accumulate all chunks
  3. _pcm_to_wav(pcm, sample_rate)  →  16-bit WAV bytes

synthesize_stream() yields each PCM chunk as its own WAV file the moment it
arrives from the VoxCPM2 runner, keeping time-to-first-chunk low.

Initialised once via VoxCpmSynthesizer.initialize(); warm-up is done there.

Requires:  /home/ubuntu/flow_voxcpm  on sys.path (injected at import time).
"""

from __future__ import annotations

import io
import os
import struct
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import structlog

from flowtts.core.config import settings
from flowtts.synthesis.base import BaseSynthesizer, SynthChunk, SynthResult

logger = structlog.get_logger(__name__)

# Inject flow_voxcpm package root once
_VOXCPM_ROOT = str(Path.home() / "flow_voxcpm")
if _VOXCPM_ROOT not in sys.path:
    sys.path.insert(0, _VOXCPM_ROOT)


def _pcm_to_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
    """float32 PCM array → 16-bit WAV bytes."""
    pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    data_size = len(pcm_int16)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                          sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_int16)
    return buf.getvalue()


class VoxCpmSynthesizer(BaseSynthesizer):
    """VoxCPM2 TTS backend.

    Runs text → diffusion latents → AudioVAE → float32 PCM entirely inside
    the nanovllm_voxcpm runner process.  Output is 48 kHz stereo-mono WAV.
    """

    def __init__(self) -> None:
        self._server = None            # AsyncVoxCPM2ServerPool
        self._model_info: dict = {}
        self._sample_rate: int = 48000
        self._prompt_latents: Optional[bytes] = None
        self._prompt_text: str = ""

    # ------------------------------------------------------------------
    # BaseSynthesizer interface
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._server is not None:
            return

        cfg = settings.voxcpm
        model_path = cfg.model_dir
        if not Path(model_path).is_dir():
            raise FileNotFoundError(
                f"VoxCPM2 model directory not found: {model_path}\n"
                f"Set FLOWTTS_VOXCPM__MODEL_DIR to the correct path."
            )

        os.environ["NANOVLLM_QUEUE_COALESCE_MS"] = str(cfg.coalesce_ms)

        logger.info("voxcpm_loading_model", model=model_path)
        from nanovllm_voxcpm import VoxCPM  # noqa: PLC0415

        server = VoxCPM.from_pretrained(
            model=model_path,
            inference_timesteps=cfg.inference_timesteps,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_num_seqs=cfg.max_num_seqs,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enforce_eager=cfg.enforce_eager,
            devices=[0],
        )
        await server.wait_for_ready()

        self._model_info = dict(await server.get_model_info())
        self._sample_rate = self._model_info.get("sample_rate", 48000)
        self._server = server

        # Encode reference audio for voice cloning (optional)
        ref_path = cfg.ref_audio
        if ref_path and os.path.isfile(ref_path):
            try:
                with open(ref_path, "rb") as fh:
                    ref_bytes = fh.read()
                self._prompt_latents = await server.encode_latents(ref_bytes, "wav")
                self._prompt_text = cfg.ref_audio_text
                logger.info("voxcpm_ref_audio_encoded", path=ref_path,
                            bytes=len(self._prompt_latents))
            except Exception as e:
                logger.warning("voxcpm_ref_audio_failed", error=str(e))
                self._prompt_latents = None
                self._prompt_text = ""
        else:
            logger.info("voxcpm_zero_shot_mode", path=ref_path)

        print("\n" + "=" * 60, flush=True)
        print("  VoxCpmSynthesizer — ready", flush=True)
        print("=" * 60, flush=True)
        print(f"  model             : {model_path}", flush=True)
        print(f"  sample_rate (out) : {self._sample_rate} Hz", flush=True)
        print(f"  encoder_sr        : {self._model_info.get('encoder_sample_rate', 'n/a')} Hz", flush=True)
        print(f"  feat_dim          : {self._model_info.get('feat_dim', 'n/a')}", flush=True)
        print(f"  patch_size        : {self._model_info.get('patch_size', 'n/a')}", flush=True)
        print(f"  inference_steps   : {cfg.inference_timesteps}", flush=True)
        print(f"  ref_audio         : {ref_path if self._prompt_latents else 'none (zero-shot)'}", flush=True)
        print("=" * 60 + "\n", flush=True)

        # Warm-up
        sentence = cfg.warmup_sentence
        if sentence:
            logger.info("voxcpm_warmup", sentence=sentence[:40])
            t0 = time.monotonic()
            try:
                await self.synthesize(sentence)
                logger.info("voxcpm_warmup_done",
                            ms=round((time.monotonic() - t0) * 1000))
            except Exception as e:
                logger.warning("voxcpm_warmup_failed", error=str(e))

    async def synthesize(self, text: str) -> SynthResult:
        if self._server is None:
            raise RuntimeError("VoxCpmSynthesizer not initialized")

        cfg = settings.voxcpm
        t0 = time.perf_counter()
        chunks: list[np.ndarray] = []
        first_llm_ms = None
        first_vae_ms = None

        async for chunk in self._server.generate(
            target_text=text,
            prompt_latents=self._prompt_latents,
            prompt_text=self._prompt_text,
            cfg_value=cfg.cfg_value,
            temperature=cfg.temperature,
            max_generate_length=cfg.max_generate_length,
        ):
            if isinstance(chunk, dict):
                first_llm_ms = chunk.get("first_llm_ms")
                first_vae_ms = chunk.get("first_vae_ms")
                continue
            chunks.append(np.asarray(chunk, dtype=np.float32))

        total_s = time.perf_counter() - t0
        pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        wav_bytes = _pcm_to_wav(pcm, self._sample_rate)

        llm_s   = round((first_llm_ms or 0) / 1000, 4)
        vae_s   = round((first_vae_ms  or 0) / 1000, 4)
        return SynthResult(
            wav_bytes=wav_bytes,
            sample_rate=self._sample_rate,
            n_tokens=0,
            llm_s=llm_s,
            decode_s=vae_s,
        )

    async def synthesize_stream(self, text: str) -> AsyncGenerator[SynthChunk, None]:
        if self._server is None:
            raise RuntimeError("VoxCpmSynthesizer not initialized")

        cfg = settings.voxcpm
        pending: Optional[np.ndarray] = None
        first_llm_ms = None
        first_vae_ms = None

        async for chunk in self._server.generate(
            target_text=text,
            prompt_latents=self._prompt_latents,
            prompt_text=self._prompt_text,
            cfg_value=cfg.cfg_value,
            temperature=cfg.temperature,
            max_generate_length=cfg.max_generate_length,
        ):
            if isinstance(chunk, dict):
                first_llm_ms = chunk.get("first_llm_ms")
                first_vae_ms = chunk.get("first_vae_ms")
                continue

            pcm = np.asarray(chunk, dtype=np.float32)
            if pending is not None:
                # Yield the previous chunk as non-final
                yield SynthChunk(
                    wav_bytes=_pcm_to_wav(pending, self._sample_rate),
                    is_final=False,
                    sample_rate=self._sample_rate,
                    n_tokens=0,
                    meta={},
                )
            pending = pcm

        # Last chunk is final and carries timing
        if pending is not None and pending.size > 0:
            pcm_final = pending
        else:
            pcm_final = np.zeros(0, dtype=np.float32)

        yield SynthChunk(
            wav_bytes=_pcm_to_wav(pcm_final, self._sample_rate),
            is_final=True,
            sample_rate=self._sample_rate,
            n_tokens=0,
            meta={
                "first_llm_ms": first_llm_ms,
                "first_vae_ms": first_vae_ms,
            },
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
