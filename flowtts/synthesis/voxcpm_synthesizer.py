"""Pipeline position: SYNTHESIS — text → PCM audio (VoxCPM2 path).

Role in pipeline:
  Drop-in alternative to FlowTtsSynthesizer for the VoxCPM2 model family.
  Instead of producing a speech-token string that is decoded separately by
  ncodec, VoxCPM2 internally runs its own AudioVAE decoder and streams raw
  float32 PCM chunks directly.

  server.py selects between the two synthesizers at startup depending on
  settings.model_type ("mira" → FlowTtsSynthesizer, "voxcpm" → VoxCpmSynthesizer).

Sequence inside VoxCpmSynthesizer.synthesize_stream():
  1. server.generate(target_text, prompt_latents, prompt_text, ...)
     → async generator yielding np.ndarray[float32] PCM chunks + timing dict
  2. Accumulate + convert chunks to WAV bytes on the fly
  3. Yield (chunk_pcm, is_final, timing_info) tuples to the caller

Initialisation (done once, lazy):
  • Loads VoxCPM2 via nanovllm_voxcpm.VoxCPM.from_pretrained().
  • Encodes optional reference audio to latent bytes (for voice cloning).
  • Warm-up: one generate() pass to prime CUDA graphs before real traffic.

Key differences from Mira/sglang path:
  • No speech-token string — audio comes out as float32 PCM frames.
  • Sample rate is 48 kHz (VAE output), not 16 kHz.
  • No external ncodec TTSCodec needed.
  • Decoding is already done inside VoxCPM2Runner.run() — server.py does not
    call TTSCodec.decode_async() for this model type.
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

logger = structlog.get_logger(__name__)

# Ensure flow_voxcpm is on the import path.
# The nanovllm_voxcpm package lives at /home/ubuntu/flow_voxcpm.
_VOXCPM_ROOT = str(Path.home() / "flow_voxcpm")
if _VOXCPM_ROOT not in sys.path:
    sys.path.insert(0, _VOXCPM_ROOT)


def _pcm_to_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert a float32 PCM array to a 16-bit WAV byte string.

    Args:
        pcm: 1-D float32 array in [-1, 1].
        sample_rate: Audio sample rate in Hz.

    Returns:
        WAV file bytes (RIFF/PCM-16 format).
    """
    # Clamp and convert to int16
    pcm_clipped = np.clip(pcm, -1.0, 1.0)
    pcm_int16 = (pcm_clipped * 32767).astype(np.int16)
    pcm_bytes = pcm_int16.tobytes()

    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack(
        "<IHHIIHH",
        16,           # chunk size
        1,            # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    ))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_bytes)
    return buf.getvalue()


class VoxCpmSynthesizer:
    """Loads AsyncVoxCPM2ServerPool once; synthesizes text → PCM/WAV.

    This synthesizer wraps the VoxCPM2 model from flow_voxcpm and exposes the
    same interface surface that server.py expects:
        - initialize()           async, called once at startup
        - synthesize(text)       async, returns (wav_bytes, sample_rate)
        - synthesize_stream(text) async-generator, yields (chunk_pcm, is_final, meta)
    """

    def __init__(self) -> None:
        self._server = None           # AsyncVoxCPM2ServerPool
        self._model_info: dict = {}
        self._sample_rate: int = 48000
        self._prompt_latents: Optional[bytes] = None
        self._prompt_text: str = ""

    async def initialize(self) -> None:
        """Load VoxCPM2 model. Safe to call multiple times — only loads once."""
        if self._server is not None:
            return

        cfg = settings.voxcpm
        model_path = cfg.model_dir

        if not Path(model_path).is_dir():
            raise FileNotFoundError(
                f"VoxCPM2 model directory not found: {model_path}\n"
                f"Set FLOWTTS_VOXCPM__MODEL_DIR to the correct path."
            )

        # Set coalescing window before the server process spawns
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

        logger.info("voxcpm_waiting_for_ready")
        await server.wait_for_ready()
        logger.info("voxcpm_model_ready")

        self._model_info = dict(await server.get_model_info())
        self._sample_rate = self._model_info.get("sample_rate", 48000)
        self._server = server

        # --- Encode reference audio for voice cloning ----------------------
        ref_path = cfg.ref_audio
        if ref_path and os.path.isfile(ref_path):
            try:
                with open(ref_path, "rb") as f:
                    ref_bytes = f.read()
                self._prompt_latents = await server.encode_latents(ref_bytes, "wav")
                self._prompt_text = cfg.ref_audio_text
                logger.info("voxcpm_ref_audio_encoded", path=ref_path,
                            latent_bytes=len(self._prompt_latents),
                            has_text=bool(self._prompt_text))
            except Exception as e:
                logger.warning("voxcpm_ref_audio_failed", error=str(e),
                               note="running in zero-shot mode")
                self._prompt_latents = None
                self._prompt_text = ""
        else:
            logger.info("voxcpm_ref_audio_not_found",
                        path=ref_path, note="zero-shot mode")

        # Print runtime summary
        print("\n" + "=" * 60, flush=True)
        print("  VoxCPM2 — Engine ready", flush=True)
        print("=" * 60, flush=True)
        print(f"  model_path          : {model_path}", flush=True)
        print(f"  sample_rate (out)   : {self._sample_rate} Hz", flush=True)
        print(f"  encoder_sample_rate : {self._model_info.get('encoder_sample_rate', 'n/a')} Hz", flush=True)
        print(f"  feat_dim            : {self._model_info.get('feat_dim', 'n/a')}", flush=True)
        print(f"  patch_size          : {self._model_info.get('patch_size', 'n/a')}", flush=True)
        print(f"  inference_timesteps : {cfg.inference_timesteps}", flush=True)
        print(f"  ref_audio           : {ref_path if self._prompt_latents else 'none (zero-shot)'}", flush=True)
        print(f"  ref_audio_text      : {self._prompt_text[:60]!r}" if self._prompt_text else "  ref_audio_text      : (none)", flush=True)
        print("=" * 60 + "\n", flush=True)

        # --- Warm-up -------------------------------------------------------
        sentence = cfg.warmup_sentence
        if sentence:
            logger.info("voxcpm_warmup_start", sentence=sentence[:40])
            t0 = time.monotonic()
            try:
                async for _ in self._generate_chunks(sentence):
                    pass
                logger.info("voxcpm_warmup_done",
                            elapsed_ms=round((time.monotonic() - t0) * 1000))
            except Exception as e:
                logger.warning("voxcpm_warmup_failed", error=str(e))

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    async def _generate_chunks(
        self,
        text: str,
    ) -> AsyncGenerator[tuple[np.ndarray, dict], None]:
        """Yield (pcm_chunk: float32 ndarray, timing: dict) from VoxCPM2.

        timing dict is non-empty only on the final sentinel item (pcm is empty).
        """
        if self._server is None:
            raise RuntimeError("VoxCpmSynthesizer not initialized")

        cfg = settings.voxcpm
        timing: dict = {}

        async for chunk in self._server.generate(
            target_text=text,
            prompt_latents=self._prompt_latents,
            prompt_text=self._prompt_text,
            cfg_value=cfg.cfg_value,
            temperature=cfg.temperature,
            max_generate_length=cfg.max_generate_length,
        ):
            if isinstance(chunk, dict):
                # timing sentinel — not actual audio
                timing = chunk
                continue
            yield np.asarray(chunk, dtype=np.float32), {}

        # Yield an empty chunk to carry the timing info
        yield np.empty(0, dtype=np.float32), timing

    # -----------------------------------------------------------------------
    # Public API (matches server.py expectations)
    # -----------------------------------------------------------------------

    async def synthesize(self, text: str) -> tuple[bytes, int]:
        """Synthesize *text* fully, return (wav_bytes, sample_rate).

        Accumulates all PCM chunks from VoxCPM2 and encodes as a single WAV.
        Used by the non-streaming (full-response) server path.
        """
        if self._server is None:
            raise RuntimeError("VoxCpmSynthesizer not initialized")

        t0 = time.monotonic()
        chunks: list[np.ndarray] = []
        first_llm_ms = None
        first_vae_ms = None

        async for pcm_chunk, timing in self._generate_chunks(text):
            if pcm_chunk.size > 0:
                chunks.append(pcm_chunk)
            if timing:
                first_llm_ms = timing.get("first_llm_ms")
                first_vae_ms = timing.get("first_vae_ms")

        if not chunks:
            # Return silent WAV if nothing was generated
            pcm_full = np.zeros(0, dtype=np.float32)
        else:
            pcm_full = np.concatenate(chunks)

        wav_bytes = _pcm_to_wav(pcm_full, self._sample_rate)
        total_ms = round((time.monotonic() - t0) * 1000)
        audio_s = pcm_full.size / self._sample_rate
        rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 0.0

        logger.info(
            "voxcpm_synthesize_done",
            text_preview=text[:40],
            total_ms=total_ms,
            samples=pcm_full.size,
            audio_s=round(audio_s, 3),
            rtf=round(rtf, 3),
            first_llm_ms=first_llm_ms,
            first_vae_ms=first_vae_ms,
        )
        return wav_bytes, self._sample_rate

    async def synthesize_stream(
        self, text: str
    ) -> AsyncGenerator[tuple[bytes, bool, dict], None]:
        """Async-generator streaming version of synthesize().

        Yields:
            (wav_chunk_bytes, is_final, meta_dict)

        Each wav_chunk_bytes is a complete WAV file containing the samples for
        that chunk (so the client can play it immediately).
        is_final is True only on the last yield.
        meta_dict carries timing info on the final yield.
        """
        if self._server is None:
            raise RuntimeError("VoxCpmSynthesizer not initialized")

        cfg = settings.voxcpm
        timing: dict = {}
        last_chunk: Optional[np.ndarray] = None
        total_samples = 0

        async for pcm_chunk, chunk_timing in self._generate_chunks(text):
            if chunk_timing:
                timing = chunk_timing

            if pcm_chunk.size > 0:
                if last_chunk is not None:
                    # Yield the previous chunk as non-final
                    wav = _pcm_to_wav(last_chunk, self._sample_rate)
                    yield wav, False, {}
                last_chunk = pcm_chunk
                total_samples += pcm_chunk.size

        # Yield the last accumulated chunk as final (or empty WAV if nothing)
        if last_chunk is not None and last_chunk.size > 0:
            wav = _pcm_to_wav(last_chunk, self._sample_rate)
        else:
            wav = _pcm_to_wav(np.zeros(0, dtype=np.float32), self._sample_rate)
        yield wav, True, timing

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
