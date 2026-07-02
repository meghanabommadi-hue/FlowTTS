"""Pipeline position: SYNTHESIS ENGINE — text → 24 kHz waveform (GPU core).

Role in pipeline:
  Owns the single k2-fsa/OmniVoice model instance and a dynamic request-level
  batcher. Every concurrent synthesize() call — including each streamed text
  chunk from many WebSocket connections — funnels into one async queue and is
  coalesced into a single batched ``model.generate([...])`` GPU call. This is the
  in-flight batching that drives high throughput.

  server.py → engine.synthesize(text, voice_id, speed, language) → np.ndarray(24k)
            → (resample) → int16 PCM → WebSocket

Batching (mirrors the proven ncodec TTSCodec queue):
  • block for the first request,
  • keep pulling for up to batch_timeout_ms or until max_batch,
  • split by has-voice-prompt (OmniVoice's cloning path keys off item[0]),
  • length-sort each group to minimize padding,
  • run generate() once per group in a single GPU worker thread,
  • resolve each caller's future with its waveform.

Acceleration levers (all opt-in, config-driven, fail-safe):
  • num_step (dominant latency knob), guidance_scale (CFG ≈ 2× compute),
  • torch.compile of the backbone/codec submodules (+ CUDA graphs via
    "reduce-overhead"), bf16 on Hopper, startup warmup to prime everything,
  • the WAV cache in server.py (biggest real-world win for repeated prompts).
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import structlog

from flowtts.core.config import settings, resolve_model_source
from flowtts.voices.registry import VoiceRegistry

logger = structlog.get_logger(__name__)


@dataclass
class _Req:
    """One queued synthesis request."""
    text: str
    prompt: Any            # VoiceClonePrompt or None (auto voice)
    speed: Optional[float]
    language: Optional[str]
    future: "asyncio.Future"


class OmniVoiceEngine:
    """Loads OmniVoice once and serves batched synthesis via an async queue."""

    def __init__(self) -> None:
        self.model = None
        self.registry: VoiceRegistry | None = None
        self.sampling_rate: int = 24000
        self.frame_rate: float = 0.0
        self.engine_info: dict = {}
        self._gen_cfg = None

        cfg = settings.omnivoice
        self._max_batch = cfg.max_batch
        self._batch_timeout = cfg.batch_timeout_ms / 1000.0

        # Single GPU worker thread: one generate() at a time (stable stream
        # ordering) while the batch loop concurrently collects the next batch.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnivoice_gpu")
        self._queue: asyncio.Queue[_Req] | None = None
        self._batch_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ load
    async def initialize(self) -> None:
        if self.model is not None:
            return
        cfg = settings.omnivoice

        # Voice registry first — cheap (loads tiny npz files, no re-encoding).
        self.registry = VoiceRegistry(settings.voices.voices_dir, settings.voices.default_voice)

        import torch
        from omnivoice import OmniVoice
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

        dtype = getattr(torch, cfg.dtype)
        model_source = resolve_model_source()
        logger.info("omnivoice_loading", source=model_source, device=cfg.device, dtype=cfg.dtype)
        self.model = OmniVoice.from_pretrained(
            model_source,
            device_map=cfg.device,
            dtype=dtype,
            load_asr=cfg.load_asr,
            trust_remote_code=cfg.trust_remote_code,
        )

        # Authoritative runtime values — never hardcode (frame_rate is null in config).
        self.sampling_rate = int(getattr(self.model, "sampling_rate", 24000))
        at = getattr(self.model, "audio_tokenizer", None)
        self.frame_rate = float(getattr(getattr(at, "config", None), "frame_rate", 0.0) or 0.0)

        self._gen_cfg = OmniVoiceGenerationConfig(
            num_step=cfg.num_step,
            guidance_scale=cfg.guidance_scale,
            t_shift=cfg.t_shift,
            layer_penalty_factor=cfg.layer_penalty_factor,
            position_temperature=cfg.position_temperature,
            class_temperature=cfg.class_temperature,
            denoise=cfg.denoise,
            audio_chunk_duration=cfg.audio_chunk_duration,
            audio_chunk_threshold=cfg.audio_chunk_threshold,
        )

        self._maybe_compile()

        self.engine_info = {
            "model_source": model_source,
            "device": cfg.device,
            "dtype": cfg.dtype,
            "num_step": cfg.num_step,
            "guidance_scale": cfg.guidance_scale,
            "sampling_rate": self.sampling_rate,
            "frame_rate": self.frame_rate,
            "compiled": cfg.compile_model,
            "voices": self.registry.aliases() if self.registry else [],
        }
        logger.info("omnivoice_ready", **{k: self.engine_info[k] for k in
                    ("sampling_rate", "frame_rate", "num_step", "voices")})

        self._ensure_batch_loop()
        if cfg.warmup:
            await self._warmup()

    def _maybe_compile(self) -> None:
        """torch.compile the backbone (and codec) if enabled — defensive/no-op-safe."""
        cfg = settings.omnivoice
        if not cfg.compile_model:
            return
        import torch

        def _try(obj, attr):
            sub = getattr(obj, attr, None)
            if isinstance(sub, torch.nn.Module):
                try:
                    setattr(obj, attr, torch.compile(sub, mode=cfg.compile_mode))
                    logger.info("torch_compiled", target=attr, mode=cfg.compile_mode)
                    return True
                except Exception as e:  # noqa: BLE001
                    logger.warning("torch_compile_failed", target=attr, error=str(e))
            return False

        # Backbone: whichever attribute name this omnivoice build exposes.
        for attr in ("model", "backbone", "transformer", "generator"):
            if _try(self.model, attr):
                break
        # Codec decoder — compiling the decode path helps the second GPU stage.
        _try(self.model, "audio_tokenizer")

    # ------------------------------------------------------------------ batch loop
    def _ensure_batch_loop(self) -> None:
        if self._batch_task is None or self._batch_task.done():
            self._queue = asyncio.Queue()
            self._batch_task = asyncio.get_event_loop().create_task(self._batch_loop())

    async def _batch_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            batch: list[_Req] = []
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=5.0))
            except asyncio.TimeoutError:
                continue

            deadline = loop.time() + self._batch_timeout
            while len(batch) < self._max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break

            try:
                await loop.run_in_executor(self._executor, self._run_batch, batch)
            except Exception as exc:  # noqa: BLE001
                self._fail(batch, exc)

    def _run_batch(self, batch: list[_Req]) -> None:
        """Split by has-prompt, length-sort, run generate() per group (GPU thread)."""
        with_prompt = [r for r in batch if r.prompt is not None]
        no_prompt = [r for r in batch if r.prompt is None]
        for group in (with_prompt, no_prompt):
            if group:
                self._generate_group(group)

    def _generate_group(self, group: list[_Req]) -> None:
        # Length-sort to keep similar-length items together → less padding waste.
        group.sort(key=lambda r: len(r.text))
        texts = [r.text for r in group]
        prompts = [r.prompt for r in group] if group[0].prompt is not None else None
        speeds = [r.speed for r in group]
        langs = [r.language for r in group]
        # Collapse per-request lists to a scalar/None when uniform (kinder to generate()).
        speed_arg = speeds if any(s is not None for s in speeds) else None
        lang_arg = langs if any(l is not None for l in langs) else None

        try:
            t0 = time.perf_counter()
            audios = self.model.generate(
                text=texts,
                voice_clone_prompt=prompts,
                language=lang_arg,
                speed=speed_arg,
                generation_config=self._gen_cfg,
            )
            dt = (time.perf_counter() - t0) * 1000
            logger.debug("omnivoice_batch", n=len(group), ms=round(dt, 1))
            for r, wav in zip(group, audios):
                if not r.future.done():
                    r.future.get_loop().call_soon_threadsafe(
                        r.future.set_result, np.asarray(wav, dtype=np.float32).reshape(-1)
                    )
        except Exception as exc:  # noqa: BLE001
            if "out of memory" in str(exc).lower():
                try:
                    import torch, gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            self._fail(group, exc)

    @staticmethod
    def _fail(reqs: list[_Req], exc: Exception) -> None:
        for r in reqs:
            if not r.future.done():
                r.future.get_loop().call_soon_threadsafe(r.future.set_exception, exc)

    # ------------------------------------------------------------------ public API
    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        speed: float | None = None,
        language: str | None = None,
    ) -> np.ndarray:
        """Enqueue one synthesis request; return its 24 kHz float32 waveform."""
        if self.model is None:
            raise RuntimeError("OmniVoiceEngine not initialized")
        self._ensure_batch_loop()

        prompt = self.registry.prompt(voice_id) if self.registry else None
        lang = language if language is not None else settings.voices.default_language

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put(_Req(text=text, prompt=prompt, speed=speed, language=lang, future=fut))
        return await fut

    async def _warmup(self) -> None:
        cfg = settings.omnivoice
        sentence = cfg.warmup_sentence
        voice = settings.voices.default_voice if (self.registry and self.registry.has(settings.voices.default_voice)) else None
        n = min(8, cfg.max_batch)
        logger.info("omnivoice_warmup", n=n, voice=voice)
        t0 = time.perf_counter()
        try:
            await asyncio.gather(*[
                self.synthesize(sentence, voice_id=voice) for _ in range(n)
            ])
            logger.info("omnivoice_warmup_done", ms=round((time.perf_counter() - t0) * 1000))
        except Exception as e:  # noqa: BLE001
            logger.warning("omnivoice_warmup_failed", error=str(e))
