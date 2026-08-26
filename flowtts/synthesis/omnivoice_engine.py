"""Pipeline position: SYNTHESIS ENGINE — text → 24 kHz waveform (GPU core).

Role in pipeline:
  Owns the single OmniVoice model instance, the accelerated backbone, and the
  dynamic request-level batcher. Every concurrent synthesize() call — including
  each streamed chunk from every WebSocket and HTTP connection — funnels into
  one async queue and is coalesced into batched ``model.generate([...])`` calls.

      api / ws → engine.synthesize(text, **params) → np.ndarray @ 24 kHz

Batching, and why it is grouped the way it is:
  ``generate()`` takes ONE generation_config for the whole call, pads every item
  to the longest one, and keys the voice-clone path off ``item[0]``. So a batch
  is only sound when its members agree on all three of those. Requests are
  therefore grouped by

      (generation config, has-voice-prompt, length bucket)

  and each group is additionally capped by a total-frames budget. Length
  bucketing is the part that matters most under mixed traffic: without it a 1 s
  first chunk batched with a 20 s paragraph pays the paragraph's cost, and the
  TTFB that chunk existed to protect is gone.

Acceleration:
  ``flowtts.trt.patch_model`` replaces ``llm.forward`` with a TensorRT /
  TensorRT-LLM / compiled-torch backbone after validating it against the real
  module (github.com/tlitech/omnivoice-trtllm's approach). Everything else in
  OmniVoice — embeddings, audio heads, the unmasking loop, the Higgs codec —
  runs upstream's code untouched, which is what keeps the audio identical.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import structlog

from flowtts.core.config import resolve_model_source, settings
from flowtts.synthesis.chunker import estimate_duration
from flowtts.voices.npz_io import save_voice_npz
from flowtts.voices.registry import VoiceRegistry

logger = structlog.get_logger(__name__)


@dataclass
class GenParams:
    """The effective OmniVoice parameters for one request.

    Starts from ``settings.generation`` and is overlaid with whatever the caller
    sent. ``batch_key`` is what decides who can share a ``generate()`` call.
    """

    config: dict = field(default_factory=dict)   # OmniVoiceGenerationConfig fields
    speed: float | None = None
    duration: float | None = None
    normalize_text: bool = False

    @classmethod
    def build(cls, overrides: dict | None = None) -> "GenParams":
        defaults = settings.generation
        cfg = defaults.as_generation_kwargs()
        speed, duration = defaults.speed, defaults.duration
        normalize = defaults.normalize_text

        for key, value in (overrides or {}).items():
            if value is None:
                continue
            if key == "speed":
                speed = float(value)
            elif key == "duration":
                duration = float(value)
            elif key == "normalize_text":
                normalize = bool(value)
            elif key in cfg:
                cfg[key] = value

        return cls(config=cfg, speed=speed, duration=duration, normalize_text=normalize)

    def for_streaming(self) -> "GenParams":
        """A copy with OmniVoice's own edge processing disabled.

        Three reasons, all of which matter per chunk:

        * ``pad_duration``/``fade_duration`` add ~100 ms of silence and a fade to
          each end of every clip. Across chunks that is a ~200 ms hole at every
          seam, and the stitcher's first job would be cutting back out what we
          just paid the model to add.
        * ``postprocess_output`` runs pydub's ``split_on_silence`` over the clip.
          When a chunk comes back unusually quiet — which a low ``num_step``
          does occasionally produce on a short utterance — every segment falls
          under its -50 dBFS gate and it returns an EMPTY array, so the caller
          gets a 200 with no audio in it.
        * that same pydub round-trip is slow, and it is on the streaming path.

        The stitcher trims edges itself, with a windowed RMS gate that keeps a
        margin instead of returning nothing.
        """
        if not settings.streaming.disable_model_edge_processing:
            return self
        cfg = dict(self.config)
        cfg["pad_duration"] = 0.0
        cfg["fade_duration"] = 0.0
        cfg["postprocess_output"] = False
        return GenParams(config=cfg, speed=self.speed, duration=self.duration,
                         normalize_text=self.normalize_text)

    def as_retry(self) -> "GenParams":
        """A more conservative copy, for retrying a degenerate generation.

        Restores deterministic position selection and raises the step count:
        an all-but-silent result is the sampler having wandered, and both of
        these make the second attempt reproducible rather than another dice roll.
        """
        cfg = dict(self.config)
        cfg["position_temperature"] = 0.0
        cfg["class_temperature"] = 0.0
        cfg["num_step"] = max(int(cfg.get("num_step", 8)), 8)
        # A zero guidance_scale is itself the most common cause of a silent
        # generation on this model, so a retry that kept it would just fail again.
        if not cfg.get("guidance_scale"):
            cfg["guidance_scale"] = 2.0
        return GenParams(config=cfg, speed=self.speed, duration=self.duration,
                         normalize_text=self.normalize_text)

    @property
    def batch_key(self) -> str:
        """Requests sharing this key may share one generate() call."""
        return hashlib.sha1(
            json.dumps(self.config, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]


# Peak amplitude below which a generated clip carries no speech. OmniVoice
# normalizes output to ~0.5 peak, so a real utterance is three orders of
# magnitude above this; anything under it is a failed generation, not quiet audio.
_SILENCE_PEAK = 1e-3


def is_silent(wav: np.ndarray) -> bool:
    """True if a waveform is empty or carries no audible content.

    Used both to decide whether a generation needs retrying and, in the API
    layer, to decide whether a stream has anything worth sending. Length alone
    is not enough: trimming a silent clip returns a short *non-empty* array, and
    streaming that is indistinguishable to the caller from a working request.
    """
    if wav is None or wav.size == 0:
        return True
    return float(np.abs(wav).max()) < _SILENCE_PEAK


@dataclass
class _Req:
    """One queued synthesis request."""

    text: str
    prompt: Any                  # VoiceClonePrompt or None (auto/design voice)
    language: Optional[str]
    instruct: Optional[str]
    params: GenParams
    est_frames: int
    future: "asyncio.Future"
    queued_at: float


class OmniVoiceEngine:
    """Loads OmniVoice once and serves batched synthesis via an async queue."""

    def __init__(self) -> None:
        self.model = None
        self.registry: VoiceRegistry | None = None
        self.sampling_rate: int = 24000
        self.frame_rate: float = 0.0
        self.engine_info: dict = {}
        self.backbone: Any = None

        cfg = settings.omnivoice
        self._max_batch = cfg.max_batch
        self._batch_timeout = cfg.batch_timeout_ms / 1000.0
        self._max_batch_frames = cfg.max_batch_frames
        self._bucket_ratio = cfg.length_bucket_ratio

        # One GPU worker thread: a single generate() at a time keeps stream
        # ordering deterministic while the batch loop collects the next batch.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnivoice_gpu")
        self._queue: asyncio.Queue[_Req] | None = None
        self._batch_task: asyncio.Task | None = None
        self._gen_config_cls = None

        # Rolling counters exposed on /v1/stats.
        self.stats = {
            "requests": 0, "batches": 0, "batched_items": 0,
            "errors": 0, "gpu_ms": 0.0, "queue_ms": 0.0,
            "degenerate": 0, "retries": 0,
        }

    # ------------------------------------------------------------------ load
    async def initialize(self) -> None:
        if self.model is not None:
            return
        cfg = settings.omnivoice

        # Voice registry first — cheap (tiny npz files, no re-encoding).
        self.registry = VoiceRegistry(settings.voices.voices_dir, settings.voices.default_voice)

        import torch
        from omnivoice import OmniVoice
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

        self._gen_config_cls = OmniVoiceGenerationConfig
        model_source = resolve_model_source()
        logger.info("omnivoice_loading", source=model_source, device=cfg.device, dtype=cfg.dtype)
        self.model = OmniVoice.from_pretrained(
            model_source,
            device_map=cfg.device,
            dtype=getattr(torch, cfg.dtype),
            load_asr=cfg.load_asr,
            trust_remote_code=cfg.trust_remote_code,
        )

        # Authoritative runtime values — never hardcode.
        self.sampling_rate = int(getattr(self.model, "sampling_rate", 24000))
        tokenizer = getattr(self.model, "audio_tokenizer", None)
        self.frame_rate = float(
            getattr(getattr(tokenizer, "config", None), "frame_rate", 0.0) or 0.0
        )

        backbone_info = self._patch_backbone()

        self.engine_info = {
            "model_source": model_source,
            "device": cfg.device,
            "dtype": cfg.dtype,
            "sampling_rate": self.sampling_rate,
            "frame_rate": self.frame_rate,
            "max_batch": self._max_batch,
            "max_batch_frames": self._max_batch_frames,
            "backbone": backbone_info,
            "generation_defaults": settings.generation.model_dump(),
            "voices": self.registry.aliases() if self.registry else [],
        }
        logger.info("omnivoice_ready", sampling_rate=self.sampling_rate,
                    frame_rate=self.frame_rate, backbone=backbone_info.get("backend"),
                    voices=len(self.engine_info["voices"]))

        self._ensure_batch_loop()
        if cfg.warmup:
            await self._warmup()

    def _patch_backbone(self) -> dict:
        """Install the configured accelerated backbone; never fatal."""
        try:
            from flowtts.trt import patch_model

            result = patch_model(self.model, settings.omnivoice)
            self.backbone = getattr(self.model, "_flowtts_backbone_patch", None)
            if result.fell_back:
                logger.warning("backbone_fallback", reason=result.reason,
                               requested=settings.omnivoice.backbone_backend)
            else:
                logger.info("backbone_ready", **result.as_dict())
            return result.as_dict()
        except Exception as exc:  # noqa: BLE001 — acceleration is never required
            logger.warning("backbone_patch_failed", error=str(exc))
            return {"backend": "pytorch", "error": str(exc)}

    # ------------------------------------------------------------------ batch loop
    def _ensure_batch_loop(self) -> None:
        if self._batch_task is None or self._batch_task.done():
            self._queue = asyncio.Queue()
            self._batch_task = asyncio.get_event_loop().create_task(self._batch_loop())

    async def _batch_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            pending: list[_Req] = []
            try:
                pending.append(await asyncio.wait_for(self._queue.get(), timeout=5.0))
            except asyncio.TimeoutError:
                continue

            # Collect for up to batch_timeout, or until we have plenty to group.
            deadline = loop.time() + self._batch_timeout
            while len(pending) < self._max_batch * 4:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    pending.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break

            for group in self._group(pending):
                try:
                    await loop.run_in_executor(self._executor, self._generate_group, group)
                except Exception as exc:  # noqa: BLE001
                    self._fail(group, exc)

    def _group(self, reqs: list[_Req]) -> list[list[_Req]]:
        """Partition into batches that generate() can legally and efficiently run.

        Split by generation config and by has-prompt (both are whole-call
        properties of generate()), then within each partition sort by estimated
        length and cut a new batch whenever the item count, the total frame
        budget, or the length-bucket ratio would be exceeded.
        """
        partitions: dict[tuple[str, bool], list[_Req]] = {}
        for req in reqs:
            partitions.setdefault((req.params.batch_key, req.prompt is not None), []).append(req)

        batches: list[list[_Req]] = []
        for items in partitions.values():
            items.sort(key=lambda r: r.est_frames)
            current: list[_Req] = []
            frames = 0
            for req in items:
                too_many = len(current) >= self._max_batch
                too_long = current and (frames + req.est_frames) > self._max_batch_frames
                # Padding waste: generate() pads to the longest item, so a short
                # request batched behind a much longer one pays the long one's cost.
                too_ragged = current and req.est_frames > current[0].est_frames * self._bucket_ratio
                if current and (too_many or too_long or too_ragged):
                    batches.append(current)
                    current, frames = [], 0
                current.append(req)
                frames += req.est_frames
            if current:
                batches.append(current)
        return batches

    def _generate_group(self, group: list[_Req]) -> None:
        """Run one batched generate() on the GPU thread."""
        params = group[0].params
        gen_config = self._gen_config_cls(**params.config)

        texts = [r.text for r in group]
        prompts = [r.prompt for r in group] if group[0].prompt is not None else None
        languages = [r.language for r in group]
        instructs = [r.instruct for r in group]

        kwargs: dict[str, Any] = {
            "text": texts,
            "voice_clone_prompt": prompts,
            "generation_config": gen_config,
            "normalize_text": params.normalize_text,
        }
        # generate() treats a list of all-None as "one value per item" and is
        # happier with a plain None when nothing was set.
        if any(lang is not None for lang in languages):
            kwargs["language"] = languages
        if any(ins is not None for ins in instructs):
            kwargs["instruct"] = instructs
        if params.speed is not None:
            kwargs["speed"] = params.speed
        if params.duration is not None:
            kwargs["duration"] = params.duration

        try:
            started = time.perf_counter()
            audios = self.model.generate(**kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000

            self.stats["batches"] += 1
            self.stats["batched_items"] += len(group)
            self.stats["gpu_ms"] += elapsed_ms
            logger.info(
                "omnivoice_batch", n=len(group), ms=round(elapsed_ms, 1),
                per_item_ms=round(elapsed_ms / len(group), 1),
                frames=sum(r.est_frames for r in group),
                cloned=prompts is not None,
            )

            for req, wav in zip(group, audios):
                if not req.future.done():
                    req.future.get_loop().call_soon_threadsafe(
                        req.future.set_result,
                        np.asarray(wav, dtype=np.float32).reshape(-1),
                    )
        except Exception as exc:  # noqa: BLE001
            if "out of memory" in str(exc).lower():
                self._free_gpu_memory()
            self._fail(group, exc)

    @staticmethod
    def _free_gpu_memory() -> None:
        try:
            import gc

            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _fail(self, reqs: list[_Req], exc: Exception) -> None:
        self.stats["errors"] += len(reqs)
        logger.error("omnivoice_batch_failed", n=len(reqs), error=str(exc))
        for req in reqs:
            if not req.future.done():
                req.future.get_loop().call_soon_threadsafe(req.future.set_exception, exc)

    # ------------------------------------------------------------------ public API
    def resolve_language(self, voice_id: str | None, language: str | None) -> str | None:
        """Language precedence: explicit request > voice's preference > default."""
        from flowtts.text import omnivoice_lang

        if language:
            return omnivoice_lang(language)
        if self.registry is not None and self.registry.language(voice_id):
            return omnivoice_lang(self.registry.language(voice_id))
        return omnivoice_lang(settings.voices.default_language)

    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        language: str | None = None,
        instruct: str | None = None,
        prompt: Any = None,
        params: GenParams | None = None,
        _retrying: bool = False,
    ) -> np.ndarray:
        """Enqueue one synthesis request; return its float32 waveform.

        ``prompt`` is an explicit VoiceClonePrompt (from an inline clone) and
        takes precedence over ``voice_id``. Passing ``instruct`` with no prompt
        is OmniVoice's voice-design mode; passing neither is auto-voice.

        A generation that comes back empty or effectively silent is retried once
        with deterministic sampling (see :meth:`GenParams.as_retry`). At a low
        ``num_step`` the sampler occasionally wanders into silence on a short
        utterance, and returning that to a live call is worse than spending one
        more GPU pass on it.
        """
        if self.model is None:
            raise RuntimeError("OmniVoiceEngine not initialized")
        self._ensure_batch_loop()

        if prompt is None and not instruct and self.registry is not None:
            prompt = self.registry.prompt(voice_id)

        params = params or GenParams.build()
        # Frames are estimated from text length rather than by calling
        # OmniVoice's duration estimator: this runs on the event loop for every
        # request, and the estimate only has to be good enough to bucket by.
        est_frames = max(1, int(estimate_duration(text) * (self.frame_rate or 25.0)))

        self.stats["requests"] += 1
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put(_Req(
            text=text,
            prompt=prompt,
            language=self.resolve_language(voice_id, language),
            instruct=instruct,
            params=params,
            est_frames=est_frames,
            future=future,
            queued_at=time.perf_counter(),
        ))
        wav = await future

        if is_silent(wav):
            self.stats["degenerate"] += 1
            if _retrying:
                logger.error("degenerate_after_retry", text=text[:80], voice=voice_id)
                raise RuntimeError(
                    "synthesis produced no audible output for this text; "
                    "try a higher num_step or a different voice"
                )
            logger.warning("degenerate_generation_retrying", text=text[:80],
                           voice=voice_id, samples=int(wav.size),
                           num_step=params.config.get("num_step"))
            self.stats["retries"] += 1
            return await self.synthesize(
                text, voice_id=voice_id, language=language, instruct=instruct,
                prompt=prompt, params=params.as_retry(), _retrying=True,
            )
        return wav

    async def create_prompt(self, audio_path: str, ref_text: str) -> Any:
        """Encode a reference clip into a VoiceClonePrompt without persisting it.

        Used by the one-shot clone endpoints, where the caller wants to hear a
        voice before deciding to keep it.
        """
        if self.model is None:
            raise RuntimeError("OmniVoiceEngine not initialized")
        if not ref_text or not ref_text.strip():
            raise ValueError("ref_text is required for voice cloning")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._encode_prompt, str(audio_path), ref_text
        )

    def _encode_prompt(self, audio_path: str, ref_text: str) -> Any:
        """Codec-encode a reference clip (GPU thread)."""
        max_seconds = settings.voices.max_reference_seconds
        ref_audio: Any = audio_path
        if max_seconds and max_seconds > 0:
            trimmed = self._load_trimmed_reference(audio_path, max_seconds)
            if trimmed is not None:
                ref_audio = trimmed
        return self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            preprocess_prompt=settings.generation.preprocess_prompt,
        )

    def _load_trimmed_reference(self, audio_path: str, max_seconds: float):
        """Load a reference clip, capped at ``max_seconds``.

        Prompt tokens are prepended to every generated chunk, so an over-long
        reference is a per-request tax on latency for as long as that voice is
        in use — not a one-off cost at clone time.
        """
        try:
            import torch
            from omnivoice.utils.audio import load_audio

            wav = load_audio(audio_path, self.sampling_rate)
            wav = torch.as_tensor(wav)
            limit = int(max_seconds * self.sampling_rate)
            if wav.shape[-1] <= limit:
                return None
            logger.info("reference_trimmed", path=audio_path,
                        seconds=round(wav.shape[-1] / self.sampling_rate, 2),
                        capped_at=max_seconds)
            return (wav[..., :limit], self.sampling_rate)
        except Exception as exc:  # noqa: BLE001 — fall back to the raw file
            logger.warning("reference_trim_failed", path=audio_path, error=str(exc))
            return None

    async def create_voice(
        self,
        voice_id: str,
        audio_path: str,
        ref_text: str,
        language: str | None = None,
    ) -> dict:
        """Clone a voice, persist it as npz, and register it live (no restart)."""
        prompt = await self.create_prompt(audio_path, ref_text)
        tokens = prompt.ref_audio_tokens.detach().cpu().numpy()

        out = Path(settings.voices.voices_dir) / f"{voice_id}.npz"
        save_voice_npz(
            out,
            ref_audio_tokens=tokens,
            ref_text=str(prompt.ref_text),
            ref_rms=float(prompt.ref_rms),
            sample_rate=self.sampling_rate,
            frame_rate=self.frame_rate,
            alias=voice_id,
            language=language,
        )
        self.registry.add(voice_id, out)
        logger.info("voice_cloned", voice_id=voice_id, tokens=list(tokens.shape),
                    language=language)
        return {
            "voice_id": voice_id,
            "tokens": list(tokens.shape),
            "reference_frames": int(tokens.shape[-1]),
            "reference_seconds": round(tokens.shape[-1] / (self.frame_rate or 25.0), 2),
            "ref_rms": round(float(prompt.ref_rms), 4),
            "ref_text": str(prompt.ref_text),
            "language": language,
            "sample_rate": self.sampling_rate,
            "npz": str(out),
        }

    def delete_voice(self, voice_id: str) -> bool:
        """Remove a voice from the registry and from disk."""
        if self.registry is None or not self.registry.has(voice_id):
            return False
        path = Path(settings.voices.voices_dir) / f"{voice_id}.npz"
        path.unlink(missing_ok=True)
        self.registry.remove(voice_id)
        logger.info("voice_deleted", voice_id=voice_id)
        return True

    def snapshot(self) -> dict:
        """Engine counters for /v1/stats."""
        stats = dict(self.stats)
        batches = max(1, stats["batches"])
        return {
            **stats,
            "avg_batch_size": round(stats["batched_items"] / batches, 2),
            "avg_batch_ms": round(stats["gpu_ms"] / batches, 1),
            "queue_depth": self._queue.qsize() if self._queue else 0,
            "engine": self.engine_info,
        }

    async def _warmup(self) -> None:
        """Prime kernels, compilation and CUDA-graph capture before real traffic."""
        cfg = settings.omnivoice
        voice = (settings.voices.default_voice
                 if self.registry and self.registry.has(settings.voices.default_voice)
                 else None)
        n = max(1, min(cfg.warmup_batch, cfg.max_batch))
        logger.info("omnivoice_warmup", n=n, voice=voice)
        started = time.perf_counter()
        try:
            await asyncio.gather(*[
                self.synthesize(cfg.warmup_sentence, voice_id=voice) for _ in range(n)
            ])
            logger.info("omnivoice_warmup_done", ms=round((time.perf_counter() - started) * 1000))
        except Exception as exc:  # noqa: BLE001
            logger.warning("omnivoice_warmup_failed", error=str(exc))
