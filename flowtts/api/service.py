"""Pipeline position: API SERVICE STATE — the process-wide singleton.

Role in pipeline:
  One model load, one batch queue, one voice registry, shared by every transport
  in the process (REST, OpenAI-compatible, WebSocket, control API).

      http_app / ws → service.synthesizer → OmniVoiceEngine → GPU

Also owns the three things that sit between a request and the GPU:

  **The WAV cache.** Call-centre prompts repeat heavily. A hit returns audio in
  microseconds and never touches the GPU, which is by a wide margin the largest
  real-world latency win available here.

  **The admission limiter.** Past a point, admitting more concurrent work makes
  every in-flight request slower without finishing any sooner. Requests beyond
  the limit wait in the semaphore instead of piling onto the batch queue.

  **OOM recovery.** A CUDA OOM frees caches and rejects new work briefly; a
  second OOM inside that window exits non-zero so the supervisor restarts a
  clean process.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from flowtts.core.config import resolve_path, settings
from flowtts.synthesis.models import OmniVoiceSynthesizer

logger = structlog.get_logger(__name__)

_OOM_RECOVERY_WINDOW_S = 5.0


class TTSService:
    """Everything the transports share. Created once per process."""

    def __init__(self) -> None:
        self.synthesizer: Optional[OmniVoiceSynthesizer] = None
        self.started_at = time.time()
        self.ready = False
        self.restarting = False
        self.oom_recovery = False

        self._init_lock = asyncio.Lock()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._cache_dir: Optional[Path] = None

        self.counters = {
            "requests": 0, "streamed": 0, "errors": 0,
            "cache_hits": 0, "cache_misses": 0, "rejected": 0,
        }
        self._ttfb_samples: list[float] = []
        self._rtf_samples: list[float] = []

    # ------------------------------------------------------------------ lifecycle
    async def initialize(self) -> None:
        """Load the model once; concurrent callers wait on the same load."""
        async with self._init_lock:
            if self.ready:
                return
            self._semaphore = asyncio.Semaphore(settings.omnivoice.max_active_requests)
            if settings.wav_cache_enabled:
                cache_dir = resolve_path(settings.wav_cache_dir)
                if cache_dir and cache_dir.is_dir():
                    self._cache_dir = cache_dir
                    logger.info("wav_cache_enabled", dir=str(cache_dir))

            synthesizer = OmniVoiceSynthesizer()
            await synthesizer.initialize()
            self.synthesizer = synthesizer
            self.ready = True
            logger.info("service_ready", **{
                k: v for k, v in synthesizer.engine_info.items()
                if k in ("sampling_rate", "frame_rate", "backbone")
            })

    @property
    def sample_rate(self) -> int:
        return self.synthesizer.sampling_rate if self.synthesizer else 24000

    def require_ready(self) -> OmniVoiceSynthesizer:
        if not self.ready or self.synthesizer is None:
            raise RuntimeError("model is still loading")
        if self.restarting:
            raise RuntimeError("server is restarting")
        if self.oom_recovery:
            raise RuntimeError("GPU memory recovery in progress")
        return self.synthesizer

    # ------------------------------------------------------------------ admission
    class _Slot:
        """Async context manager that holds one admission slot."""

        def __init__(self, service: "TTSService") -> None:
            self._service = service

        async def __aenter__(self) -> "TTSService._Slot":
            sem = self._service._semaphore
            if sem is not None:
                await sem.acquire()
            return self

        async def __aexit__(self, *exc_info) -> None:
            sem = self._service._semaphore
            if sem is not None:
                sem.release()

    def slot(self) -> "TTSService._Slot":
        return TTSService._Slot(self)

    @property
    def active_requests(self) -> int:
        if self._semaphore is None:
            return 0
        return settings.omnivoice.max_active_requests - self._semaphore._value

    # ------------------------------------------------------------------ cache
    @staticmethod
    def cache_key(text: str, voice_id: str | None, language: str | None,
                  overrides: dict | None) -> str:
        """SHA-256 over everything that changes the audio.

        Generation overrides are part of the key: two requests for the same text
        at num_step 4 and num_step 32 are different audio, and serving one for
        the other is exactly the silent quality regression this cache must not
        introduce.
        """
        payload = json.dumps(
            {"t": text, "v": voice_id, "l": language, "g": overrides or {}},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def cache_lookup(self, key: str) -> bytes | None:
        """Return cached WAV bytes for *key*, or None."""
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.wav"
        if path.exists():
            self.counters["cache_hits"] += 1
            return path.read_bytes()
        self.counters["cache_misses"] += 1
        return None

    def cache_lookup_legacy(self, text: str) -> bytes | None:
        """Look up the pre-existing sha256(text) cache layout.

        The caches already on the boxes are keyed on the raw transcript alone.
        Honouring that layout keeps every prompt generated before this version
        useful, and only for default settings — cache_key() covers everything else.
        """
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{hashlib.sha256(text.encode()).hexdigest()}.wav"
        if path.exists():
            self.counters["cache_hits"] += 1
            return path.read_bytes()
        return None

    def cache_store(self, key: str, wav_bytes: bytes) -> None:
        if self._cache_dir is None:
            return
        try:
            (self._cache_dir / f"{key}.wav").write_bytes(wav_bytes)
        except OSError as exc:
            logger.warning("wav_cache_write_failed", error=str(exc))

    # ------------------------------------------------------------------ metrics
    def record_ttfb(self, ms: float) -> None:
        self._ttfb_samples.append(ms)
        if len(self._ttfb_samples) > 5000:
            del self._ttfb_samples[:2500]

    def record_rtf(self, rtf: float) -> None:
        self._rtf_samples.append(rtf)
        if len(self._rtf_samples) > 5000:
            del self._rtf_samples[:2500]

    @staticmethod
    def _percentiles(samples: list[float]) -> dict:
        if not samples:
            return {}
        ordered = sorted(samples)
        def pick(p: float) -> float:
            return ordered[min(len(ordered) - 1, int(len(ordered) * p))]
        return {
            "count": len(ordered),
            "p50": round(pick(0.50), 1),
            "p90": round(pick(0.90), 1),
            "p99": round(pick(0.99), 1),
            "max": round(ordered[-1], 1),
        }

    def stats(self) -> dict:
        engine = self.synthesizer.engine.snapshot() if self.synthesizer else {}
        return {
            "ready": self.ready,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "active_requests": self.active_requests,
            "max_active_requests": settings.omnivoice.max_active_requests,
            "counters": dict(self.counters),
            "ttfb_ms": self._percentiles(self._ttfb_samples),
            "real_time_factor": self._percentiles(self._rtf_samples),
            "oom_recovery": self.oom_recovery,
            "restarting": self.restarting,
            **engine,
        }

    # ------------------------------------------------------------------ OOM
    @staticmethod
    def is_oom(exc: BaseException) -> bool:
        message = str(exc).lower()
        return "out of memory" in message

    async def handle_oom(self) -> None:
        """Free caches and pause admission; exit if it happens again immediately."""
        if self.restarting:
            return
        if not self.oom_recovery:
            logger.error("cuda_oom_recovering")
            self.oom_recovery = True
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

            async def _clear() -> None:
                await asyncio.sleep(_OOM_RECOVERY_WINDOW_S)
                if not self.restarting:
                    self.oom_recovery = False
                    logger.info("cuda_oom_window_cleared")

            asyncio.create_task(_clear())
            return

        logger.error("cuda_oom_during_recovery_restarting")
        self.restarting = True

        async def _exit() -> None:
            await asyncio.sleep(1.0)      # let in-flight sends flush
            sys.exit(1)                    # the supervisor starts a clean process

        asyncio.create_task(_exit())


service = TTSService()
