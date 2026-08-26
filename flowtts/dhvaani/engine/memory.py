"""Pipeline position: VRAM GOVERNANCE — watchdog, admission control, OOM recovery.

Role in pipeline:
  A background task plus a few synchronous hooks the scheduler and gateway call.
  Nothing here runs per ODE step.

What actually causes VRAM to grow in a TTS server
-------------------------------------------------
Not a leak, usually -- fragmentation. Variable-length outputs (here: the
vocoder's waveforms and any out-of-bucket tensor) make the caching allocator
hold blocks it cannot reuse for the next differently-sized request, so
`reserved` drifts up while `allocated` stays flat. Three defences, in order of
importance:

  1. `expandable_segments:True` in PYTORCH_CUDA_ALLOC_CONF (set in config.py
     before the first CUDA call). This is the big one -- it lets the allocator
     grow a segment instead of stranding blocks.
  2. Pre-allocated bucket arenas (engine/arena.py) so the flow path allocates
     nothing at all.
  3. This watchdog, which collects only when the reserved-vs-allocated slack
     actually exceeds a threshold.

`torch.cuda.empty_cache()` synchronises the device and drops the allocator's
cache, which makes the *next* allocations slower. Calling it per request would
be a serious throughput regression. It is rate-limited here and must stay that
way.
"""

from __future__ import annotations

import asyncio
import gc
import time

import structlog
from prometheus_client import Counter, Gauge

from flowtts.dhvaani.config import dhv_settings

logger = structlog.get_logger(__name__)

VRAM_ALLOCATED = Gauge("dhvaani_vram_allocated_bytes", "CUDA memory allocated by torch")
VRAM_RESERVED = Gauge("dhvaani_vram_reserved_bytes", "CUDA memory reserved by torch")
VRAM_TOTAL = Gauge("dhvaani_vram_total_bytes", "Total device memory")
VRAM_FREE = Gauge("dhvaani_vram_free_bytes", "Free device memory reported by the driver")
GC_COLLECTIONS = Counter("dhvaani_gc_collections_total", "Watchdog cache collections")
OOM_EVENTS = Counter("dhvaani_oom_events_total", "CUDA OOM events observed")
ADMISSION_BLOCKED = Counter(
    "dhvaani_admission_blocked_total", "Admissions refused by the VRAM ceiling"
)


def is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and ("cuda" in msg or "gpu" in msg)


class VramWatchdog:
    """Polls VRAM, collects when fragmented, and gates admission."""

    def __init__(self, settings=None):
        self._s = settings or dhv_settings
        self._last_gc = 0.0
        self._requests_since_gc = 0
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._collections = 0

        # OOM recovery state, mirroring flowtts/server.py's two-stage approach:
        # first OOM -> free caches and block admission briefly; a second OOM
        # inside that window means the soft path is not working and the process
        # should be recycled by its supervisor.
        self.recovery_active = False
        self.restart_requested = False
        self._recovery_until = 0.0
        self._recovery_window_s = 5.0

        import torch

        self._cuda = torch.cuda.is_available()
        self._device = torch.device(self._s.model.device) if self._cuda else None
        if self._cuda:
            VRAM_TOTAL.set(torch.cuda.get_device_properties(self._device).total_memory)

    # -- observation ---------------------------------------------------------
    def snapshot(self) -> dict:
        if not self._cuda:
            return {"cuda": False}
        import torch

        alloc = torch.cuda.memory_allocated(self._device)
        resv = torch.cuda.memory_reserved(self._device)
        free, total = torch.cuda.mem_get_info(self._device)
        VRAM_ALLOCATED.set(alloc)
        VRAM_RESERVED.set(resv)
        VRAM_FREE.set(free)
        VRAM_TOTAL.set(total)
        return {
            "cuda": True,
            "allocated": alloc,
            "reserved": resv,
            "free": free,
            "total": total,
            "allocated_frac": round(alloc / total, 4) if total else 0.0,
            "slack_frac": round((resv - alloc) / total, 4) if total else 0.0,
            "collections": self._collections,
            "recovery_active": self.recovery_active,
        }

    # -- collection ----------------------------------------------------------
    def maybe_collect(self, force: bool = False) -> bool:
        if not self._cuda:
            return False
        import torch

        now = time.monotonic()
        if not force and (now - self._last_gc) < self._s.memory.gc_min_interval_s:
            return False

        alloc = torch.cuda.memory_allocated(self._device)
        resv = torch.cuda.memory_reserved(self._device)
        total = torch.cuda.get_device_properties(self._device).total_memory
        slack = (resv - alloc) / total if total else 0.0

        if not force and slack < self._s.memory.gc_reserved_slack_fraction:
            return False

        gc.collect()
        torch.cuda.empty_cache()  # synchronises the device -- hence the rate limit
        self._last_gc = time.monotonic()
        self._requests_since_gc = 0
        self._collections += 1
        GC_COLLECTIONS.inc()
        logger.info(
            "vram_collected",
            slack_frac=round(slack, 4),
            forced=force,
            freed_mib=round((resv - torch.cuda.memory_reserved(self._device)) / 2**20, 1),
        )
        return True

    def note_request_done(self, n: int = 1) -> None:
        self._requests_since_gc += n
        if self._requests_since_gc >= self._s.memory.gc_every_n_requests:
            self.maybe_collect(force=True)

    # -- admission -----------------------------------------------------------
    def admission_allowed(self) -> bool:
        if self.restart_requested:
            return False
        if self.recovery_active:
            if time.monotonic() < self._recovery_until:
                ADMISSION_BLOCKED.inc()
                return False
            # Window elapsed: soft recovery worked, resume admitting.
            self.recovery_active = False
        if not self._cuda:
            return True
        import torch

        # Gate on DEVICE-WIDE occupancy, not our own allocation over total. On a
        # shared GPU those differ enormously: with a co-tenant holding 15 GB of a
        # 46 GB card, our own allocation could sit at 40% while the device is at
        # 95% and the next allocation OOMs. `mem_get_info` sees everyone.
        free, total = torch.cuda.mem_get_info(self._device)
        if not total:
            return True
        used_frac = 1.0 - (free / total)
        if used_frac >= self._s.memory.admission_vram_ceiling:
            ADMISSION_BLOCKED.inc()
            return False
        return True

    # -- OOM recovery --------------------------------------------------------
    def handle_oom(self, exc: BaseException) -> str:
        """Returns "soft" (recovered, retry later) or "restart" (recycle me).

        Deliberately does not exit the process -- library code must not. The
        server entrypoint owns that decision.
        """
        OOM_EVENTS.inc()
        now = time.monotonic()
        if self.recovery_active and now < self._recovery_until:
            self.restart_requested = True
            logger.error("vram_oom_during_recovery", error=str(exc)[:300])
            return "restart"

        self.recovery_active = True
        self._recovery_until = now + self._recovery_window_s
        self.maybe_collect(force=True)
        logger.warning(
            "vram_oom_soft_recovery",
            window_s=self._recovery_window_s,
            error=str(exc)[:300],
        )
        return "soft"

    # -- background loop -----------------------------------------------------
    async def run(self) -> None:
        self._stopping = False
        while not self._stopping:
            try:
                self.snapshot()
                self.maybe_collect()
            except Exception as e:  # pragma: no cover
                logger.warning("vram_watchdog_error", error=str(e))
            await asyncio.sleep(self._s.memory.vram_poll_interval_s)

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="dhvaani-vram-watchdog")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
