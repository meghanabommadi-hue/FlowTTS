"""Pipeline position: MEMORY ARENAS — pre-allocated batch slots, one set per bucket.

Role in pipeline:
  Every in-flight flow trajectory lives in a slot of a pre-allocated arena. The
  scheduler writes conditions into a slot on admission, mutates `x` in place for
  `num_step` iterations, then reads the generated region out and frees the slot.

Why arenas at all
-----------------
The naive implementation allocates `x`, `text_condition` and `speech_condition`
per request. At 200 RPS with several spans each, that is thousands of
allocations per second of sizes that vary with text length. PyTorch's caching
allocator handles it, but the block-size distribution fragments the pool: the
reserved-minus-allocated gap grows monotonically and the process looks like it
is leaking VRAM. This is the single most common way a TTS server dies after
several hours.

Allocating once, at startup, in a fixed set of bucket widths removes the problem
entirely. Steady-state VRAM is flat by construction.

(Classifier-free guidance still materialises a doubled `[uncond; cond]` batch on
each step, but only ever at one of the fixed bucket shapes, so those blocks come
straight back out of the caching allocator instead of growing it.)

Packing invariant
-----------------
Occupied slots are kept contiguous at the front of each arena, so the scheduler
can take `arena.x[:n]` -- a free, zero-copy view -- rather than an index_select,
which would copy the entire batch on every ODE step. Releasing a slot swaps the
last occupied row into the hole. That costs one row copy (~300 KB at T=512) per
retirement, paid once, versus a full-batch gather paid `num_step` times. The
scheduler is told which row moved so it can fix up its slot -> span map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import structlog

from flowtts.dhvaani.config import N_MELS, dhv_settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Slot:
    bucket: int
    index: int


class BucketArena:
    """Pre-allocated tensors for one bucket width."""

    def __init__(self, frames: int, max_batch: int, device, dtype, n_mels: int = N_MELS):
        import torch

        self.frames = int(frames)
        self.max_batch = int(max_batch)
        self.device = device
        self.dtype = dtype
        self._n = 0  # number of occupied rows, always packed at [0, _n)

        z = lambda *shape, dt=dtype: torch.zeros(*shape, device=device, dtype=dt)  # noqa: E731
        self.x = z(max_batch, frames, n_mels)
        self.text_c = z(max_batch, frames, n_mels)
        self.speech_c = z(max_batch, frames, n_mels)
        self.mask = z(max_batch, frames, dt=torch.bool)
        self.t = z(max_batch, dt=torch.float32)
        self.dt = z(max_batch, dt=torch.float32)
        self.gs = z(max_batch, dt=torch.float32)

    # -- slot management -----------------------------------------------------
    def acquire(self) -> int | None:
        if self._n >= self.max_batch:
            return None
        idx = self._n
        self._n += 1
        return idx

    def release(self, index: int) -> int | None:
        """Free `index`, keeping occupancy packed.

        Returns the row index that was MOVED into `index` (i.e. the old last
        occupied row), or None when `index` was already last. The caller must
        remap its bookkeeping for the returned row.
        """
        assert 0 <= index < self._n, (index, self._n)
        last = self._n - 1
        if index != last:
            self.x[index].copy_(self.x[last])
            self.text_c[index].copy_(self.text_c[last])
            self.speech_c[index].copy_(self.speech_c[last])
            self.mask[index].copy_(self.mask[last])
            moved = last
        else:
            moved = None
        self._n -= 1
        return moved

    def swap(self, a: int, b: int) -> None:
        """Swap two occupied rows (used to partition a bucket by CFG eligibility).

        Only the large per-frame buffers live here. The scalar per-row state
        (t, dt, gs) is mirrored on the host by the scheduler and uploaded in one
        copy per tick -- reading it back off the device to swap would cost a
        synchronisation, which is exactly what the tick must never do.
        """
        if a == b:
            return
        for buf in (self.x, self.text_c, self.speech_c, self.mask):
            tmp = buf[a].clone()
            buf[a].copy_(buf[b])
            buf[b].copy_(tmp)

    @property
    def n_occupied(self) -> int:
        return self._n

    def occupied(self) -> list[int]:
        return list(range(self._n))

    def n_free(self) -> int:
        return self.max_batch - self._n

    def nbytes(self) -> int:
        return (
            self.x.numel() * self.x.element_size() * 3
            + self.mask.numel()
            + (self.t.numel() + self.dt.numel() + self.gs.numel()) * 4
        )


class ArenaPool:
    """All bucket arenas, sized once against a VRAM budget."""

    def __init__(self, settings=None, device=None, dtype=None):
        import torch

        self._s = settings or dhv_settings
        self.device = device or torch.device(
            self._s.model.device if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self._arenas: dict[int, BucketArena] = {}
        self._plan: dict[int, int] = self._make_plan()
        if self._s.memory.preallocate_arenas:
            self.preallocate()

    # -- sizing --------------------------------------------------------------
    def _budget_vram(self) -> int:
        """Memory the arenas may size themselves against.

        Deliberately FREE memory, not total. On a shared GPU -- another training
        job, another inference server -- sizing against total would happily plan
        arenas for memory somebody else already owns, and the first allocation
        past the real limit takes down both processes. Reading free memory means
        a co-tenant's footprint is respected automatically.

        `mem_get_info` reports what the driver says is free right now, which
        already excludes our own weights if they are loaded (they are: the
        engine loads the model before building the pool).
        """
        import torch

        if self.device.type != "cuda":
            return 8 * 2**30  # a nominal budget so CPU tests still exercise the path
        free, total = torch.cuda.mem_get_info(self.device)
        return int(free)

    def _bytes_per_slot(self, frames: int) -> int:
        import torch

        el = torch.tensor([], dtype=self.dtype).element_size()
        return frames * N_MELS * el * 3 + frames + 12

    def _make_plan(self) -> dict[int, int]:
        """Choose per-bucket slot counts that fit the arena VRAM budget.

        Larger buckets cost proportionally more per slot, so a flat slot count
        would let the biggest bucket dominate. We weight the allocation by
        1/frames so every bucket gets a similar *share of bytes*, then clamp to
        [floor, engine.max_batch_size].
        """
        buckets = self._s.buckets.buckets
        budget = int(self._budget_vram() * self._s.memory.arena_vram_fraction)
        cap = self._s.engine.max_batch_size
        floor = 4

        weights = {b: 1.0 / b for b in buckets}
        wsum = sum(weights.values())

        # The arenas are only part of the story: a forward pass allocates
        # activations too, and the Zipformer's attention scores are
        # O(batch * frames^2). Sizing slots purely on arena bytes therefore
        # plans batches whose ACTIVATIONS will not fit -- 64 slots at 1536
        # frames looks like 60 MB of arena and needs many GB to actually run.
        # Cap batch * frames so peak activation memory stays roughly flat
        # across buckets.
        max_frame_rows = self._s.engine.max_batch_size * 512

        plan: dict[int, int] = {}
        for b in buckets:
            share = budget * (weights[b] / wsum)
            n = int(share // max(self._bytes_per_slot(b), 1))
            n = min(n, max(1, max_frame_rows // b))
            plan[b] = max(floor, min(cap, n))

        # If the floor pushed us over budget, shrink the largest buckets first --
        # they are the rarest in practice (most spans are 1-5 s).
        def total() -> int:
            return sum(plan[b] * self._bytes_per_slot(b) for b in buckets)

        for b in sorted(buckets, reverse=True):
            while total() > budget and plan[b] > floor:
                plan[b] -= 1

        logger.info(
            "arena_plan",
            budget_mib=round(budget / 2**20),
            planned_mib=round(total() / 2**20),
            buckets={b: plan[b] for b in buckets},
        )
        return plan

    # -- access --------------------------------------------------------------
    def preallocate(self) -> None:
        for b in self._s.buckets.buckets:
            self.arena(b)
        logger.info("arenas_preallocated", total_mib=round(self.total_bytes() / 2**20, 1))

    def arena(self, bucket: int) -> BucketArena:
        a = self._arenas.get(bucket)
        if a is None:
            a = BucketArena(bucket, self._plan[bucket], self.device, self.dtype)
            self._arenas[bucket] = a
        return a

    def buckets(self) -> tuple[int, ...]:
        return self._s.buckets.buckets

    def acquire(self, bucket: int) -> Slot | None:
        idx = self.arena(bucket).acquire()
        return None if idx is None else Slot(bucket, idx)

    def release(self, slot: Slot) -> int | None:
        return self.arena(slot.bucket).release(slot.index)

    def total_bytes(self) -> int:
        return sum(a.nbytes() for a in self._arenas.values())

    def total_active(self) -> int:
        return sum(a.n_occupied for a in self._arenas.values())

    def capacity(self, bucket: int) -> int:
        return self._plan.get(bucket, 0)

    def stats(self) -> dict:
        return {
            "total_mib": round(self.total_bytes() / 2**20, 1),
            "active": self.total_active(),
            "buckets": {
                b: {"occupied": a.n_occupied, "capacity": a.max_batch}
                for b, a in sorted(self._arenas.items())
            },
        }
