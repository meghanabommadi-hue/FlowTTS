"""Pipeline position: SCHEDULER — continuous-batching Euler ODE over the flow decoder.

Role in pipeline:
  Owns the GPU. One asyncio task drives every in-flight flow trajectory to
  completion, mixing spans from unrelated requests into the same forward pass.

      SpanRequest --submit()--> queue --admit--> arena slot
        --> [ N Euler steps, batched with everyone else ] -->
      generated mel --future--> vocoder

Why this can do in-flight batching at all
-----------------------------------------
DhVaani is a flow-matching model: every span needs exactly `num_step`
evaluations of the same network, with no KV cache and no token-by-token
dependency. Naively you would batch whole spans together and everyone waits for
the slowest -- classic static batching, with all its head-of-line blocking.

The escape hatch is that `TTSZipformer.forward` accepts a **per-sample timestep**
`t` of shape `(B,)`. Upstream never uses it that way (`solver.DiffusionModel`
asserts `t.dim() == 0`), but the network genuinely supports it. So a single
forward pass can contain a span on its 1st Euler step next to one on its 7th.

The consequence is exactly LLM-style in-flight batching:
  * a new span joins the running batch at the very next tick, not at the next
    batch boundary, so time-to-first-audio does not depend on what else is
    running;
  * a finished span leaves immediately and frees its slot;
  * the batch stays full, which is what keeps a 512-wide Zipformer near roofline.

Invariants this module maintains
--------------------------------
  * NO host synchronisation inside the tick. Every quantity the Python code
    needs (prompt_len, gen_len, num_step, t) is tracked host-side in `_Row`;
    nothing is ever read back off a device tensor.
  * Occupied arena rows stay packed at the front, so each step operates on a
    contiguous `arena.x[:n]` view rather than a gather.
  * Scalar per-row state (t, dt, gs) is mirrored on the host and uploaded with
    one pinned copy per bucket per tick.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Iterable

import structlog
import torch

from flowtts.dhvaani.config import N_MELS, dhv_settings
from flowtts.dhvaani.engine.arena import ArenaPool, Slot
from flowtts.dhvaani.engine.memory import VramWatchdog, is_cuda_oom
from flowtts.dhvaani.model import ops
from flowtts.dhvaani.model.text_encoder import TextEncoder
from flowtts.dhvaani.model.triton_kernels import fused_cfg_combine, fused_euler_update
from flowtts.dhvaani.types import (
    QueueFull,
    RequestCancelled,
    SpanRequest,
    SpanState,
)

logger = structlog.get_logger(__name__)


class _Row:
    """Host-side state for one in-flight trajectory. Mirrors an arena row."""

    __slots__ = (
        "span", "slot", "step", "num_step", "grid", "gs", "cfg_until_t",
        "prompt_len", "gen_len", "future", "admitted_at", "deadline",
    )

    def __init__(self, span: SpanRequest, slot: Slot, grid: list[float], future):
        self.span = span
        self.slot = slot
        self.step = 0
        self.num_step = span.params.num_step
        self.grid = grid
        self.gs = span.params.guidance_scale
        self.cfg_until_t = span.params.cfg_until_t
        self.prompt_len = span.voice.mel_frames
        self.gen_len = max(0, span.total_frames - span.voice.mel_frames)
        self.future = future
        self.admitted_at = time.perf_counter()
        self.deadline = self.admitted_at + dhv_settings.engine.request_timeout_s

    @property
    def t(self) -> float:
        return self.grid[self.step]

    @property
    def dt(self) -> float:
        return self.grid[self.step + 1] - self.grid[self.step]

    def uses_cfg(self) -> bool:
        # CFG doubles this row's cost, so we honour cfg_until_t per row: the
        # unconditional branch shapes the trajectory most at low t.
        return self.gs != 0.0 and self.t <= self.cfg_until_t

    def done(self) -> bool:
        return self.step >= self.num_step


class FlowScheduler:
    """Drives every in-flight span's Euler ODE, batched across requests."""

    def __init__(
        self,
        backend,
        text_encoder: TextEncoder,
        arenas: ArenaPool,
        watchdog: VramWatchdog,
        settings=None,
        fallback_backend=None,
    ):
        self._s = settings or dhv_settings
        self._backend = backend
        self._fallback = fallback_backend
        self._te = text_encoder
        self._arenas = arenas
        self._wd = watchdog
        self.device = arenas.device
        self.dtype = arenas.dtype

        self._queue: deque[tuple[SpanRequest, asyncio.Future]] = deque()
        self._rows: dict[int, list[_Row]] = defaultdict(list)  # bucket -> rows, index-aligned
        self._by_request: dict[str, list[_Row]] = defaultdict(list)
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # Timestep grids are identical for every span sharing (num_step, t_shift),
        # so build each once instead of per row per step.
        self._grids: dict[tuple[int, float], list[float]] = {}

        # Pinned host staging for the scalar per-row state.
        # Separate pinned staging buffers per quantity. Sharing one would be a
        # data race: `copy_(..., non_blocking=True)` from pinned memory returns
        # before the transfer completes, so reusing the buffer for `dt` could
        # overwrite `t` while it is still being read by the DMA engine.
        cap = max(self._s.engine.max_batch_size, 1)
        pin = self.device.type == "cuda"
        self._h_t = torch.zeros(cap, dtype=torch.float32, pin_memory=pin)
        self._h_gs = torch.zeros(cap, dtype=torch.float32, pin_memory=pin)
        self._h_dt = torch.zeros(cap, dtype=torch.float32, pin_memory=pin)

        # Private RNG so noise draws never touch the global CUDA generator.
        self._noise_gen = torch.Generator(device=self.device)
        self._noise_gen.manual_seed(1234)

        self._unsupported_logged: set[tuple[int, int]] = set()
        self.stats_admitted = 0
        self.stats_retired = 0
        self.stats_failed = 0
        self.stats_steps = 0
        self.stats_batch_rows = 0
        self._started_at = time.perf_counter()

    # -- lifecycle -----------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopping = False
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="dhvaani-flow-scheduler")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        self._drain()

    def _drain(self) -> None:
        """Fail every outstanding span so no caller is left awaiting forever.

        Without this a shutdown with work in flight strands each request's
        `await fut` permanently: the scheduler task that would have resolved it
        is gone. Callers see a clean error instead of a hang.
        """
        err = RequestCancelled("scheduler is shutting down")
        while self._queue:
            _span, fut = self._queue.popleft()
            self._resolve_exc(fut, err)
        for bucket in list(self._rows.keys()):
            for row in list(self._rows[bucket]):
                try:
                    self._retire_row(row, error=err)
                except Exception:
                    self._resolve_exc(row.future, err)
        self._rows.clear()
        self._by_request.clear()

    # -- submission ----------------------------------------------------------
    async def submit(self, span: SpanRequest) -> "asyncio.Future":
        if len(self._queue) >= self._s.engine.max_queue_depth:
            raise QueueFull(
                f"scheduler queue is full ({len(self._queue)} spans); "
                "the GPU cannot keep up with the arrival rate"
            )
        loop = self._loop or asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        span.state = SpanState.QUEUED
        self._queue.append((span, fut))
        return fut

    def cancel(self, request_id: str) -> int:
        n = 0
        remaining = deque()
        while self._queue:
            span, fut = self._queue.popleft()
            if span.request_id == request_id:
                span.state = SpanState.CANCELLED
                self._resolve_exc(fut, RequestCancelled("cancelled by client"))
                n += 1
            else:
                remaining.append((span, fut))
        self._queue = remaining

        for row in list(self._by_request.get(request_id, [])):
            self._retire_row(row, error=RequestCancelled("cancelled by client"))
            n += 1
        return n

    # -- grids ---------------------------------------------------------------
    def _grid(self, num_step: int, t_shift: float) -> list[float]:
        key = (num_step, round(t_shift, 6))
        g = self._grids.get(key)
        if g is None:
            g = ops.get_time_steps(num_step, t_shift, device="cpu").tolist()
            self._grids[key] = g
        return g

    # -- main loop -----------------------------------------------------------
    async def _run(self) -> None:
        idle = self._s.engine.idle_sleep_s
        while not self._stopping:
            try:
                did_work = self._tick()
            except Exception as e:  # pragma: no cover - never let the loop die
                logger.exception("scheduler_tick_failed", error=str(e))
                did_work = False
            if not did_work:
                await asyncio.sleep(idle)
            else:
                # Yield so the API coroutines can drain futures and enqueue more.
                await asyncio.sleep(0)

    def _tick(self) -> bool:
        worked = self._admit()
        for bucket in list(self._rows.keys()):
            if self._rows[bucket]:
                self._step_bucket(bucket)
                worked = True
        worked |= self._retire_expired()
        return worked

    # -- admission -----------------------------------------------------------
    def _admit(self) -> bool:
        if not self._queue:
            return False
        if not self._wd.admission_allowed():
            return False

        # Batch everything admissible this tick into ONE text-encoder call --
        # per-span encoding would be a launch-bound disaster at high RPS.
        take: list[tuple[SpanRequest, asyncio.Future, int, Slot]] = []
        deferred: deque = deque()

        while self._queue and len(take) < self._s.engine.max_batch_size:
            span, fut = self._queue.popleft()
            if fut.cancelled():
                continue
            try:
                total = self._te.estimate_total_frames(
                    span.voice.mel_frames,
                    len(span.voice.token_ids),
                    span.n_tokens,
                    span.params.speed,
                )
            except Exception as e:
                self._resolve_exc(fut, e)
                continue

            max_bucket = self._s.buckets.buckets[-1]
            if total > max_bucket:
                # The chunker is supposed to prevent this; failing loudly beats
                # silently truncating someone's audio.
                self._resolve_exc(
                    fut,
                    ValueError(
                        f"span needs {total} frames, exceeding the largest bucket "
                        f"({max_bucket}). Reduce chunk.steady_chunk_seconds or "
                        f"voice.max_prompt_seconds."
                    ),
                )
                self.stats_failed += 1
                continue

            bucket = self._s.bucket_for(total)
            slot = self._arenas.acquire(bucket)
            if slot is None:
                deferred.append((span, fut))
                continue
            span.total_frames = total
            span.gen_frames = total - span.voice.mel_frames
            span.bucket = bucket
            take.append((span, fut, total, slot))

        self._queue.extendleft(reversed(deferred))
        if not take:
            return False

        self.stats_admitted += self._write_slots(take)
        return True

    def _write_slots(self, take: list) -> int:
        """Encode the newly admitted group and write it into its arena slots.

        Returns the number of spans successfully admitted. A failure is scoped
        to the bucket it happened in: slots for that bucket are released and its
        futures failed, while buckets that already succeeded keep their rows.
        Releasing everything on any failure would strand rows that are already
        in `self._rows` pointing at freed slots.
        """
        # Group by bucket: conditions must be built at the bucket width.
        by_bucket: dict[int, list] = defaultdict(list)
        for item in take:
            by_bucket[item[3].bucket].append(item)

        admitted = 0
        for bucket, items in by_bucket.items():
            try:
                self._write_bucket(bucket, items)
                admitted += len(items)
            except Exception as e:
                logger.error(
                    "admission_failed", bucket=bucket, n=len(items), error=str(e)[:300]
                )
                # Release in reverse acquisition order so the arena's
                # swap-with-last stays consistent with an empty row list.
                for span, fut, _total, slot in reversed(items):
                    try:
                        self._arenas.release(slot)
                    except Exception:
                        pass
                    self._resolve_exc(fut, e)
                self.stats_failed += len(items)
                if is_cuda_oom(e):
                    self._wd.handle_oom(e)
        return admitted

    def _write_bucket(self, bucket: int, items: list) -> None:
        arena = self._arenas.arena(bucket)
        cat_ids = [it[0].voice.token_ids + it[0].token_ids for it in items]
        prompt_tok = [len(it[0].voice.token_ids) for it in items]
        prompt_mels = [it[0].voice.mel for it in items]
        prompt_frames = [it[0].voice.mel_frames for it in items]
        speeds = [it[0].params.speed for it in items]

        cond = self._te.build_conditions(
            cat_ids, prompt_tok, prompt_mels, prompt_frames, speeds, bucket
        )

        for j, (span, fut, _total, slot) in enumerate(items):
            i = slot.index
            arena.text_c[i].copy_(cond.text_condition[j])
            arena.speech_c[i].copy_(cond.speech_condition[j])
            arena.mask[i].copy_(cond.padding_mask[j])

            # x0 ~ N(0, 1) over the whole bucket width; padded positions are
            # zeroed on the first Euler update anyway, and generating the full
            # row keeps the RNG call shape-stable.
            # Always draw from a scheduler-owned generator, never the global one.
            # The global CUDA generator participates in CUDA-graph capture, and
            # once any graph has been captured, drawing from it outside a capture
            # can raise "Offset increment outside graph capture encountered
            # unexpectedly". A private generator is also what makes a seeded
            # request reproducible regardless of what else is running.
            gen = self._noise_gen
            if span.params.seed is not None:
                gen = torch.Generator(device=self.device)
                gen.manual_seed(span.params.seed + span.span_index)
            noise = torch.randn(
                (bucket, N_MELS), device=self.device, dtype=self.dtype, generator=gen
            )
            arena.x[i].copy_(noise)

            grid = self._grid(span.params.num_step, span.params.t_shift)
            row = _Row(span, slot, grid, fut)
            span.state = SpanState.FLOWING
            span.started_at = time.perf_counter()

            rows = self._rows[bucket]
            assert len(rows) == i, (len(rows), i)  # arena packing invariant
            rows.append(row)
            self._by_request[span.request_id].append(row)

    # -- stepping ------------------------------------------------------------
    def _partition_cfg(self, bucket: int) -> int:
        """Reorder the bucket so CFG rows come first. Returns the CFG count.

        Rows cross the `cfg_until_t` boundary at most once in their life, so the
        partition is almost always already correct and this is a no-op scan.
        """
        rows = self._rows[bucket]
        arena = self._arenas.arena(bucket)
        lo = 0
        hi = len(rows) - 1
        while lo <= hi:
            if rows[lo].uses_cfg():
                lo += 1
            elif not rows[hi].uses_cfg():
                hi -= 1
            else:
                arena.swap(lo, hi)
                rows[lo], rows[hi] = rows[hi], rows[lo]
                rows[lo].slot = Slot(bucket, lo)
                rows[hi].slot = Slot(bucket, hi)
                lo += 1
                hi -= 1
        return lo

    def _upload_scalars(self, rows: list[_Row], arena, n: int) -> None:
        for i in range(n):
            self._h_t[i] = rows[i].t
            self._h_gs[i] = rows[i].gs
        arena.t[:n].copy_(self._h_t[:n], non_blocking=True)
        arena.gs[:n].copy_(self._h_gs[:n], non_blocking=True)

    def _run_backend(self, x, text_c, speech_c, t, mask):
        B, T = x.shape[0], x.shape[1]
        be = self._backend
        if not be.supports_bucket(B, T):
            if (B, T) not in self._unsupported_logged:
                self._unsupported_logged.add((B, T))
                logger.info(
                    "backend_shape_unsupported_fallback",
                    backend=getattr(be, "name", "?"), batch=B, frames=T,
                )
            be = self._fallback or self._backend
        return be.fm_step(x, text_c, speech_c, t, mask)

    def _step_bucket(self, bucket: int) -> None:
        rows = self._rows[bucket]
        if not rows:
            return
        arena = self._arenas.arena(bucket)
        n_cfg = self._partition_cfg(bucket)
        n = len(rows)
        self._upload_scalars(rows, arena, n)

        try:
            if n_cfg:
                self._step_slice(arena, rows, 0, n_cfg, cfg=True)
            if n - n_cfg:
                self._step_slice(arena, rows, n_cfg, n, cfg=False)
        except Exception as e:
            kind = self._wd.handle_oom(e) if is_cuda_oom(e) else None
            logger.error(
                "flow_step_failed", bucket=bucket, rows=n, oom=kind, error=str(e)[:300]
            )
            # Fail only this bucket's rows; other buckets are unaffected.
            for row in list(rows):
                self._retire_row(row, error=e)
            return

        self.stats_steps += 1
        self.stats_batch_rows += n

        for row in list(rows):
            row.step += 1
            if row.done():
                self._retire_row(row)

    def _step_slice(self, arena, rows, lo: int, hi: int, cfg: bool) -> None:
        x = arena.x[lo:hi]
        text_c = arena.text_c[lo:hi]
        speech_c = arena.speech_c[lo:hi]
        mask = arena.mask[lo:hi]
        t = arena.t[lo:hi]
        gs = arena.gs[lo:hi]

        if cfg:
            x2, text2, speech2, mask2, t2, gs_eff = ops.cfg_expand(
                x, text_c, speech_c, mask, t, gs
            )
            v2 = self._run_backend(x2, text2, speech2, t2, mask2)
            v = fused_cfg_combine(v2.contiguous(), gs_eff)
        else:
            v = self._run_backend(x, text_c, speech_c, t, mask)

        # dt is per-row and depends on each row's own grid position.
        #
        # Indexed by ABSOLUTE row position, not from zero: `_step_bucket` calls
        # this twice per tick (the CFG partition, then the non-CFG one), and a
        # non_blocking copy from pinned memory returns before the DMA completes.
        # Writing both partitions at offset 0 would let the second overwrite the
        # first while it was still in flight.
        for i in range(lo, hi):
            self._h_dt[i] = rows[i].dt
        dt = self._h_dt[lo:hi].to(self.device, non_blocking=True)
        fused_euler_update(x, v, dt, mask)

    # -- retirement ----------------------------------------------------------
    def _retire_expired(self) -> bool:
        now = time.perf_counter()
        worked = False
        for bucket in list(self._rows.keys()):
            for row in list(self._rows[bucket]):
                if now > row.deadline:
                    worked = True
                    self._retire_row(
                        row,
                        error=TimeoutError(
                            f"span exceeded engine.request_timeout_s "
                            f"({self._s.engine.request_timeout_s}s)"
                        ),
                    )
        return worked

    def _retire_row(self, row: _Row, error: BaseException | None = None) -> None:
        bucket = row.slot.bucket
        rows = self._rows[bucket]
        idx = row.slot.index
        if idx >= len(rows) or rows[idx] is not row:
            try:
                idx = rows.index(row)
            except ValueError:
                return

        arena = self._arenas.arena(bucket)
        result = None
        if error is None:
            try:
                # Slice out the generated region and copy it: the slot is about
                # to be handed to another span, so a view would alias.
                start = row.prompt_len
                stop = start + row.gen_len
                mel = arena.x[idx, start:stop].to(torch.float32).div_(
                    self._s.flow.feat_scale
                ).clone()
                result = mel
            except Exception as e:
                error = e

        # `arena.release` keeps occupancy packed by copying the LAST occupied
        # row into the hole. The python row list must mirror that exactly --
        # a list.pop(idx) would shift every later element left by one while the
        # arena moved only the last row, silently pairing every row after `idx`
        # with the wrong slot's tensors.
        arena.release(idx)
        last = len(rows) - 1
        if idx != last:
            rows[idx] = rows[last]
            rows[idx].slot = Slot(bucket, idx)
        rows.pop()

        siblings = self._by_request.get(row.span.request_id)
        if siblings is not None:
            try:
                siblings.remove(row)
            except ValueError:
                pass
            if not siblings:
                self._by_request.pop(row.span.request_id, None)

        row.span.finished_at = time.perf_counter()
        if error is None:
            row.span.state = SpanState.VOCODING
            self.stats_retired += 1
            self._resolve(row.future, result)
        else:
            row.span.state = SpanState.FAILED
            row.span.error = str(error)
            self.stats_failed += 1
            self._resolve_exc(row.future, error)

    @staticmethod
    def _resolve(fut, value) -> None:
        # The scheduler task and the awaiting coroutines share one event loop,
        # so a direct set_result is correct here; no thread-safe hop needed.
        if not fut.done():
            fut.set_result(value)

    @staticmethod
    def _resolve_exc(fut, exc: BaseException) -> None:
        if not fut.done():
            fut.set_exception(exc)

    # -- observability -------------------------------------------------------
    def stats(self) -> dict:
        elapsed = max(time.perf_counter() - self._started_at, 1e-9)
        active = {b: len(r) for b, r in self._rows.items() if r}
        return {
            "queue_depth": len(self._queue),
            "active_rows": sum(active.values()),
            "active_by_bucket": active,
            "admitted": self.stats_admitted,
            "retired": self.stats_retired,
            "failed": self.stats_failed,
            "steps": self.stats_steps,
            "steps_per_s": round(self.stats_steps / elapsed, 1),
            "mean_batch": round(
                self.stats_batch_rows / self.stats_steps, 2
            ) if self.stats_steps else 0.0,
            "arenas": self._arenas.stats(),
        }
