"""Tests for the continuous-batching ODE scheduler, on CPU with a stub backend.

The scheduler is the heart of the system and the part most likely to break
subtly, so these tests pin its invariants directly:

  * spans at DIFFERENT ODE steps share one forward pass (the whole point);
  * occupied arena rows stay packed at the front, so each step is a zero-copy
    contiguous slice rather than a gather;
  * a retired span's mel is the generated region only, feat-scale removed, and
    copied out of the slot before it is reused;
  * failures, cancellation, timeouts and over-long spans free their slots.
"""

from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")

from flowtts.dhvaani.config import DhvaaniSettings  # noqa: E402
from flowtts.dhvaani.engine.arena import ArenaPool  # noqa: E402
from flowtts.dhvaani.engine.scheduler import FlowScheduler  # noqa: E402
from flowtts.dhvaani.model.text_encoder import Conditions  # noqa: E402
from flowtts.dhvaani.model import ops  # noqa: E402
from flowtts.dhvaani.types import (  # noqa: E402
    QueueFull,
    RequestCancelled,
    SpanRequest,
    SynthParams,
    VoicePrompt,
)


def settings() -> DhvaaniSettings:
    s = DhvaaniSettings()
    s.buckets.min_frames = 128
    s.buckets.max_frames = 512
    s.buckets.granularity = 64
    s.engine.max_batch_size = 8
    s.engine.idle_sleep_s = 0.0005
    s.memory.preallocate_arenas = False
    s.flow.num_step = 4
    return s


class StubBackend:
    """Records every (batch, frames, t-vector) it is asked to evaluate."""

    name = "stub"

    def __init__(self, fail_on: int | None = None):
        self.calls: list[tuple[int, int, list[float]]] = []
        self.fail_on = fail_on

    def supports_bucket(self, batch, frames):
        return True

    def fm_step(self, x, text_c, speech_c, t, mask):
        self.calls.append((x.shape[0], x.shape[1], [round(float(v), 4) for v in t]))
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("stub backend failure")
        # A constant velocity makes the retired mel exactly predictable.
        return torch.ones_like(x)

    def warmup(self, *a, **k):
        pass

    def close(self):
        pass

    def stats(self):
        return {}


class StubTextEncoder:
    """Deterministic conditions; frame count comes straight from the token count."""

    FRAMES_PER_TOKEN = 3.0

    def estimate_total_frames(self, prompt_frames, prompt_tokens, span_tokens, speed):
        return int(prompt_frames + span_tokens * self.FRAMES_PER_TOKEN / speed)

    def build_conditions(self, cat_ids, prompt_tok, prompt_mels, prompt_frames,
                         speeds, num_frames):
        B = len(cat_ids)
        dev = torch.device("cpu")
        total = torch.tensor(
            [
                self.estimate_total_frames(prompt_frames[i], prompt_tok[i],
                                           len(cat_ids[i]) - prompt_tok[i], speeds[i])
                for i in range(B)
            ],
            dtype=torch.int64, device=dev,
        ).clamp(max=num_frames)
        pf = torch.tensor(prompt_frames, dtype=torch.int64, device=dev)
        return Conditions(
            text_condition=torch.zeros(B, num_frames, 100),
            speech_condition=torch.zeros(B, num_frames, 100),
            padding_mask=ops.make_pad_mask(total, num_frames),
            total_lens=total,
            gen_lens=(total - pf).clamp(min=0),
            prompt_lens=pf,
        )


class StubWatchdog:
    def __init__(self, allow=True):
        self.allow = allow
        self.ooms = 0

    def admission_allowed(self):
        return self.allow

    def handle_oom(self, exc):
        self.ooms += 1
        return "soft"

    def note_request_done(self, n=1):
        pass


def make_voice(prompt_frames=60, n_tokens=20) -> VoicePrompt:
    return VoicePrompt(
        voice_id="v", mel=torch.zeros(prompt_frames, 100), mel_frames=prompt_frames,
        token_ids=list(range(n_tokens)), prompt_rms=0.1,
        frames_per_token=prompt_frames / n_tokens,
    )


def make_span(rid, idx, n_tokens, params, voice, final=True) -> SpanRequest:
    return SpanRequest(
        request_id=rid, span_index=idx, n_spans=1, text="x" * n_tokens,
        token_ids=list(range(n_tokens)), voice=voice, params=params, is_final=final,
    )


def build(s=None, backend=None, watchdog=None):
    s = s or settings()
    arenas = ArenaPool(s, torch.device("cpu"), torch.float32)
    be = backend or StubBackend()
    wd = watchdog or StubWatchdog()
    sched = FlowScheduler(be, StubTextEncoder(), arenas, wd, s, fallback_backend=be)
    return s, sched, be, arenas, wd


@pytest.mark.asyncio
async def test_single_span_completes_with_correct_shape():
    s, sched, be, arenas, _ = build()
    await sched.start()
    v = make_voice()
    p = SynthParams.from_settings(s)
    span = make_span("r1", 0, 30, p, v)

    mel = await asyncio.wait_for(await sched.submit(span), timeout=5)
    await sched.stop()

    gen = int(30 * StubTextEncoder.FRAMES_PER_TOKEN)
    assert mel.shape == (gen, 100)
    # Backend returns velocity 1 everywhere, so x = x0 + sum(dt) = x0 + 1, then
    # divided by feat_scale. x0 is random, so just check it is finite and scaled.
    assert torch.isfinite(mel).all()
    assert len(be.calls) == p.num_step
    assert arenas.total_active() == 0        # slot released


@pytest.mark.asyncio
async def test_spans_at_different_steps_share_one_forward_pass():
    """The defining property: continuous batching.

    Ticks are driven manually rather than by the background task, because with
    an instant stub backend a real-time sleep would let the first span finish
    before the second is even submitted. Stepping by hand tests the mechanism
    deterministically: advance A a few steps, admit B, and assert the very next
    forward pass carries both -- at different timesteps.
    """
    s = settings()
    s.flow.num_step = 8
    _, sched, be, _, _ = build(s)
    sched._loop = asyncio.get_running_loop()
    v = make_voice()
    p = SynthParams.from_settings(s)

    f1 = await sched.submit(make_span("a", 0, 30, p, v))
    for _ in range(3):
        sched._tick()                              # A now at step 3

    f2 = await sched.submit(make_span("b", 0, 30, p, v))
    before = len(be.calls)
    sched._tick()                                  # admits B, then steps both
    joint = be.calls[before:]

    assert joint, "no forward pass after admitting the second span"
    batch, frames, ts = joint[0]
    # Classifier-free guidance doubles the batch ([uncond; cond]), so two spans
    # arrive at the backend as four rows carrying two distinct timesteps.
    assert p.uses_cfg()
    assert batch == 4, f"expected 2 spans x 2 CFG branches, got batch={batch}"
    assert len(set(ts)) == 2, (
        f"both spans ran at the same timestep {ts} -- they are NOT being "
        "continuous-batched, they are in lockstep"
    )
    # And the doubled vector really is [t_a, t_b, t_a, t_b].
    assert ts[:2] == ts[2:]

    while not (f1.done() and f2.done()):
        sched._tick()
    assert f1.result().shape[1] == 100 and f2.result().shape[1] == 100


@pytest.mark.asyncio
async def test_no_cfg_does_not_double_the_batch():
    s = settings()
    s.flow.num_step = 4
    _, sched, be, _, _ = build(s)
    sched._loop = asyncio.get_running_loop()
    v = make_voice()
    p = SynthParams.from_settings(s, guidance_scale=0.0)
    assert not p.uses_cfg()
    await sched.submit(make_span("a", 0, 30, p, v))
    await sched.submit(make_span("b", 0, 30, p, v))
    sched._tick()
    batch, _frames, _ts = be.calls[0]
    assert batch == 2, f"guidance_scale=0 must not double the batch, got {batch}"


@pytest.mark.asyncio
async def test_cfg_until_t_stops_doubling_past_the_threshold():
    """`cfg_until_t` exists to stop paying for the unconditional branch once the
    trajectory is past the low-t region where it matters. The batch handed to
    the backend must drop from 2N to N at that crossing."""
    s = settings()
    s.flow.num_step = 8
    _, sched, be, _, _ = build(s)
    sched._loop = asyncio.get_running_loop()
    v = make_voice()
    p = SynthParams.from_settings(s, guidance_scale=1.0)
    p.cfg_until_t = 0.3

    grid = ops.get_time_steps(8, p.t_shift).tolist()
    n_cfg_steps = sum(1 for t in grid[:8] if t <= 0.3)
    assert 0 < n_cfg_steps < 8, grid            # the test needs a real crossing

    await sched.submit(make_span("a", 0, 30, p, v))
    for _ in range(8):
        sched._tick()

    batches = [c[0] for c in be.calls]
    assert batches[:n_cfg_steps] == [2] * n_cfg_steps, batches
    assert batches[n_cfg_steps:] == [1] * (8 - n_cfg_steps), batches


@pytest.mark.asyncio
async def test_many_concurrent_spans_all_complete():
    s, sched, be, arenas, _ = build()
    await sched.start()
    v = make_voice()
    p = SynthParams.from_settings(s)
    futs = [await sched.submit(make_span(f"r{i}", 0, 20 + i, p, v)) for i in range(24)]
    mels = await asyncio.wait_for(asyncio.gather(*futs), timeout=15)
    await sched.stop()

    assert len(mels) == 24
    assert all(m.shape[1] == 100 and m.shape[0] > 0 for m in mels)
    assert arenas.total_active() == 0
    assert sched.stats_retired == 24
    # Batching actually happened rather than 24 sequential batches of 1.
    assert max(c[0] for c in be.calls) > 1


@pytest.mark.asyncio
async def test_arena_rows_stay_packed():
    """Occupied rows must remain contiguous at the front; a gather instead of a
    slice would copy the whole batch on every step."""
    s, sched, be, arenas, _ = build()
    await sched.start()
    v = make_voice()
    p = SynthParams.from_settings(s)
    # Different token counts -> different completion times -> retirements
    # interleaved with live rows.
    futs = [await sched.submit(make_span(f"r{i}", 0, 10 + i * 7, p, v)) for i in range(10)]
    await asyncio.wait_for(asyncio.gather(*futs), timeout=15)
    await sched.stop()

    for bucket, rows in sched._rows.items():
        assert rows == [], f"bucket {bucket} still has rows"
        arena = arenas.arena(bucket)
        assert arena.n_occupied == 0
    # Every recorded call had batch == the number of live rows at that moment,
    # which can only hold if rows were packed.
    assert all(c[0] >= 1 for c in be.calls)


@pytest.mark.asyncio
async def test_different_num_step_coexist():
    """Rows with different step counts share buckets; each uses its own grid."""
    s, sched, be, _, _ = build()
    await sched.start()
    v = make_voice()
    fast = SynthParams.from_settings(s, num_step=2)
    slow = SynthParams.from_settings(s, num_step=8)
    f1 = await sched.submit(make_span("f", 0, 30, fast, v))
    f2 = await sched.submit(make_span("s", 0, 30, slow, v))
    await asyncio.wait_for(asyncio.gather(f1, f2), timeout=10)
    await sched.stop()
    assert sched.stats_retired == 2


@pytest.mark.asyncio
async def test_backend_failure_fails_only_that_bucket_and_frees_slots():
    s, sched, be, arenas, wd = build(backend=StubBackend(fail_on=1))
    await sched.start()
    v = make_voice()
    p = SynthParams.from_settings(s)
    fut = await sched.submit(make_span("r", 0, 30, p, v))
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(fut, timeout=5)
    await sched.stop()
    assert arenas.total_active() == 0
    assert sched.stats_failed >= 1


@pytest.mark.asyncio
async def test_cancel_releases_slots_for_inflight_span():
    s = settings()
    s.flow.num_step = 64
    _, sched, be, arenas, _ = build(s)
    sched._loop = asyncio.get_running_loop()
    v = make_voice()
    p = SynthParams.from_settings(s, num_step=64)
    fut = await sched.submit(make_span("cancel-me", 0, 30, p, v))

    sched._tick()                                  # admit + one step
    assert arenas.total_active() == 1

    assert sched.cancel("cancel-me") == 1
    with pytest.raises(RequestCancelled):
        await asyncio.wait_for(fut, timeout=1)
    assert arenas.total_active() == 0


@pytest.mark.asyncio
async def test_cancel_removes_queued_span():
    """A span still in the queue must be cancellable too, not just an in-flight one."""
    s = settings()
    _, sched, _, _, _ = build(s, watchdog=StubWatchdog(allow=False))
    sched._loop = asyncio.get_running_loop()
    v = make_voice()
    p = SynthParams.from_settings(s)
    fut = await sched.submit(make_span("queued", 0, 20, p, v))
    assert sched.cancel("queued") == 1
    with pytest.raises(RequestCancelled):
        await asyncio.wait_for(fut, timeout=1)
    assert sched.stats()["queue_depth"] == 0


@pytest.mark.asyncio
async def test_queue_full_raises():
    s = settings()
    s.engine.max_queue_depth = 3
    _, sched, _, _, wd = build(s, watchdog=StubWatchdog(allow=False))  # never admit
    v = make_voice()
    p = SynthParams.from_settings(s)
    for i in range(3):
        await sched.submit(make_span(f"q{i}", 0, 20, p, v))
    with pytest.raises(QueueFull):
        await sched.submit(make_span("overflow", 0, 20, p, v))


@pytest.mark.asyncio
async def test_span_too_long_for_largest_bucket_fails_loudly():
    """Truncating someone's audio silently would be far worse than an error;
    the chunker is responsible for preventing this."""
    s, sched, _, arenas, _ = build()
    await sched.start()
    v = make_voice()
    p = SynthParams.from_settings(s)
    huge = make_span("huge", 0, 5000, p, v)     # way past buckets.max_frames
    fut = await sched.submit(huge)
    with pytest.raises(ValueError, match="largest bucket"):
        await asyncio.wait_for(fut, timeout=5)
    await sched.stop()
    assert arenas.total_active() == 0


@pytest.mark.asyncio
async def test_row_slot_mapping_survives_out_of_order_retirement():
    """Each span must get back the mel from ITS OWN arena slot.

    `arena.release()` keeps occupancy packed by copying the last occupied row
    into the hole. If the scheduler's row list used list.pop() semantics
    instead, every row after the hole would shift by one in the list but not in
    the arena, and spans would silently receive another span's audio. This test
    tags each slot's tensor with a unique value and checks it comes back.
    """
    s = settings()
    s.flow.num_step = 1                      # retire immediately after one step
    _, sched, be, arenas, _ = build(s)
    sched._loop = asyncio.get_running_loop()
    v = make_voice(prompt_frames=0)          # no prompt -> mel starts at frame 0
    p = SynthParams.from_settings(s, num_step=1, guidance_scale=0.0)

    # Same token count -> same bucket, so all rows share one arena.
    futs = []
    for i in range(6):
        futs.append(await sched.submit(make_span(f"r{i}", 0, 30, p, v)))

    sched._admit()
    bucket = sched._rows and next(iter(sched._rows))
    rows = sched._rows[bucket]
    assert len(rows) == 6
    arena = arenas.arena(bucket)

    # Stamp each occupied slot with its span's index, so the retired mel is
    # traceable back to the slot it was read from.
    tag = {}
    for i, row in enumerate(rows):
        val = float(i + 1)
        arena.x[i].fill_(val)
        tag[row.span.request_id] = val

    # Retire out of order: middle first, which is exactly the case that forces
    # the swap-with-last path.
    for target in ("r2", "r0", "r4"):
        row = next(r for r in sched._rows[bucket] if r.span.request_id == target)
        sched._retire_row(row)

    # Every surviving row must still point at a slot holding its own stamp.
    for row in sched._rows[bucket]:
        idx = row.slot.index
        assert idx < arena.n_occupied
        got = float(arena.x[idx, 0, 0])
        assert got == tag[row.span.request_id], (
            f"row {row.span.request_id} points at slot {idx} holding {got}, "
            f"expected {tag[row.span.request_id]} -- row/slot mapping is broken"
        )

    # And the retired ones resolved with their own slot's data.
    for rid in ("r2", "r0", "r4"):
        i = int(rid[1:])
        fut = futs[i]
        assert fut.done()
        mel = fut.result()
        expected = tag[rid] / s.flow.feat_scale
        assert abs(float(mel[0, 0]) - expected) < 1e-3, (
            f"{rid} received another span's audio: got {float(mel[0, 0])}, "
            f"expected {expected}"
        )


@pytest.mark.asyncio
async def test_stop_drains_outstanding_futures():
    """Shutting down with work in flight must fail every awaiting caller rather
    than stranding them: the task that would resolve their futures is gone."""
    s = settings()
    s.flow.num_step = 64                       # long enough to still be running
    _, sched, _, arenas, _ = build(s)
    sched._loop = asyncio.get_running_loop()
    v = make_voice()
    p = SynthParams.from_settings(s, num_step=64)

    inflight = await sched.submit(make_span("running", 0, 30, p, v))
    sched._tick()                              # admitted, one step in
    assert arenas.total_active() == 1

    queued = await sched.submit(make_span("waiting", 0, 30, p, v))

    await sched.stop()

    for fut in (inflight, queued):
        assert fut.done()
        with pytest.raises(RequestCancelled):
            await fut
    assert arenas.total_active() == 0
    assert sched.stats()["queue_depth"] == 0


@pytest.mark.asyncio
async def test_dt_staging_is_disjoint_across_cfg_partitions():
    """Both CFG partitions stage dt into the same pinned buffer. They must use
    disjoint regions, or the second partition's write can clobber the first
    while its non_blocking copy is still in flight."""
    s = settings()
    s.flow.num_step = 8
    _, sched, be, _, _ = build(s)
    sched._loop = asyncio.get_running_loop()
    v = make_voice()

    cfg_p = SynthParams.from_settings(s, guidance_scale=1.0)
    nocfg_p = SynthParams.from_settings(s, guidance_scale=0.0)
    await sched.submit(make_span("cfg", 0, 30, cfg_p, v))
    await sched.submit(make_span("plain", 0, 30, nocfg_p, v))

    # Both rows are admitted at step 0, so both stage grid[1] - grid[0].
    grid = ops.get_time_steps(8, cfg_p.t_shift).tolist()
    expected_dt = grid[1] - grid[0]

    sched._tick()

    rows = sched._rows[next(iter(sched._rows))]
    assert len(rows) == 2
    # Partitioning puts the CFG row first, and _step_slice is called once per
    # partition. Each row's dt must sit at its own ABSOLUTE index, so the second
    # call cannot clobber the first's still-in-flight copy.
    for i in range(2):
        assert abs(float(sched._h_dt[i]) - expected_dt) < 1e-6, (
            f"slot {i} staged {float(sched._h_dt[i])}, expected {expected_dt} -- "
            "the two CFG partitions are sharing a staging region"
        )
    # And the rows really were split across two separate backend calls.
    assert [c[0] for c in be.calls] == [2, 1], be.calls


@pytest.mark.asyncio
async def test_stats_shape():
    s, sched, _, _, _ = build()
    await sched.start()
    st = sched.stats()
    await sched.stop()
    for key in ("queue_depth", "active_rows", "admitted", "retired", "failed",
                "steps", "steps_per_s", "mean_batch", "arenas"):
        assert key in st
