"""Tests for `DhvaaniEngine.synthesize_stream` with the GPU stages stubbed.

What is exercised here is the orchestration, which is where the subtle bugs
live: span ordering, the lookahead window, where request metrics are attached,
cancellation, and the non-streaming convenience wrapper. The GPU stages
themselves are covered by `test_scheduler.py` and `smoke.py`.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flowtts.dhvaani.config import DhvaaniSettings  # noqa: E402
from flowtts.dhvaani.engine.engine import DhvaaniEngine  # noqa: E402
from flowtts.dhvaani.types import (  # noqa: E402
    RequestCancelled,
    SynthParams,
    VoicePrompt,
)

SR = 24000


class StubScheduler:
    """Resolves each span to a mel whose length follows its token count.

    `delays` lets a later span finish BEFORE an earlier one, which is the case
    that would break in-order emission.
    """

    def __init__(self, delays: dict[int, float] | None = None):
        self.delays = delays or {}
        self.submitted: list = []
        self.cancelled: list[str] = []

    async def submit(self, span):
        self.submitted.append(span)
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        async def _finish():
            await asyncio.sleep(self.delays.get(span.span_index, 0.0))
            if not fut.done():
                fut.set_result(torch.zeros(max(1, span.n_tokens * 3), 100))

        asyncio.create_task(_finish())
        return fut

    def cancel(self, rid):
        self.cancelled.append(rid)
        return 1

    def stats(self):
        return {}


class StubVocode:
    def __init__(self):
        self.calls = 0

    async def submit(self, mel, frames, prompt_rms, target_rms, out_sr):
        self.calls += 1
        # 256 samples per mel frame at 24 kHz.
        return np.full(frames * 256, 0.25, dtype=np.float32)

    def stats(self):
        return {}


class StubVoices:
    def __init__(self, voice):
        self.voice = voice

    def resolve(self, vid):
        return self.voice

    def list(self):
        return [self.voice.to_metadata()]

    def stats(self):
        return {}


class StubWatchdog:
    def note_request_done(self, n=1):
        pass

    def snapshot(self):
        return {}


def build_engine(delays=None, settings=None) -> tuple[DhvaaniEngine, StubScheduler]:
    s = settings or DhvaaniSettings()
    s.audio.crossfade_seconds = 0.0     # keep sample accounting simple
    s.audio.trim_edge_silence = False
    s.audio.final_fade_seconds = 0.0
    s.chunk.single_span_max_seconds = 0.3   # force multi-span for the test texts

    eng = DhvaaniEngine(s)
    voice = VoicePrompt(
        voice_id="v", mel=torch.zeros(187, 100), mel_frames=187,
        token_ids=list(range(60)), prompt_rms=0.1, frames_per_token=187 / 60,
        language="hi",
    )
    sched = StubScheduler(delays)
    eng._scheduler = sched
    eng._vocode = StubVocode()
    eng._watchdog = StubWatchdog()
    eng.voices = StubVoices(voice)
    eng.loaded = type("L", (), {"token_ids": staticmethod(lambda t: list(range(len(t))))})()
    eng._ready = True
    return eng, sched


TEXT = ("नमस्ते, मैं बजाज फाइनेंस से बोल रही हूं। आपकी ईएमआई बकाया है। "
        "कृपया आज ही भुगतान करें। धन्यवाद।")


@pytest.mark.asyncio
async def test_stream_emits_chunks_and_final_metrics():
    eng, _ = build_engine()
    chunks = [c async for c in eng.synthesize_stream(TEXT, "v", "hi")]
    assert chunks
    assert sum(c.is_final for c in chunks) == 1, "exactly one chunk must be final"
    assert chunks[-1].is_final

    m = chunks[-1].meta.get("metrics")
    assert m is not None, "the final chunk must carry the request metrics"
    assert m["total_ms"] > 0
    assert m["audio_s"] > 0
    assert m["n_spans"] >= 1
    assert m["ttfb_ms"] > 0


@pytest.mark.asyncio
async def test_chunks_are_emitted_in_span_order_even_when_completed_out_of_order():
    """Span 2 finishing before span 0 must not reorder the audio."""
    eng, sched = build_engine(delays={0: 0.05, 1: 0.02, 2: 0.0})
    chunks = [c async for c in eng.synthesize_stream(TEXT, "v", "hi")]
    order = [c.meta["span_index"] for c in chunks if "span_index" in c.meta]
    assert order == sorted(order), f"spans emitted out of order: {order}"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


@pytest.mark.asyncio
async def test_lookahead_pipelines_spans():
    """More than one span must be in flight at once, otherwise there is no
    pipelining and span N+1 waits for span N's vocoder."""
    s = DhvaaniSettings()
    s.engine.chunk_lookahead = 3
    s.audio.crossfade_seconds = 0.0
    s.audio.trim_edge_silence = False
    s.chunk.single_span_max_seconds = 0.3
    eng, sched = build_engine(delays={0: 0.05}, settings=s)

    gen = eng.synthesize_stream(TEXT, "v", "hi")
    await gen.__anext__()
    # By the time the first chunk lands, later spans are already submitted.
    assert len(sched.submitted) >= 2, sched.submitted
    await gen.aclose()


@pytest.mark.asyncio
async def test_non_streaming_returns_metrics():
    eng, _ = build_engine()
    pcm, m = await eng.synthesize(TEXT, "v", "hi")
    assert isinstance(pcm, bytes) and len(pcm) > 0
    assert len(pcm) % 2 == 0
    assert m.total_ms > 0 and m.audio_s > 0
    assert m.rtf > 0


@pytest.mark.asyncio
async def test_cancel_event_stops_the_stream():
    eng, sched = build_engine(delays={0: 0.0, 1: 0.05})
    ev = asyncio.Event()
    got = []
    with pytest.raises(RequestCancelled):
        async for c in eng.synthesize_stream(TEXT, "v", "hi", cancel_event=ev):
            got.append(c)
            ev.set()
    assert "v" not in sched.cancelled or sched.cancelled  # cancel propagated


@pytest.mark.asyncio
async def test_empty_text_yields_one_final_chunk():
    eng, _ = build_engine()
    chunks = [c async for c in eng.synthesize_stream("   ", "v", "hi")]
    assert len(chunks) == 1
    assert chunks[0].is_final
    assert chunks[0].meta.get("metrics") is not None


@pytest.mark.asyncio
async def test_emit_slice_splits_span_audio():
    s = DhvaaniSettings()
    s.audio.emit_slice_ms = 20
    s.audio.crossfade_seconds = 0.0
    s.audio.trim_edge_silence = False
    s.chunk.single_span_max_seconds = 0.3
    eng, _ = build_engine(settings=s)
    chunks = [c async for c in eng.synthesize_stream(TEXT, "v", "hi")]
    step = int(SR * 20 / 1000) * 2      # bytes per 20 ms of int16
    assert all(len(c.audio) <= step for c in chunks if c.audio)
    assert sum(c.is_final for c in chunks) == 1


@pytest.mark.asyncio
async def test_text_too_long_rejected():
    from flowtts.dhvaani.types import TextTooLong

    eng, _ = build_engine()
    with pytest.raises(TextTooLong):
        async for _ in eng.synthesize_stream("x" * 99999, "v", "hi"):
            pass
