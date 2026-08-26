"""Unit tests for flowtts.processing.stitch — seamless chunk joining.

The artifacts these guard against are the ones a listener actually notices in a
streamed utterance: the ~200 ms silence hole OmniVoice's edge padding leaves at
every seam, DC steps that click, and the 3 dB power dip a linear crossfade
causes. Pure NumPy — no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from flowtts.processing.stitch import (
    StreamStitcher,
    equal_power_crossfade,
    match_level,
    remove_dc,
    stitch_all,
    trim_silence,
)

SR = 24000


def tone(freq: float, seconds: float, amp: float = 0.3, dc: float = 0.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t) + dc).astype(np.float32)


def padded(wav: np.ndarray, seconds: float = 0.1) -> np.ndarray:
    """A clip as OmniVoice emits it: real audio between two silent margins."""
    silence = np.zeros(int(SR * seconds), dtype=np.float32)
    return np.concatenate([silence, wav, silence])


def rms(wav: np.ndarray) -> float:
    return float(np.sqrt((wav.astype(np.float64) ** 2).mean())) if wav.size else 0.0


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def test_remove_dc():
    assert abs(remove_dc(tone(200, 0.5, dc=0.05)).mean()) < 1e-6


def test_remove_dc_on_empty():
    assert remove_dc(np.zeros(0, dtype=np.float32)).size == 0


def test_trim_silence_removes_the_padding():
    trimmed = trim_silence(padded(tone(200, 1.0), 0.1), SR, keep_ms=20)
    # 1.0 s of audio plus the 20 ms margin kept on each side.
    assert 1.02 <= len(trimmed) / SR <= 1.06, len(trimmed) / SR


def test_trim_silence_keeps_all_speech():
    speech = tone(200, 1.0)
    trimmed = trim_silence(padded(speech, 0.1), SR, keep_ms=20)
    assert rms(trimmed) > rms(speech) * 0.9


def test_trim_silence_on_a_fully_silent_chunk():
    out = trim_silence(np.zeros(SR, dtype=np.float32), SR)
    assert 0 < out.size <= SR


def test_trim_silence_on_a_chunk_shorter_than_one_window():
    tiny = tone(200, 0.005)
    assert trim_silence(tiny, SR).size == tiny.size


def test_equal_power_crossfade_holds_power():
    """Constant power across the overlap, for the signals this actually joins.

    Two adjacent speech chunks are uncorrelated, so they sum in power rather
    than amplitude. Under that assumption the sin/cos pair holds RMS constant
    while a linear ramp dips ~3 dB in the middle — which is why the linear
    version is audible as a dip at every seam. (For *identical* inputs the
    relationship inverts, which is not the case being optimized for here.)
    """
    rng = np.random.default_rng(0)
    n = 4800
    a = rng.standard_normal(n).astype(np.float32) * 0.3
    b = rng.standard_normal(n).astype(np.float32) * 0.3

    fused = equal_power_crossfade(a, b)
    assert fused.size == n

    # RMS measured in the middle of the overlap vs. at its edges.
    mid = rms(fused[n // 2 - 200: n // 2 + 200])
    edge = (rms(fused[:200]) + rms(fused[-200:])) / 2
    assert abs(mid - edge) / edge < 0.15, (mid, edge)

    ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
    linear = a * ramp + b * (1 - ramp)
    linear_mid = rms(linear[n // 2 - 200: n // 2 + 200])
    assert linear_mid < mid, (linear_mid, mid)


def test_crossfade_endpoints_belong_to_their_own_chunk():
    a = np.ones(100, dtype=np.float32)
    b = np.full(100, 2.0, dtype=np.float32)
    fused = equal_power_crossfade(a, b)
    assert fused[0] == pytest.approx(1.0, abs=0.05)
    assert fused[-1] == pytest.approx(2.0, abs=0.05)


def test_match_level_is_clamped():
    loud = np.full(1000, 0.5, dtype=np.float32)
    quiet = np.full(1000, 0.001, dtype=np.float32)
    gain = match_level(loud, quiet, max_gain_db=1.5)
    assert gain <= 10 ** (1.5 / 20) + 1e-6, gain


def test_match_level_on_silence_is_unity():
    assert match_level(np.zeros(100, dtype=np.float32),
                       np.ones(100, dtype=np.float32)) == 1.0


# ---------------------------------------------------------------------------
# The streaming stitcher
# ---------------------------------------------------------------------------
def _chunks():
    return [padded(tone(200, 1.0, dc=0.01)),
            padded(tone(200, 1.0, 0.28, dc=-0.02)),
            padded(tone(200, 0.8, 0.31))]


def test_padding_between_chunks_is_removed():
    """Naive concatenation leaves a ~200 ms hole at every seam."""
    chunks = _chunks()
    naive = sum(len(c) for c in chunks) / SR
    stitched = len(stitch_all(chunks, SR)) / SR
    assert stitched < naive - 0.35, (stitched, naive)
    # …without eating the speech itself.
    assert stitched > 2.7, stitched


def test_stitched_output_has_no_dc_step():
    out = stitch_all(_chunks(), SR)
    assert abs(out.mean()) < 1e-3


def test_stitched_output_has_no_click():
    """A discontinuity would show up as a sample step far above the waveform's own."""
    out = stitch_all(_chunks(), SR)
    largest_step = np.abs(np.diff(out)).max()
    natural_step = 0.31 * 2 * np.pi * 200 / SR       # a 200 Hz tone's own slope
    assert largest_step < natural_step * 4, (largest_step, natural_step)


def test_output_never_clips():
    out = stitch_all([padded(tone(200, 0.5, 0.95)), padded(tone(200, 0.5, 0.95))], SR)
    assert np.abs(out).max() <= 1.0


def test_streaming_emits_before_the_last_chunk_arrives():
    """The whole point: audio ships while later chunks are still generating."""
    stitcher = StreamStitcher(SR)
    first = stitcher.push(padded(tone(200, 1.0)), is_final=False)
    assert first.size > 0


def test_held_overlap_is_flushed_on_the_final_chunk():
    stitcher = StreamStitcher(SR, overlap_ms=20)
    emitted = [stitcher.push(c, is_final=(i == 2)) for i, c in enumerate(_chunks())]
    assert stitcher.flush().size == 0        # already flushed by is_final
    assert sum(e.size for e in emitted) > 0


def test_flush_is_idempotent():
    stitcher = StreamStitcher(SR)
    stitcher.push(padded(tone(200, 0.5)), is_final=False)
    first = stitcher.flush()
    assert stitcher.flush().size == 0
    assert first.size > 0


def test_stream_ends_on_a_fade():
    """A stream cut mid-sample clicks in the client's player."""
    stitcher = StreamStitcher(SR, final_fade_ms=12)
    parts = [stitcher.push(c, is_final=(i == 2)) for i, c in enumerate(_chunks())]
    tail = np.concatenate([p for p in parts if p.size])[-32:]
    assert abs(tail[-1]) < 0.02, tail[-1]


def test_empty_chunk_does_not_break_the_stream():
    stitcher = StreamStitcher(SR)
    stitcher.push(padded(tone(200, 0.5)), is_final=False)
    assert stitcher.push(np.zeros(0, dtype=np.float32), is_final=False).size == 0
    assert stitcher.push(padded(tone(200, 0.5)), is_final=True).size > 0


def test_chunk_shorter_than_the_overlap():
    stitcher = StreamStitcher(SR, overlap_ms=20)
    stitcher.push(padded(tone(200, 0.5)), is_final=False)
    tiny = tone(200, 0.005)
    stitcher.push(tiny, is_final=False)
    assert stitcher.push(padded(tone(200, 0.5)), is_final=True).size > 0


def test_single_chunk_passes_through():
    out = stitch_all([padded(tone(200, 1.0))], SR)
    assert 1.0 <= len(out) / SR <= 1.1


def test_no_chunks():
    assert stitch_all([], SR).size == 0


def test_trim_can_be_disabled():
    chunks = _chunks()
    kept = len(stitch_all(chunks, SR, trim=False)) / SR
    trimmed = len(stitch_all(chunks, SR, trim=True)) / SR
    assert kept > trimmed
