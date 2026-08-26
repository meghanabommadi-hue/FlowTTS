"""Tests for streaming span stitching.

Spans are rendered by independent flow trajectories, so joining them naively
produces an audible click at every boundary. These tests pin the two properties
that matter: the seam is smooth, and no sample is invented or lost beyond the
crossfade overlap itself.
"""

from __future__ import annotations

import numpy as np

from flowtts.dhvaani.engine.stitch import SpanStitcher

SR = 24000


def _stitch(spans, crossfade_s=0.06, trim=False):
    st = SpanStitcher(SR, crossfade_s, final_fade_s=0.0, trim_edges=trim)
    out = [st.push(s, is_final=(i == len(spans) - 1)) for i, s in enumerate(spans)]
    return np.concatenate(out) if out else np.zeros(0, np.float32), st


def test_length_shrinks_by_exactly_one_overlap_per_join():
    """Overlap-add consumes `k` samples per join -- the same accounting as
    upstream `cross_fade_concat`."""
    k = int(0.06 * SR)
    spans = [np.full(int(0.5 * SR), 0.5, np.float32),
             np.full(int(1.0 * SR), -0.5, np.float32),
             np.full(int(0.8 * SR), 0.25, np.float32)]
    out, _ = _stitch(spans)
    assert out.size == sum(s.size for s in spans) - 2 * k


def test_seam_is_smooth():
    spans = [np.full(int(0.5 * SR), 0.5, np.float32),
             np.full(int(0.5 * SR), -0.5, np.float32)]
    out, _ = _stitch(spans)
    naive = np.concatenate(spans)
    assert np.abs(np.diff(out)).max() < 0.01
    # The naive join steps a full 1.0 in one sample.
    assert np.abs(np.diff(naive)).max() > 0.9


def test_crossfade_is_monotonic_between_levels():
    k = int(0.06 * SR)
    spans = [np.full(int(0.5 * SR), 1.0, np.float32),
             np.full(int(0.5 * SR), -1.0, np.float32)]
    out, _ = _stitch(spans)
    j = int(0.5 * SR) - k
    seg = out[j:j + k]
    assert np.all(np.diff(seg) <= 1e-6)
    assert seg[0] > 0.9 and seg[-1] < -0.9


def test_no_nan_or_clipping():
    rng = np.random.default_rng(0)
    spans = [rng.standard_normal(int(0.4 * SR)).astype(np.float32) * 0.3 for _ in range(4)]
    out, _ = _stitch(spans)
    assert np.isfinite(out).all()


def test_flush_drains_held_tail_losslessly():
    st = SpanStitcher(SR, 0.06, final_fade_s=0.0, trim_edges=False)
    span = np.ones(int(0.5 * SR), np.float32)
    a = st.push(span, is_final=False)
    b = st.flush()
    assert a.size + b.size == span.size
    assert st.flush().size == 0  # idempotent


def test_single_final_span_passes_through():
    st = SpanStitcher(SR, 0.06, final_fade_s=0.0, trim_edges=False)
    span = np.ones(int(0.3 * SR), np.float32)
    out = st.push(span, is_final=True)
    assert out.size == span.size


def test_final_fade_applied():
    st = SpanStitcher(SR, 0.0, final_fade_s=0.02, trim_edges=False)
    out = st.push(np.ones(int(0.3 * SR), np.float32), is_final=True)
    assert out[-1] < 0.05          # faded to silence
    assert out[0] == 1.0           # head untouched


def test_span_shorter_than_crossfade_keeps_all_samples():
    """Degenerate case: the next span is shorter than the held tail. We must not
    silently drop the held audio."""
    st = SpanStitcher(SR, 0.06, final_fade_s=0.0, trim_edges=False)
    a = st.push(np.ones(int(0.5 * SR), np.float32), is_final=False)
    tiny = np.full(100, 0.2, np.float32)
    b = st.push(tiny, is_final=True)
    held = int(0.06 * SR)
    assert a.size == int(0.5 * SR) - held
    assert b.size == held + tiny.size


def test_edge_silence_trim():
    st = SpanStitcher(SR, 0.0, final_fade_s=0.0, trim_edges=True, threshold_db=-45.0)
    sig = np.concatenate([
        np.zeros(int(0.3 * SR), np.float32),
        np.full(int(0.5 * SR), 0.4, np.float32),
        np.zeros(int(0.3 * SR), np.float32),
    ])
    out = st.push(sig, is_final=True)
    assert out.size < sig.size
    assert out.size >= int(0.5 * SR)      # speech itself is preserved


def test_all_quiet_span_does_not_vanish():
    """A silent span (model artefact on punctuation) must still emit something:
    a client counting samples would otherwise desynchronise."""
    st = SpanStitcher(SR, 0.0, final_fade_s=0.0, trim_edges=True)
    out = st.push(np.zeros(int(0.4 * SR), np.float32), is_final=True)
    assert out.size > 0
