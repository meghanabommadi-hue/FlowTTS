"""Tests for the ONNX/TensorRT export helpers.

`rel_shift` replaces the Zipformer's relative-to-absolute position shift with a
shape-dynamic formulation. Upstream has two implementations of that shift -- an
`as_strided` view for eager mode and a `torch.gather` path for tracing -- and
all three must agree exactly, otherwise a TensorRT engine would produce subtly
different attention scores from the PyTorch reference.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from flowtts.dhvaani.model.export_patch import rel_shift  # noqa: E402


def _as_strided_ref(pos_scores: torch.Tensor, T: int) -> torch.Tensor:
    """Literal transcription of the eager branch in
    RelPositionMultiheadAttentionWeights.forward."""
    H, B = pos_scores.shape[0], pos_scores.shape[1]
    return pos_scores.as_strided(
        (H, B, T, T),
        (
            pos_scores.stride(0),
            pos_scores.stride(1),
            pos_scores.stride(2) - pos_scores.stride(3),
            pos_scores.stride(3),
        ),
        storage_offset=pos_scores.stride(3) * (T - 1),
    )


def _gather_ref(pos_scores: torch.Tensor, T: int) -> torch.Tensor:
    """Literal transcription of the `torch.jit.is_tracing()` branch."""
    num_heads, batch_size, time1, n = pos_scores.shape
    rows = torch.arange(start=time1 - 1, end=-1, step=-1)
    cols = torch.arange(T)
    rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
    indexes = rows + cols
    flat = pos_scores.reshape(-1, n)
    out = torch.gather(flat, dim=1, index=indexes)
    return out.reshape(num_heads, batch_size, time1, T)


@pytest.mark.parametrize("T", [1, 2, 3, 5, 8, 13, 17, 64, 128])
@pytest.mark.parametrize("shape", [(4, 1), (4, 3), (2, 8)])
def test_rel_shift_matches_both_upstream_paths(T, shape):
    H, B = shape
    torch.manual_seed(T * 31 + H)
    x = torch.randn(H, B, T, 2 * T - 1)
    got = rel_shift(x, T)
    assert got.shape == (H, B, T, T)
    assert torch.equal(got, _as_strided_ref(x, T).contiguous())
    assert torch.equal(got, _gather_ref(x, T))


def test_rel_shift_index_semantics():
    """out[h, b, t, j] == x[h, b, t, T-1-t+j]."""
    T = 6
    x = torch.arange(1 * 1 * T * (2 * T - 1), dtype=torch.float32).reshape(1, 1, T, 2 * T - 1)
    got = rel_shift(x, T)
    for t in range(T):
        for j in range(T):
            assert got[0, 0, t, j] == x[0, 0, t, T - 1 - t + j]


def test_rel_shift_is_shape_dynamic():
    """The same code path must work for two different T without recompiling --
    this is what upstream's traced gather cannot do, because it bakes the
    batch size and sequence length in as constants."""
    for T in (32, 96):
        x = torch.randn(4, 2, T, 2 * T - 1)
        assert rel_shift(x, T).shape == (4, 2, T, T)


def test_dynamic_axes_declares_batch_and_frames():
    from flowtts.dhvaani.model.export_patch import dynamic_axes

    ax = dynamic_axes()
    assert ax["x_cat"] == {0: "batch", 1: "frames"}
    assert ax["t"] == {0: "batch"}
    assert ax["v"] == {0: "batch", 1: "frames"}
