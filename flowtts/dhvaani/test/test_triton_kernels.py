"""Tests for the fused OpenAI-Triton kernels.

Each kernel exists only to save CUDA launch overhead in the ODE loop; the torch
fallback is the semantic reference. These tests pin that the fallback itself is
correct (runs anywhere) and, when a GPU with Triton is present, that the kernel
agrees with it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from flowtts.dhvaani.model import triton_kernels as tk  # noqa: E402

CUDA = torch.cuda.is_available()
gpu_only = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def test_concat_fallback_matches_torch_cat():
    B, T, C = 3, 16, 100
    x, a, b = (torch.randn(B, T, C) for _ in range(3))
    out = tk.fused_concat_conditions(x, a, b)
    assert torch.equal(out, torch.cat([x, a, b], dim=2))


def test_concat_writes_into_supplied_buffer():
    B, T, C = 2, 8, 100
    x, a, b = (torch.randn(B, T, C) for _ in range(3))
    buf = torch.empty(B, T, 3 * C)
    out = tk.fused_concat_conditions(x, a, b, out=buf)
    assert out.data_ptr() == buf.data_ptr()
    assert torch.equal(buf, torch.cat([x, a, b], dim=2))


def test_euler_update_fallback():
    B, T, C = 3, 8, 100
    x = torch.zeros(B, T, C)
    v = torch.ones(B, T, C)
    dt = torch.tensor([0.1, 0.2, 0.3])
    tk.fused_euler_update(x, v, dt)
    assert torch.allclose(x[0], torch.full((T, C), 0.1))
    assert torch.allclose(x[2], torch.full((T, C), 0.3))


def test_euler_update_zeroes_padded_positions():
    """Padded frames still get a velocity from the network; letting it
    accumulate over num_step iterations can push the real frames out of fp16
    range through the attention softmax."""
    B, T, C = 2, 8, 100
    x = torch.zeros(B, T, C)
    v = torch.ones(B, T, C)
    dt = torch.tensor([1.0, 1.0])
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 5:] = True
    tk.fused_euler_update(x, v, dt, mask)
    assert torch.all(x[:, :5] == 1.0)
    assert torch.all(x[:, 5:] == 0.0)


def test_cfg_combine_fallback():
    B, T, C = 3, 8, 100
    uncond = torch.randn(B, T, C)
    cond = torch.randn(B, T, C)
    v2 = torch.cat([uncond, cond], dim=0)
    gs = torch.tensor([0.0, 1.0, 2.0])
    out = tk.fused_cfg_combine(v2, gs)
    for i, w in enumerate(gs.tolist()):
        assert torch.allclose(out[i], (1 + w) * cond[i] - w * uncond[i], atol=1e-5)


def test_cfg_combine_zero_guidance_is_identity():
    B, T, C = 2, 4, 100
    cond = torch.randn(B, T, C)
    v2 = torch.cat([torch.randn(B, T, C), cond], dim=0)
    out = tk.fused_cfg_combine(v2, torch.zeros(B))
    assert torch.allclose(out, cond, atol=1e-6)


@gpu_only
@pytest.mark.parametrize("shape", [(1, 128), (8, 384), (32, 512)])
def test_kernels_match_fallback_on_gpu(shape):
    B, T = shape
    C = 100
    dev = "cuda"
    for dtype in (torch.float16, torch.float32):
        x, a, b = (torch.randn(B, T, C, device=dev, dtype=dtype) for _ in range(3))
        ref = torch.cat([x, a, b], dim=2)
        got = tk.fused_concat_conditions(x, a, b)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

        xa = x.clone()
        xb = x.clone()
        v = torch.randn(B, T, C, device=dev, dtype=dtype)
        dt = torch.rand(B, device=dev)
        mask = torch.zeros(B, T, device=dev, dtype=torch.bool)
        mask[:, T - 16:] = True
        tk.fused_euler_update(xa, v, dt, mask)
        xb.add_(v * dt.view(-1, 1, 1).to(v.dtype)).masked_fill_(mask.unsqueeze(-1), 0.0)
        torch.testing.assert_close(xa, xb, rtol=1e-3, atol=1e-3)

        v2 = torch.randn(2 * B, T, C, device=dev, dtype=dtype)
        gs = torch.rand(B, device=dev)
        got = tk.fused_cfg_combine(v2, gs)
        u, c = v2.chunk(2, dim=0)
        w = gs.view(-1, 1, 1).to(dtype)
        torch.testing.assert_close(got, (1 + w) * c - w * u, rtol=1e-2, atol=1e-2)


def test_non_contiguous_input_falls_back_correctly():
    """A non-contiguous view would make the kernel read garbage, so the
    dispatcher must route it to torch."""
    B, T, C = 2, 8, 100
    base = torch.randn(B, T, C * 2)
    x = base[..., :C]            # non-contiguous
    a, b = torch.randn(B, T, C), torch.randn(B, T, C)
    out = tk.fused_concat_conditions(x, a, b)
    assert torch.equal(out, torch.cat([x, a, b], dim=2))
