"""Pipeline position: FUSED KERNELS — elementwise glue around the flow decoder.

Role in pipeline:
  The ODE loop is dominated by one big Zipformer call, but around it sit several
  elementwise ops per step per bucket:

      concat(x, text_c, speech_c) -> (B, T, 300)
      classifier-free-guidance combine of the doubled output
      x += v * dt   (with padded positions zeroed)

  In eager PyTorch that is roughly six kernel launches plus a large temporary
  for the concat. At num_step=8 and several active buckets it becomes thousands
  of launches per second of pure overhead, and the concat temporary is exactly
  the kind of variable-size allocation that fragments VRAM.

  These OpenAI-Triton kernels fuse each group into one launch and write into a
  caller-supplied buffer so the concat temporary can live in an arena.

Every kernel has a pure-torch fallback with identical semantics. The fallback is
the correctness reference; `flowtts/dhvaani/test/test_triton_kernels.py` asserts
they agree. If Triton is unavailable, or an input is non-contiguous, or the
feature says off, the fallback runs and nothing else changes.
"""

from __future__ import annotations

import structlog
import torch

logger = structlog.get_logger(__name__)

try:  # pragma: no cover - depends on the install
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception as _e:  # pragma: no cover
    triton = None
    tl = None
    HAVE_TRITON = False
    logger.info("triton_kernels_unavailable", error=str(_e))


def _enabled() -> bool:
    from flowtts.dhvaani.config import dhv_settings

    return HAVE_TRITON and dhv_settings.backend.use_triton_kernels


# ---------------------------------------------------------------------------
# concat(x, text, speech) -> (B, T, 3C)
# ---------------------------------------------------------------------------
if HAVE_TRITON:

    @triton.jit
    def _concat3_kernel(
        X, TXT, SPCH, OUT,
        n_rows, C: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """One program per (row = b*T + t). Writes 3C contiguous outputs."""
        row = tl.program_id(0)
        if row >= n_rows:
            return
        offs = tl.arange(0, BLOCK)
        m = offs < C
        src = row * C + offs
        dst = row * 3 * C + offs
        tl.store(OUT + dst, tl.load(X + src, mask=m, other=0.0), mask=m)
        tl.store(OUT + dst + C, tl.load(TXT + src, mask=m, other=0.0), mask=m)
        tl.store(OUT + dst + 2 * C, tl.load(SPCH + src, mask=m, other=0.0), mask=m)

    @triton.jit
    def _euler_kernel(
        X, V, DT, MASK,
        n_rows, T, C: tl.constexpr, BLOCK: tl.constexpr, HAS_MASK: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        b = row // T
        offs = tl.arange(0, BLOCK)
        m = offs < C
        base = row * C + offs
        dt = tl.load(DT + b)
        x = tl.load(X + base, mask=m, other=0.0)
        v = tl.load(V + base, mask=m, other=0.0)
        out = x + v * dt
        if HAS_MASK:
            padded = tl.load(MASK + row)
            out = tl.where(padded != 0, 0.0, out)
        tl.store(X + base, out, mask=m)

    @triton.jit
    def _cfg_kernel(
        V2, GS, OUT,
        n_rows, T, C: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """V2 is (2B, T, C) with [uncond; cond]; n_rows == B * T."""
        row = tl.program_id(0)
        if row >= n_rows:
            return
        b = row // T
        offs = tl.arange(0, BLOCK)
        m = offs < C
        uncond = tl.load(V2 + row * C + offs, mask=m, other=0.0)
        cond = tl.load(V2 + (n_rows + row) * C + offs, mask=m, other=0.0)
        w = tl.load(GS + b)
        tl.store(OUT + row * C + offs, (1.0 + w) * cond - w * uncond, mask=m)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Public API (Triton when possible, torch otherwise)
# ---------------------------------------------------------------------------
def fused_concat_conditions(
    x: torch.Tensor,
    text_c: torch.Tensor,
    speech_c: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """`cat([x, text_c, speech_c], dim=2)` -> `(B, T, 3C)`.

    Pass `out` to reuse a buffer instead of allocating; this is what keeps the
    300-channel activation off the caching allocator's hot path.
    """
    B, T, C = x.shape
    if out is None:
        out = torch.empty((B, T, 3 * C), device=x.device, dtype=x.dtype)

    contiguous = x.is_contiguous() and text_c.is_contiguous() and speech_c.is_contiguous()
    if _enabled() and x.is_cuda and contiguous and out.is_contiguous():
        n_rows = B * T
        _concat3_kernel[(n_rows,)](
            x, text_c, speech_c, out, n_rows, C=C, BLOCK=_next_pow2(C)
        )
        return out

    out[:, :, :C].copy_(x)
    out[:, :, C : 2 * C].copy_(text_c)
    out[:, :, 2 * C :].copy_(speech_c)
    return out


def fused_euler_update(
    x: torch.Tensor,
    v: torch.Tensor,
    dt: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-place `x += v * dt[:, None, None]`, zeroing padded positions.

    Zeroing matters: padded frames are garbage that the Zipformer still produces
    a velocity for, and leaving it to accumulate over `num_step` iterations can
    grow large enough to affect the fp16 range of the real frames through the
    attention softmax.
    """
    B, T, C = x.shape
    contiguous = x.is_contiguous() and v.is_contiguous()
    if _enabled() and x.is_cuda and contiguous:
        mask_arg = padding_mask.contiguous() if padding_mask is not None else x
        _euler_kernel[(B * T,)](
            x, v, dt.to(torch.float32).contiguous(), mask_arg,
            B * T, T, C=C, BLOCK=_next_pow2(C), HAS_MASK=padding_mask is not None,
        )
        return x

    x.add_(v * dt.view(-1, 1, 1).to(v.dtype))
    if padding_mask is not None:
        x.masked_fill_(padding_mask.unsqueeze(-1), 0.0)
    return x


def fused_cfg_combine(
    v2: torch.Tensor, gs: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """`(1 + w) * v_cond - w * v_uncond` from a `(2B, T, C)` doubled output."""
    twoB, T, C = v2.shape
    B = twoB // 2
    if out is None:
        out = torch.empty((B, T, C), device=v2.device, dtype=v2.dtype)

    if _enabled() and v2.is_cuda and v2.is_contiguous() and out.is_contiguous():
        _cfg_kernel[(B * T,)](
            v2, gs.to(torch.float32).contiguous(), out, B * T, T,
            C=C, BLOCK=_next_pow2(C),
        )
        return out

    v_uncond, v_cond = v2.chunk(2, dim=0)
    w = gs.view(-1, 1, 1).to(v2.dtype)
    torch.sub((1.0 + w) * v_cond, w * v_uncond, out=out)
    return out
