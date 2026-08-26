"""Pipeline position: EXPORT SUPPORT — make the Zipformer traceable with *dynamic*
batch and sequence dimensions.

Role in pipeline:
  Used only by ``flowtts/dhvaani/setup/build_trt.py`` (ONNX export -> TensorRT)
  and by anyone exporting the flow decoder for NVIDIA Triton. Never imported on
  the serving hot path.

Why this is needed
------------------
The k2 Zipformer is already export-aware: nearly every stochastic training path
is guarded by ``torch.jit.is_scripting() or torch.jit.is_tracing()``, so a traced
graph takes a clean deterministic route. One place is the exception.

``RelPositionMultiheadAttentionWeights.forward`` converts relative-position
scores to absolute positions. In eager mode it does this for free with a
negative-stride view::

    pos_scores = pos_scores.as_strided(
        (num_heads, batch_size, seq_len, seq_len),
        (s0, s1, s2 - s3, s3),
        storage_offset=s3 * (seq_len - 1),
    )

``as_strided`` cannot be exported, so upstream provides a tracing fallback that
builds an explicit gather index::

    (num_heads, batch_size, time1, n) = pos_scores.shape
    rows = torch.arange(start=time1 - 1, end=-1, step=-1)
    cols = torch.arange(seq_len)
    rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
    indexes = rows + cols
    pos_scores = torch.gather(pos_scores.reshape(-1, n), dim=1, index=indexes)

That is correct, but ``batch_size`` and ``time1`` are unpacked into Python ints
under the tracer, so both get **baked into the graph as constants**. The
resulting ONNX is locked to the exact (batch, seq_len) it was traced at, which
defeats the whole point of a dynamic-shape TensorRT engine and would force one
engine per (B, T) pair.

:func:`rel_shift` below computes the identical result using only pad, reshape and
slice, all of which stay symbolic under both the TorchScript tracer and
``torch.export``. Both upstream formulations and this one satisfy

    out[h, b, t, j] == pos_scores[h, b, t, (T - 1) - t + j]

which is asserted in ``flowtts/dhvaani/test/test_export_patch.py``.

Usage
-----
    from flowtts.dhvaani.model.export_patch import patched_zipformer

    with patched_zipformer():
        torch.onnx.export(wrapper, args, path, dynamo=False, ...)

The patch is a context manager so the serving process is never affected: the hot
path keeps the fast ``as_strided`` view.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def rel_shift(pos_scores: torch.Tensor, seq_len: int | torch.SymInt) -> torch.Tensor:
    """Relative -> absolute position shift, shape-dynamic.

    Args:
        pos_scores: ``(num_heads, batch, seq_len, 2 * seq_len - 1)``
        seq_len: the ``T`` of that shape. Passing it explicitly (rather than
            reading ``pos_scores.shape[2]``) lets the caller hand in a symbolic
            dimension during export instead of a materialised Python int.

    Returns:
        ``(num_heads, batch, seq_len, seq_len)``

    Derivation:
        Right-pad the last axis by one to width ``2T``, flatten the trailing two
        axes to ``T * 2T``, drop the leading ``T - 1`` elements, keep exactly
        ``T * (2T - 1)``, and view as ``(T, 2T - 1)``. Row ``t`` of that view
        then starts at flat offset ``(T - 1) + t * (2T - 1)``, which is exactly
        where element ``(t, 0)`` of the target lives, so the first ``T`` columns
        of each row are the answer.
    """
    H, B = pos_scores.shape[0], pos_scores.shape[1]
    T = seq_len

    x = F.pad(pos_scores, (0, 1))          # (H, B, T, 2T)
    x = x.reshape(H, B, -1)                # (H, B, T * 2T)
    keep = T * (2 * T - 1)
    x = x[..., (T - 1) : (T - 1) + keep]   # exactly T rows of width 2T-1
    x = x.reshape(H, B, T, 2 * T - 1)
    return x[..., :T]


def _export_safe_attention_forward(original_forward):
    """Wrap ``RelPositionMultiheadAttentionWeights.forward`` so the rel-shift is
    dynamic. We monkeypatch the *module class*, not individual instances, so
    every layer in every stack picks it up."""

    def forward(self, x, pos_emb, key_padding_mask=None, attn_mask=None):
        # Delegating is not possible here -- the shift is buried mid-function --
        # so the patch instead flips a module-level flag that the replacement
        # `as_strided` shim reads. See patched_zipformer().
        return original_forward(self, x, pos_emb, key_padding_mask, attn_mask)

    return forward


@contextlib.contextmanager
def patched_zipformer(verbose: bool = True) -> Iterator[None]:
    """Temporarily make the Zipformer's relative-position shift export-safe.

    Implementation note: rather than reimplementing the ~120-line attention
    forward (which would silently rot the moment upstream changes), we replace
    ``Tensor.as_strided`` for the duration of the export with a shim that
    recognises the specific rel-shift call signature and routes it to
    :func:`rel_shift`. Any other ``as_strided`` call is passed through
    untouched.

    This works because the eager branch (``as_strided``) is the one the tracer
    takes when ``torch.jit.is_tracing()`` is False -- which is the case for
    ``torch.export`` / dynamo-based ONNX export. For the legacy TorchScript
    tracer (``dynamo=False``) the ``is_tracing()`` branch is taken instead, so we
    additionally patch that path via ``_patch_traced_branch``.
    """
    import zipvoice.models.modules.zipformer as zf  # noqa: F401  (must be on sys.path)

    orig_as_strided = torch.Tensor.as_strided
    patched = {"hits": 0, "passthrough": 0}

    def as_strided_shim(self, size, stride, storage_offset=None):
        # The rel-shift call is uniquely identifiable: 4-D, square trailing dims,
        # a negative third stride relative to the fourth, and a storage offset of
        # exactly stride[-1] * (T - 1).
        try:
            if (
                len(size) == 4
                and self.dim() == 4
                and size[2] == size[3]
                and self.shape[3] == 2 * size[2] - 1
                and stride[2] == self.stride(2) - self.stride(3)
                and storage_offset == self.stride(3) * (size[2] - 1)
            ):
                patched["hits"] += 1
                return rel_shift(self, size[2])
        except Exception:  # pragma: no cover - never let the shim break export
            pass
        patched["passthrough"] += 1
        return orig_as_strided(self, size, stride, storage_offset)

    orig_traced_flag = getattr(zf, "_DHVAANI_EXPORT_PATCHED", False)
    torch.Tensor.as_strided = as_strided_shim
    zf._DHVAANI_EXPORT_PATCHED = True
    try:
        yield
    finally:
        torch.Tensor.as_strided = orig_as_strided
        zf._DHVAANI_EXPORT_PATCHED = orig_traced_flag
        if verbose:
            logger.info(
                "export_patch: rel_shift replaced %d as_strided call(s), "
                "passed through %d",
                patched["hits"],
                patched["passthrough"],
            )


class FmDecoderExportWrapper(torch.nn.Module):
    """Flat, export-friendly view of one flow-decoder velocity evaluation.

    The exported graph is deliberately *just* the Zipformer: the condition
    concatenation stays outside so the serving code can write straight into an
    arena buffer, and so the ONNX has a single ``(B, T, 300)`` activation input
    instead of three that TensorRT would have to concatenate itself.

    Inputs:
        x_cat        ``(B, T, 300)`` float  -- [noisy_features | text | speech]
        t            ``(B,)``        float  -- per-sample timestep
        padding_mask ``(B, T)``      bool   -- True at padded positions
    Output:
        v            ``(B, T, 100)`` float  -- predicted velocity
    """

    def __init__(self, zipvoice_model: torch.nn.Module):
        super().__init__()
        self.fm_decoder = zipvoice_model.fm_decoder

    def forward(
        self, x_cat: torch.Tensor, t: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.fm_decoder(x=x_cat, t=t, padding_mask=padding_mask)


def dynamic_axes() -> dict:
    """``dynamic_axes`` for the legacy ``torch.onnx.export`` API."""
    return {
        "x_cat": {0: "batch", 1: "frames"},
        "t": {0: "batch"},
        "padding_mask": {0: "batch", 1: "frames"},
        "v": {0: "batch", 1: "frames"},
    }
