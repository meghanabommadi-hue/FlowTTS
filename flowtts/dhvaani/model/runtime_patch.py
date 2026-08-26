"""Pipeline position: RUNTIME PATCH — make the Zipformer inference-clean.

Role in pipeline:
  Applied once by ``model/loader.py`` right after the weights load, before any
  inference. Purely a swap of training-time modules for inference-equivalent
  ones; no weights are touched.

Why
---
ZipVoice's ``SwooshR``/``SwooshL`` dispatch on whether ``k2`` is importable::

    elif "k2" not in sys.modules:
        return SwooshRFunction.apply(x)

k2 is an optional dependency that must be built against the exact torch version,
which is impossible on an NGC container. So the PyTorch fallback runs, and it
has two problems for a serving path:

1. **It cannot be CUDA-graph captured.** Its first statement is

       zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)

   which is a host-to-device copy from pageable memory. During stream capture
   that raises ``operation not permitted when stream is capturing``, and the
   failed capture then poisons the CUDA context for the rest of the process.
   Since the Zipformer issues several hundred kernel launches per forward, CUDA
   graphs are worth ~10x at small batch, so losing them is expensive.

2. **It silently returns fp32.** The autograd Function computes ``y``, then::

       if not requires_grad:
           return y
       ...
       if x.dtype == torch.float16 or torch.is_autocast_enabled():
           y = y.to(torch.float16)

   Under ``torch.inference_mode`` ``requires_grad`` is always False, so it
   returns at the early exit -- *before* the fp16 cast -- having already upcast
   its input to fp32 a few lines earlier. Every activation downstream of an
   activation function is therefore fp32, silently doubling bandwidth.

The replacements below are algebraically identical:

    swoosh_r(x) = log(1 + exp(x - 1)) - 0.08x - 0.313261687
    swoosh_l(x) = log(1 + exp(x - 4)) - 0.08x - 0.035

and ``log(1 + exp(z)) == softplus(z) == logaddexp(0, z)``, so the constants and
the numerics are unchanged -- while ``F.softplus`` allocates nothing, preserves
dtype, and is capturable. ``test_runtime_patch.py`` asserts the equivalence.
"""

from __future__ import annotations

import structlog
import torch
import torch.nn.functional as F
from torch import nn

logger = structlog.get_logger(__name__)


class CaptureSafeSwooshR(nn.Module):
    """``log(1 + exp(x - 1)) - 0.08x - 0.313261687``, allocation-free."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x - 1.0) - 0.08 * x - 0.313261687


class CaptureSafeSwooshL(nn.Module):
    """``log(1 + exp(x - 4)) - 0.08x - 0.035``, allocation-free."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x - 4.0) - 0.08 * x - 0.035


def _get_parent(model: nn.Module, dotted: str):
    parent, _, child = dotted.rpartition(".")
    return (model.get_submodule(parent) if parent else model), child


def make_inference_safe(model: nn.Module, drop_training_modules: bool = True) -> dict:
    """Swap training-time modules for inference-equivalent ones, in place.

    Args:
        model: the ZipVoice module (or any submodule tree).
        drop_training_modules: also replace Balancer / Whiten / Dropout3 with
            Identity. Those are already no-ops in eval mode, but removing them
            removes a Python-level module call per invocation, and there are
            hundreds per forward.

    Returns:
        A count of what was replaced, for logging.
    """
    try:
        from zipvoice.models.modules.scaling import (
            Balancer, Dropout3, SwooshL, SwooshR, Whiten,
        )
    except Exception as e:  # pragma: no cover
        logger.warning("runtime_patch_unavailable", error=str(e))
        return {}

    swaps: dict[str, nn.Module] = {}
    counts = {"swoosh_r": 0, "swoosh_l": 0, "identity": 0}

    for name, mod in model.named_modules():
        if isinstance(mod, SwooshR):
            swaps[name] = CaptureSafeSwooshR()
            counts["swoosh_r"] += 1
        elif isinstance(mod, SwooshL):
            swaps[name] = CaptureSafeSwooshL()
            counts["swoosh_l"] += 1
        elif drop_training_modules and isinstance(mod, (Balancer, Whiten, Dropout3)):
            swaps[name] = nn.Identity()
            counts["identity"] += 1

    for name, replacement in swaps.items():
        parent, child = _get_parent(model, name)
        setattr(parent, child, replacement)

    if swaps:
        logger.info("runtime_patch_applied", **counts)
    return counts
