"""Pipeline position: BACKEND BASE — shared plumbing for flow-step executors.

Role in pipeline:
  `types.FmStepBackend` is the structural contract; this is the concrete shared
  implementation the three backends inherit (timing, shape bookkeeping, the
  condition concat, and a scratch buffer for the (B, T, 300) activation).

A backend's only job is: given a heterogeneous batch of noisy features and their
conditions, return the predicted velocity. Everything about the ODE -- the
timestep grid, CFG, the Euler update -- lives in the scheduler, so a new backend
never has to reimplement the maths.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Sequence

import structlog
import torch

from flowtts.dhvaani.config import N_MELS, dhv_settings
from flowtts.dhvaani.model.triton_kernels import fused_concat_conditions

logger = structlog.get_logger(__name__)


class BaseFmBackend:
    """Common behaviour: scratch buffers, shape stats, the concat step."""

    name = "base"

    def __init__(self, loaded, settings=None):
        self._s = settings or dhv_settings
        self._m = loaded
        self.device = loaded.device
        self.dtype = loaded.dtype
        # One scratch buffer per (batch, frames) shape holding the concatenated
        # (B, T, 300) activation. Allocated on first use for a shape and then
        # reused forever, so the ODE loop never allocates.
        self._scratch: dict[tuple[int, int], torch.Tensor] = {}
        self._calls: dict[tuple[int, int], int] = defaultdict(int)
        self._ns: dict[tuple[int, int], float] = defaultdict(float)

    # -- helpers -------------------------------------------------------------
    def _cat_buffer(self, B: int, T: int) -> torch.Tensor:
        key = (B, T)
        buf = self._scratch.get(key)
        if buf is None:
            buf = torch.empty((B, T, N_MELS * 3), device=self.device, dtype=self.dtype)
            self._scratch[key] = buf
        return buf

    def _concat(
        self, x: torch.Tensor, text_c: torch.Tensor, speech_c: torch.Tensor
    ) -> torch.Tensor:
        B, T, _ = x.shape
        return fused_concat_conditions(x, text_c, speech_c, out=self._cat_buffer(B, T))

    def _record(self, B: int, T: int, t0: float) -> None:
        key = (B, T)
        self._calls[key] += 1
        self._ns[key] += time.perf_counter() - t0

    # -- contract ------------------------------------------------------------
    def supports_bucket(self, batch: int, frames: int) -> bool:
        return True

    def fm_step(self, x, text_condition, speech_condition, t, padding_mask):
        raise NotImplementedError

    def warmup(self, buckets: Sequence[int], batch_sizes: Sequence[int]) -> None:
        return None

    def close(self) -> None:
        self._scratch.clear()

    def stats(self) -> dict:
        shapes = {}
        for key, n in sorted(self._calls.items()):
            shapes[f"b{key[0]}_t{key[1]}"] = {
                "calls": n,
                "mean_ms": round(self._ns[key] / n * 1000, 3) if n else 0.0,
            }
        return {
            "backend": self.name,
            "scratch_mib": round(
                sum(b.numel() * b.element_size() for b in self._scratch.values()) / 2**20, 1
            ),
            "shapes": shapes,
        }
