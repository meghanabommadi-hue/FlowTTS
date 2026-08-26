"""Pipeline position: FLOW BACKEND (PyTorch) — the reference executor.

Role in pipeline:
  Runs `ZipVoice.fm_decoder` directly. Always available, always correct: the TRT
  and Triton backends are validated against this one, and the scheduler falls
  back to it for any shape the others cannot serve.

CUDA graphs
-----------
The Zipformer has no data-dependent control flow at inference -- every
stochastic branch is gated on `self.training` or `torch.jit.is_*`, so in eval
mode the graph is fixed for a given (batch, frames). That makes it a good CUDA
graph candidate: a captured replay removes ~500 kernel launches per step, which
at 8-16 steps per span and hundreds of spans per second is a substantial slice
of CPU time.

Capture requires static input/output buffers, so each shape costs its own set.
We bound the number of graphs and evict least-recently-used ones, because the
memory pool a graph reserves is not returned to the general allocator.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Sequence

import structlog
import torch

from flowtts.dhvaani.config import N_MELS, dhv_settings
from flowtts.dhvaani.backends.base import BaseFmBackend

logger = structlog.get_logger(__name__)

# A captured graph holds its own static buffers plus a private memory pool.
_MAX_GRAPHS = 48

# Graphs are keyed by (batch, frames). Frames are already bucketed, but the
# batch size varies continuously with load -- capturing a graph per distinct
# batch would thrash the cache and spend all its time recapturing. Rounding the
# batch up to this ladder bounds the graph count to len(ladder) x n_buckets and
# makes replays hit almost always. The padded rows are duplicates of a real row
# (never all-masked, which would produce NaN in the attention softmax) and are
# sliced off the output.
_BATCH_LADDER = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)

# Capture graphs only in the launch-bound regime. Each captured graph reserves a
# PRIVATE memory pool sized for that shape's peak activations, and the Zipformer's
# attention scores are O(batch * frames^2) -- at batch 64 x 896 frames a single
# pool runs to roughly a gigabyte, so capturing the whole ladder reserved ~28 GB
# on an L40S and starved everything else on the card.
#
# Above this batch the forward is compute-bound anyway: measured on an L40S, a
# graph replay saves ~14 ms at batch 1 (7.7 vs 21.5 ms) but well under 5% at
# batch 32. So the memory buys nothing there.
_GRAPH_MAX_BATCH = 16
_GRAPH_MAX_FRAME_ROWS = 16384   # batch * frames ceiling for a captured shape


def _pad_batch_to(n: int) -> int:
    for b in _BATCH_LADDER:
        if n <= b:
            return b
    return n


class _Graph:
    __slots__ = ("graph", "x_cat", "t", "mask", "out")

    def __init__(self, graph, x_cat, t, mask, out):
        self.graph = graph
        self.x_cat = x_cat
        self.t = t
        self.mask = mask
        self.out = out


class TorchFmBackend(BaseFmBackend):
    name = "torch"

    def __init__(self, loaded, settings=None):
        super().__init__(loaded, settings)
        self._fm = loaded.zipvoice.fm_decoder
        self._graphs: OrderedDict[tuple[int, int], _Graph] = OrderedDict()
        self._autocast = self.device.type == "cuda" and self.dtype != torch.float32
        self._use_graphs = (
            self._s.backend.use_cuda_graphs and self.device.type == "cuda"
        )
        self._compiled = None

        if self._s.backend.use_torch_compile:
            try:
                self._compiled = torch.compile(
                    self._fm, mode=self._s.backend.compile_mode, dynamic=True
                )
                logger.info("torch_compile_enabled", mode=self._s.backend.compile_mode)
            except Exception as e:
                logger.warning("torch_compile_failed", error=str(e))
                self._compiled = None

    # -- core ----------------------------------------------------------------
    def _run(self, x_cat: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        fn = self._compiled or self._fm
        if self._autocast:
            # The Zipformer's `timestep_embedding` hard-casts to fp32 (it calls
            # `.float()` on the timesteps), so its output always hits the fp16
            # `time_embed` Linear as Float and raises
            #     "mat1 and mat2 must have the same dtype".
            # autocast fixes that generically -- it casts the inputs of eligible
            # ops rather than us patching vendored model code. cache_enabled is
            # off because a cached weight cast is not safe to capture in a CUDA
            # graph.
            with torch.autocast("cuda", dtype=self.dtype, cache_enabled=False):
                return fn(x=x_cat, t=t, padding_mask=mask)
        return fn(x=x_cat, t=t, padding_mask=mask)

    @torch.inference_mode()
    def fm_step(
        self,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        t: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        t0 = time.perf_counter()
        B, T, _ = x.shape
        x_cat = self._concat(x, text_condition, speech_condition)
        # t must stay (B,) -- the batch is heterogeneous in timestep by design.
        t = t.to(x_cat.dtype).view(B)

        if self._use_graphs and self._graphable(B, T):
            pad_b = _pad_batch_to(B)
            g = self._graphs.get((pad_b, T))
            if g is None and pad_b == B:
                # Capture lazily for a shape warmup did not cover, so the first
                # request at an unseen bucket pays once instead of every time.
                try:
                    self._capture(pad_b, T)
                    g = self._graphs.get((pad_b, T))
                except Exception as e:
                    logger.warning(
                        "cuda_graph_capture_failed_disabling",
                        batch=pad_b, frames=T, error=str(e)[:200],
                    )
                    self._use_graphs = False
                    self._graphs.clear()
            if g is not None:
                self._graphs.move_to_end((pad_b, T))
                if pad_b == B:
                    g.x_cat.copy_(x_cat)
                    g.t.copy_(t)
                    g.mask.copy_(padding_mask)
                else:
                    # Fill the padding rows with a copy of the last real row.
                    g.x_cat[:B].copy_(x_cat)
                    g.x_cat[B:].copy_(x_cat[-1:].expand(pad_b - B, -1, -1))
                    g.t[:B].copy_(t)
                    g.t[B:].fill_(float(t[-1]) if B else 0.5)
                    g.mask[:B].copy_(padding_mask)
                    g.mask[B:].copy_(padding_mask[-1:].expand(pad_b - B, -1))
                g.graph.replay()
                self._record(B, T, t0)
                return g.out[:B]

        out = self._run(x_cat, t, padding_mask)
        self._record(B, T, t0)
        return out

    # -- graph capture -------------------------------------------------------
    @staticmethod
    def _graphable(batch: int, frames: int) -> bool:
        return (
            _pad_batch_to(batch) <= _GRAPH_MAX_BATCH
            and _pad_batch_to(batch) * frames <= _GRAPH_MAX_FRAME_ROWS
        )

    def _capture(self, B: int, T: int) -> None:
        if (B, T) in self._graphs:
            return
        assert not self._fm.training, "CUDA graph capture requires eval mode"

        x_cat = torch.zeros((B, T, N_MELS * 3), device=self.device, dtype=self.dtype)
        t = torch.full((B,), 0.5, device=self.device, dtype=self.dtype)
        mask = torch.zeros((B, T), device=self.device, dtype=torch.bool)

        # Autotuning (cuDNN/cuBLAS algorithm selection) must happen OUTSIDE the
        # capture, otherwise the graph records the benchmarking passes.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                self._run(x_cat, t, mask)
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph):
                out = self._run(x_cat, t, mask)

        while len(self._graphs) >= _MAX_GRAPHS:
            old_key, _ = self._graphs.popitem(last=False)
            logger.info("cuda_graph_evicted", shape=old_key)
        self._graphs[(B, T)] = _Graph(graph, x_cat, t, mask, out)
        logger.info("cuda_graph_captured", batch=B, frames=T, n_graphs=len(self._graphs))

    def warmup(self, buckets: Sequence[int], batch_sizes: Sequence[int]) -> None:
        if self.device.type != "cuda":
            return
        # Only ladder sizes are ever captured, so warming anything else just
        # wastes startup time.
        batch_sizes = sorted({_pad_batch_to(b) for b in batch_sizes})
        for T in buckets:
            for B in batch_sizes:
                x = torch.zeros((B, T, N_MELS), device=self.device, dtype=self.dtype)
                t = torch.full((B,), 0.5, device=self.device, dtype=torch.float32)
                m = torch.zeros((B, T), device=self.device, dtype=torch.bool)
                try:
                    self.fm_step(x, x, x, t, m)
                except Exception as e:
                    logger.warning(
                        "fm_warmup_failed", frames=T, batch=B, error=str(e)[:200]
                    )
                    continue
                if not self._use_graphs or not self._graphable(B, T):
                    continue
                try:
                    self._capture(B, T)
                except Exception as e:
                    # A half-finished capture can leave the CUDA context and the
                    # RNG state unusable for everything else in the process
                    # ("Offset increment outside graph capture"), so treat any
                    # capture failure as fatal to the feature, not to the server.
                    logger.warning(
                        "cuda_graph_capture_failed_disabling",
                        frames=T, batch=B, error=str(e)[:200],
                    )
                    self._use_graphs = False
                    self._graphs.clear()
                    try:
                        torch.cuda.synchronize(self.device)
                    except Exception:
                        pass
                    break
        torch.cuda.synchronize(self.device)

    def close(self) -> None:
        self._graphs.clear()
        super().close()

    def stats(self) -> dict:
        d = super().stats()
        d["cuda_graphs"] = len(self._graphs)
        d["torch_compile"] = self._compiled is not None
        return d
