"""Pipeline position: FLOW BACKEND (TensorRT) — in-process, zero-copy flow step.

Role in pipeline:
  Drop-in replacement for `torch_backend.TorchFmBackend` that runs the flow
  decoder as a TensorRT engine bound directly to the scheduler's arena tensors.

Engine compatibility
--------------------
The engine contract here is deliberately IDENTICAL to the one upstream ZipVoice
produces (`python -m zipvoice.bin.tensorrt_export`, see
`runtime/nvidia_triton/` in the k2-fsa/ZipVoice repo):

    input   x             (N, T, 300)  fp16   [noisy | text_cond | speech_cond]
    input   t             (N,)         fp16   per-sample timestep
    input   padding_mask  (N, T)       fp16   1.0 at padded positions
    output  v             (N, T, 100)  fp16   predicted velocity

    optional input  guidance_scale (N,) fp16  -- ZipVoice-Distill only, absent
                                                 for DhVaani-0.5

So an engine built by upstream's script works here unchanged, and an engine
built by `flowtts.dhvaani.setup.build_trt` works under upstream's Triton model.
That interop is worth more than a marginally tidier interface.

Differences from upstream's `TrtContextWrapper`
-----------------------------------------------
Upstream calls `torch.cuda.current_stream().synchronize()` twice per invocation
(once before switching to its private stream, once after execution). At 8-16 ODE
steps per span that is 16-32 full device synchronisations per utterance, which
serialises the GPU against the host and destroys throughput under concurrency.

This backend executes on the *current* stream instead, so the flow step queues
behind whatever the scheduler already enqueued and never blocks the host. The
only ordering requirement is that the arena writes precede the execution, which
same-stream ordering guarantees for free.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import structlog
import torch

from flowtts.dhvaani.config import N_MELS, dhv_settings
from flowtts.dhvaani.backends.base import BaseFmBackend
from flowtts.dhvaani.types import DhvaaniError

logger = structlog.get_logger(__name__)

_IN_X, _IN_T, _IN_MASK, _IN_GS = "x", "t", "padding_mask", "guidance_scale"


class TrtUnavailable(DhvaaniError):
    status_code = 503
    code = "tensorrt_unavailable"


class _Engine:
    """One deserialized plan plus its execution context and shape profile."""

    def __init__(self, path: Path, trt, device):
        self.path = path
        self.trt = trt
        logger_trt = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger_trt).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise TrtUnavailable(f"failed to deserialize TensorRT engine at {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise TrtUnavailable(
                f"could not create an execution context for {path} -- "
                "usually not enough free VRAM. Lower engine.max_batch_size or "
                "memory.arena_vram_fraction."
            )

        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [
            n for n in self.names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self.outputs = [n for n in self.names if n not in self.inputs]
        self.has_gs = _IN_GS in self.inputs
        self.out_name = self.outputs[-1]

        # Shape bounds come from the engine itself, so an engine built by
        # upstream's script (or by anyone else) is described correctly without a
        # sidecar file. The .json emitted by our build script is informational.
        lo = self.engine.get_tensor_profile_shape(_IN_X, 0)[0]
        hi = self.engine.get_tensor_profile_shape(_IN_X, 0)[2]
        self.min_batch, self.min_frames = int(lo[0]), int(lo[1])
        self.max_batch, self.max_frames = int(hi[0]), int(hi[1])

        meta = path.with_suffix(".json")
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}

    def covers(self, batch: int, frames: int) -> bool:
        return (
            self.min_batch <= batch <= self.max_batch
            and self.min_frames <= frames <= self.max_frames
        )


class TrtFmBackend(BaseFmBackend):
    name = "trt"

    def __init__(self, loaded, settings=None):
        super().__init__(loaded, settings)
        try:
            import tensorrt as trt  # noqa: F401
        except ImportError as e:
            raise TrtUnavailable(
                "the `tensorrt` package is not installed. Either\n"
                "  pip install tensorrt-cu12\n"
                "and build engines with\n"
                "  python -m flowtts.dhvaani.setup.build_trt\n"
                "or run with DHVAANI_BACKEND__KIND=torch."
            ) from e
        self._trt = trt

        engine_dir = Path(self._s.backend.trt_engine_dir).expanduser()
        plans = sorted(engine_dir.glob("*.plan")) if engine_dir.is_dir() else []
        if not plans and self._s.backend.trt_build_on_missing:
            from flowtts.dhvaani.setup.build_trt import build_engines

            build_engines(self._s)
            plans = sorted(engine_dir.glob("*.plan"))
        if not plans:
            raise TrtUnavailable(
                f"no TensorRT engines in {engine_dir}. Build them with:\n"
                "  python -m flowtts.dhvaani.setup.build_trt\n"
                "(engines produced by upstream's zipvoice.bin.tensorrt_export "
                "are also accepted)"
            )

        self._engines = [_Engine(p, trt, self.device) for p in plans]
        # Prefer the tightest profile that covers a shape: a narrow engine is
        # specialised for its range and usually faster than the catch-all.
        self._engines.sort(key=lambda e: (e.max_batch, e.max_frames))
        self._out: dict[tuple[int, int], torch.Tensor] = {}
        self._unsupported: set[tuple[int, int]] = set()

        logger.info(
            "trt_engines_loaded",
            dir=str(engine_dir),
            engines=[
                {
                    "file": e.path.name,
                    "batch": [e.min_batch, e.max_batch],
                    "frames": [e.min_frames, e.max_frames],
                    "guidance_scale_input": e.has_gs,
                }
                for e in self._engines
            ],
        )

    # -- shape routing -------------------------------------------------------
    def _pick(self, batch: int, frames: int) -> _Engine | None:
        for e in self._engines:
            if e.covers(batch, frames):
                return e
        return None

    def supports_bucket(self, batch: int, frames: int) -> bool:
        ok = self._pick(batch, frames) is not None
        if not ok and (batch, frames) not in self._unsupported:
            self._unsupported.add((batch, frames))
            logger.info("trt_shape_uncovered", batch=batch, frames=frames)
        return ok

    def _out_buffer(self, B: int, T: int) -> torch.Tensor:
        key = (B, T)
        buf = self._out.get(key)
        if buf is None:
            buf = torch.empty(
                (B, T, N_MELS), device=self.device, dtype=torch.float16
            )
            self._out[key] = buf
        return buf

    # -- execution -----------------------------------------------------------
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
        eng = self._pick(B, T)
        if eng is None:
            raise TrtUnavailable(f"no TensorRT profile covers (batch={B}, frames={T})")

        x_cat = self._concat(x, text_condition, speech_condition).to(torch.float16).contiguous()
        t_h = t.to(torch.float16).contiguous()
        # The engine declares the mask as fp16 (TensorRT has no bool IO here),
        # matching upstream's exporter.
        mask_h = padding_mask.to(torch.float16).contiguous()
        out = self._out_buffer(B, T)

        ctx = eng.context
        ctx.set_input_shape(_IN_X, (B, T, N_MELS * 3))
        ctx.set_input_shape(_IN_T, (B,))
        ctx.set_input_shape(_IN_MASK, (B, T))
        ctx.set_tensor_address(_IN_X, x_cat.data_ptr())
        ctx.set_tensor_address(_IN_T, t_h.data_ptr())
        ctx.set_tensor_address(_IN_MASK, mask_h.data_ptr())

        gs_h = None
        if eng.has_gs:
            # Only ZipVoice-Distill engines take this. DhVaani-0.5 applies CFG by
            # doubling the batch (see ops.cfg_expand), so a neutral 0 is correct.
            gs_h = torch.zeros(B, device=self.device, dtype=torch.float16)
            ctx.set_input_shape(_IN_GS, (B,))
            ctx.set_tensor_address(_IN_GS, gs_h.data_ptr())

        ctx.set_tensor_address(eng.out_name, out.data_ptr())

        stream = torch.cuda.current_stream(self.device)
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise TrtUnavailable("TensorRT execute_async_v3 returned false")
        # Deliberately NO synchronize: staying on the current stream keeps
        # ordering correct and lets the host run ahead.

        self._record(B, T, t0)
        return out.to(self.dtype) if self.dtype != torch.float16 else out

    def warmup(self, buckets: Sequence[int], batch_sizes: Sequence[int]) -> None:
        if self.device.type != "cuda":
            return
        for T in buckets:
            for B in batch_sizes:
                if not self.supports_bucket(B, T):
                    continue
                x = torch.zeros((B, T, N_MELS), device=self.device, dtype=self.dtype)
                t = torch.full((B,), 0.5, device=self.device, dtype=torch.float32)
                m = torch.zeros((B, T), device=self.device, dtype=torch.bool)
                try:
                    self.fm_step(x, x, x, t, m)
                except Exception as e:
                    logger.warning("trt_warmup_failed", batch=B, frames=T, error=str(e)[:200])
        torch.cuda.synchronize(self.device)

    def close(self) -> None:
        self._out.clear()
        self._engines.clear()
        super().close()

    def stats(self) -> dict:
        d = super().stats()
        d["engines"] = [
            {"file": e.path.name, "max_batch": e.max_batch, "max_frames": e.max_frames}
            for e in self._engines
        ]
        d["out_buffers_mib"] = round(
            sum(b.numel() * b.element_size() for b in self._out.values()) / 2**20, 1
        )
        return d
