"""Pipeline position: BACKEND FACTORY — choose the flow-step executor.

`dhv_settings.backend.kind` selects between:
  torch   always available, the correctness reference, CUDA graphs optional
  trt     in-process TensorRT, lowest latency, needs prebuilt engines
  triton  NVIDIA Triton Inference Server client (see triton_backend's docstring
          for when that is and is not the right choice)

A requested backend that cannot start degrades to `torch` with a loud warning
rather than preventing the server from booting -- a TTS gateway that refuses to
start because an engine file is missing is worse than one running a few
milliseconds slower.
"""

from __future__ import annotations

import structlog

from flowtts.dhvaani.config import dhv_settings
from flowtts.dhvaani.backends.base import BaseFmBackend
from flowtts.dhvaani.backends.torch_backend import TorchFmBackend

logger = structlog.get_logger(__name__)

__all__ = ["build_backend", "BaseFmBackend", "TorchFmBackend"]


def build_backend(loaded, settings=None, kind: str | None = None) -> BaseFmBackend:
    s = settings or dhv_settings
    kind = kind or s.backend.kind

    if kind == "torch":
        return TorchFmBackend(loaded, s)

    if kind == "trt":
        try:
            from flowtts.dhvaani.backends.trt_backend import TrtFmBackend

            return TrtFmBackend(loaded, s)
        except Exception as e:
            logger.warning("trt_backend_unavailable_falling_back", error=str(e))
            return TorchFmBackend(loaded, s)

    if kind == "triton":
        try:
            from flowtts.dhvaani.backends.triton_backend import TritonFmBackend

            return TritonFmBackend(loaded, s)
        except Exception as e:
            logger.warning("triton_backend_unavailable_falling_back", error=str(e))
            return TorchFmBackend(loaded, s)

    raise ValueError(f"unknown backend kind {kind!r}; choose torch, trt or triton")
