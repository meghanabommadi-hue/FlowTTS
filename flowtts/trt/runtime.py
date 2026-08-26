"""Pipeline position: ACCELERATION — engine runtimes behind one contract.

Role in pipeline:
  Three interchangeable implementations of the same call, chosen by config and
  swapped into ``model.llm.forward`` by :mod:`flowtts.trt.patcher`:

      backbone(hidden_states[B,S,H], input_lengths[B]) -> [B,S,H]

  ``TRTBackbone``     — a TensorRT engine built by :mod:`flowtts.trt.build_trt`.
  ``TRTLLMBackbone``  — a TensorRT-LLM engine built by :mod:`flowtts.trt.build_trtllm`,
                        run through ``tensorrt_llm.runtime.Session`` exactly as
                        github.com/tlitech/omnivoice-trtllm does.
  ``TorchBackbone``   — the PyTorch mirror, optionally torch.compile'd with CUDA
                        graphs. No engine build, no extra dependency, and the
                        reference every other backend is validated against.

  All three own their RoPE tables and produce them the same way, so switching
  backends cannot change the audio for reasons other than arithmetic precision.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Optional

import torch

from flowtts.trt.backbone import BackboneConfig, Qwen3Backbone, precompute_rope
from flowtts.trt.build_trt import ENGINE_NAME, INPUT_NAMES, MANIFEST_NAME, OUTPUT_NAME

logger = logging.getLogger(__name__)


def _cuda_stream_guard(func):
    """Run on the backend's own stream, synchronizing with the caller's.

    Ported from upstream's ``cuda_stream_guard``. OmniVoice's generation loop
    runs on the default stream; the engine runs on its own so its work can be
    captured or overlapped, and the two are joined at the boundary.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        external = torch.cuda.current_stream()
        if external != self.stream:
            external.synchronize()
            torch.cuda.set_stream(self.stream)
        try:
            return func(self, *args, **kwargs)
        finally:
            if external != self.stream:
                self.stream.synchronize()
                torch.cuda.set_stream(external)

    return wrapper


class _RopeMixin:
    """Shared RoPE table handling: precompute once, slice and broadcast per call."""

    def _init_rope(self, cfg: BackboneConfig, device, dtype, max_seq_len: int) -> None:
        self.cfg = cfg
        self.device = device
        self.rope_dtype = dtype
        self.max_seq_len = max_seq_len
        self.rope_cos, self.rope_sin = precompute_rope(cfg, device, dtype, max_seq_len)

    def _rope_for(self, batch: int, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq > self.max_seq_len:
            # Grow rather than fail: a long reference clip plus long text can
            # exceed the table built at startup, and truncating it would corrupt
            # positions instead of just costing a little time here.
            self.max_seq_len = max(seq, self.max_seq_len * 2)
            self.rope_cos, self.rope_sin = precompute_rope(
                self.cfg, self.device, self.rope_dtype, self.max_seq_len
            )
        cos = self.rope_cos[:seq].unsqueeze(0).expand(batch, -1, -1).contiguous()
        sin = self.rope_sin[:seq].unsqueeze(0).expand(batch, -1, -1).contiguous()
        return cos, sin


class TorchBackbone(_RopeMixin):
    """PyTorch backbone, optionally compiled with CUDA graphs.

    ``torch.compile(mode="reduce-overhead")`` captures a CUDA graph per input
    shape. That matters here far more than for a normal LLM: the backbone runs
    ``num_step`` times per utterance at *identical* shapes, and a 28-layer model
    is ~200 kernel launches per pass, so launch overhead is a real fraction of
    a short-chunk generation.
    """

    kind = "torch"

    def __init__(
        self,
        llm: torch.nn.Module,
        *,
        compile_model: bool = False,
        compile_mode: str = "reduce-overhead",
        max_seq_len: int = 4096,
    ) -> None:
        cfg = BackboneConfig.from_hf(llm.config)
        self.module = Qwen3Backbone.from_llm(llm, cfg)
        param = next(self.module.parameters())
        self._init_rope(cfg, param.device, param.dtype, max_seq_len)
        self.stream = torch.cuda.current_stream()
        self.compiled = False
        if compile_model:
            try:
                self.module = torch.compile(self.module, mode=compile_mode, dynamic=True)
                self.compiled = True
            except Exception as exc:  # noqa: BLE001 — compilation is best-effort
                logger.warning("torch.compile failed, running eager: %s", exc)

    @torch.no_grad()
    def __call__(self, hidden_states: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        cos, sin = self._rope_for(hidden_states.shape[0], hidden_states.shape[1])
        return self.module(hidden_states.to(self.rope_dtype), cos, sin, input_lengths)

    def info(self) -> dict:
        return {"kind": self.kind, "compiled": self.compiled,
                "dtype": str(self.rope_dtype).replace("torch.", "")}


class TRTBackbone(_RopeMixin):
    """A TensorRT engine for the Qwen3 backbone (TensorRT 10 runtime API)."""

    kind = "tensorrt"

    def __init__(
        self,
        engine_dir: str | Path,
        cfg: BackboneConfig,
        *,
        device: torch.device | str = "cuda",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        import tensorrt as trt

        self._trt = trt
        engine_dir = Path(engine_dir)
        engine_path = engine_dir / ENGINE_NAME
        if not engine_path.exists():
            raise FileNotFoundError(
                f"No TensorRT engine at {engine_path} — build it with "
                f"`python -m flowtts.trt.build_trt`"
            )

        manifest_path = engine_dir / MANIFEST_NAME
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

        self.device = torch.device(device)
        self.stream = stream or torch.cuda.Stream(self.device)

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine at {engine_path}")
        self.context = self.engine.create_execution_context()

        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        expected = set(INPUT_NAMES) | {OUTPUT_NAME}
        if names != expected:
            raise RuntimeError(f"engine tensor mismatch — expected {expected}, found {names}")

        self._dtypes = {
            n: torch.from_numpy(
                __import__("numpy").empty(0, dtype=trt.nptype(self.engine.get_tensor_dtype(n)))
            ).dtype
            for n in names
        }
        self._input_dtype = self._dtypes["hidden_states"]
        self._output = None   # reused across calls when the shape is unchanged

        self._init_rope(cfg, self.device, self._input_dtype,
                        int(self.manifest.get("max_seq", 2048)))

    def _output_buffer(self, batch: int, seq: int) -> torch.Tensor:
        shape = (batch, seq, self.cfg.hidden_size)
        if self._output is None or tuple(self._output.shape) != shape:
            self._output = torch.empty(shape, dtype=self._dtypes[OUTPUT_NAME],
                                       device=self.device)
        return self._output

    @torch.no_grad()
    @_cuda_stream_guard
    def __call__(self, hidden_states: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        batch, seq = hidden_states.shape[0], hidden_states.shape[1]
        cos, sin = self._rope_for(batch, seq)

        tensors = {
            "hidden_states": hidden_states.to(self._input_dtype).contiguous(),
            "rope_cos": cos.to(self._input_dtype),
            "rope_sin": sin.to(self._input_dtype),
            "input_lengths": input_lengths.to(torch.int32).contiguous(),
        }
        for name, tensor in tensors.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        out = self._output_buffer(batch, seq)
        self.context.set_tensor_address(OUTPUT_NAME, out.data_ptr())

        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        self.stream.synchronize()
        # Cloned because the buffer is reused by the next call, and the caller
        # holds this tensor across the rest of the denoise step.
        return out.clone()

    def info(self) -> dict:
        return {"kind": self.kind, **{k: self.manifest.get(k) for k in
                                      ("precision", "max_batch", "max_seq",
                                       "tensorrt_version", "engine_bytes")}}


class TRTLLMBackbone(_RopeMixin):
    """A TensorRT-LLM engine, run through ``tensorrt_llm.runtime.Session``.

    This is the upstream path from github.com/tlitech/omnivoice-trtllm
    (``model_repo_omnivoice/omnivoice/1/omnivoice_trtllm.py``), kept intact so a
    box that does have TensorRT-LLM installed runs exactly what upstream runs.
    """

    kind = "trtllm"

    def __init__(
        self,
        engine_dir: str | Path,
        cfg: BackboneConfig,
        *,
        device: torch.device | str = "cuda",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        from tensorrt_llm._utils import str_dtype_to_torch
        from tensorrt_llm.runtime.session import Session

        engine_dir = Path(engine_dir)
        config = json.loads((engine_dir / "config.json").read_text())
        self.dtype = config["pretrained_config"]["dtype"]
        self._input_dtype = str_dtype_to_torch(self.dtype)

        self.device = torch.device(device)
        self.stream = stream or torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        self.session = Session.from_serialized_engine(
            (engine_dir / "rank0.engine").read_bytes()
        )
        names = {self.session.engine.get_tensor_name(i)
                 for i in range(self.session.engine.num_io_tensors)}
        expected = set(INPUT_NAMES) | {OUTPUT_NAME}
        if names != expected:
            raise RuntimeError(f"engine tensor mismatch — expected {expected}, found {names}")

        self._outputs: dict[str, torch.Tensor] = {}
        self._init_rope(cfg, self.device, self._input_dtype, 4096)

    def _setup_outputs(self, batch: int, seq: int) -> None:
        import tensorrt as trt
        from tensorrt_llm._utils import trt_dtype_to_torch

        engine = self.session.engine
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = list(engine.get_tensor_shape(name))
            shape[0], shape[1] = batch, seq
            dtype = trt_dtype_to_torch(engine.get_tensor_dtype(name))
            existing = self._outputs.get(name)
            if existing is None or list(existing.shape) != shape:
                self._outputs[name] = torch.empty(shape, dtype=dtype, device=self.device)

    @torch.no_grad()
    @_cuda_stream_guard
    def __call__(self, hidden_states: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        batch, seq = hidden_states.shape[0], hidden_states.shape[1]
        self._setup_outputs(batch, seq)
        cos, sin = self._rope_for(batch, seq)

        inputs = {
            "hidden_states": hidden_states.to(self._input_dtype).contiguous(),
            "rope_cos": cos.to(self._input_dtype),
            "rope_sin": sin.to(self._input_dtype),
            "input_lengths": input_lengths.to(torch.int32).contiguous(),
        }
        self.session.set_shapes(inputs)
        if not self.session.run(inputs, self._outputs, self.stream.cuda_stream):
            raise RuntimeError("TRT-LLM engine execution failed")
        return self._outputs[OUTPUT_NAME].clone()

    def info(self) -> dict:
        return {"kind": self.kind, "dtype": self.dtype}
