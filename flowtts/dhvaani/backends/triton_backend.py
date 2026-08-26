"""Pipeline position: FLOW BACKEND (NVIDIA Triton Inference Server).

Role in pipeline:
  Executes the flow step on a Triton Inference Server instead of in-process,
  so the GPU can be shared with other models and managed by Triton's own
  scheduler, metrics and model-control plane.

Honest tradeoff -- read before choosing this
--------------------------------------------
This backend puts a serialisation + IPC hop on EVERY Euler step. A span at
num_step=8 with CFG makes 8 round trips; at 200 RPS with ~2 spans per request
that is ~3200 round trips per second. Even over gRPC on loopback with CUDA
shared memory that is milliseconds of pure overhead per span, and it is
overhead the in-process `trt` backend does not pay at all.

So:
  * Want the lowest TTFB on a dedicated box?           -> backend.kind = "trt"
  * Sharing the GPU with ASR/LLM/other models, or need
    Triton's model repository, versioning, and metrics? -> deploy the whole
    pipeline as a Triton model instead (see `triton/model_repository/`, generated
    by `flowtts.dhvaani.setup.build_triton_repo`), which is what upstream
    ZipVoice's `runtime/nvidia_triton` does. That keeps the ODE loop server-side
    and pays the IPC cost once per request rather than once per step.

This backend exists for the middle case: an existing Triton fleet that already
hosts the fm_decoder engine and that you want FlowTTS to call into.

CUDA shared memory
------------------
Registering a shm region is expensive and must NOT happen per call -- that is
the classic Triton performance trap. Regions are cached by (name, shape) and
reused for the process lifetime.
"""

from __future__ import annotations

import time
from typing import Sequence

import structlog
import torch

from flowtts.dhvaani.config import N_MELS, dhv_settings
from flowtts.dhvaani.backends.base import BaseFmBackend
from flowtts.dhvaani.types import DhvaaniError

logger = structlog.get_logger(__name__)


class TritonUnavailable(DhvaaniError):
    status_code = 503
    code = "triton_unavailable"


class TritonFmBackend(BaseFmBackend):
    name = "triton"

    def __init__(self, loaded, settings=None):
        super().__init__(loaded, settings)
        b = self._s.backend
        self._model = b.triton_model_fm_step
        self._use_shm = b.triton_use_cuda_shm and self.device.type == "cuda"

        try:
            if b.triton_protocol == "grpc":
                import tritonclient.grpc as tc
            else:
                import tritonclient.http as tc
        except ImportError as e:
            raise TritonUnavailable(
                "tritonclient is not installed: pip install 'tritonclient[all]'\n"
                "Or run with DHVAANI_BACKEND__KIND=torch."
            ) from e
        self._tc = tc

        try:
            self._client = tc.InferenceServerClient(url=b.triton_url, verbose=False)
            if not self._client.is_server_live():
                raise TritonUnavailable(f"Triton at {b.triton_url} is not live")
            if not self._client.is_model_ready(self._model):
                raise TritonUnavailable(
                    f"model {self._model!r} is not READY on {b.triton_url}. "
                    "Generate a repository with "
                    "`python -m flowtts.dhvaani.setup.build_triton_repo` and start "
                    "tritonserver against it."
                )
        except TritonUnavailable:
            raise
        except Exception as e:
            # Fail at construction, never per request -- a backend that only
            # discovers it cannot work under load is far worse than one that
            # refuses to boot.
            raise TritonUnavailable(
                f"could not reach Triton at {b.triton_url}: {e}"
            ) from e

        self._shm: dict[str, object] = {}
        logger.info(
            "triton_backend_ready",
            url=b.triton_url,
            protocol=b.triton_protocol,
            model=self._model,
            cuda_shm=self._use_shm,
        )

    # -- shared memory -------------------------------------------------------
    def _shm_region(self, key: str, nbytes: int):
        """Register (once) and return a CUDA shared-memory handle for `key`."""
        import tritonclient.utils.cuda_shared_memory as cshm

        entry = self._shm.get(key)
        if entry is not None and entry[1] >= nbytes:  # type: ignore[index]
            return entry[0]  # type: ignore[index]
        if entry is not None:
            self._client.unregister_cuda_shared_memory(key)
            cshm.destroy_shared_memory_region(entry[0])  # type: ignore[index]
        handle = cshm.create_shared_memory_region(key, nbytes, 0)
        self._client.register_cuda_shared_memory(
            key, cshm.get_raw_handle(handle), 0, nbytes
        )
        self._shm[key] = (handle, nbytes)
        return handle

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
        tc = self._tc

        x_cat = self._concat(x, text_condition, speech_condition).to(torch.float16).contiguous()
        t_h = t.to(torch.float16).contiguous()
        mask_h = padding_mask.to(torch.float16).contiguous()

        if self._use_shm:
            out = self._infer_shm(x_cat, t_h, mask_h, B, T)
        else:
            out = self._infer_numpy(x_cat, t_h, mask_h, B, T)

        self._record(B, T, t0)
        return out.to(self.dtype)

    def _infer_numpy(self, x_cat, t_h, mask_h, B, T):
        """Fallback path: tensors cross the process boundary as numpy.

        This copies D2H then H2D for every input on every step. It works
        everywhere but is roughly an order of magnitude more overhead than the
        shared-memory path; it exists so the backend is usable against a remote
        Triton where shm is impossible.
        """
        tc = self._tc
        inputs = [
            tc.InferInput("x", [B, T, N_MELS * 3], "FP16"),
            tc.InferInput("t", [B], "FP16"),
            tc.InferInput("padding_mask", [B, T], "FP16"),
        ]
        inputs[0].set_data_from_numpy(x_cat.cpu().numpy())
        inputs[1].set_data_from_numpy(t_h.cpu().numpy())
        inputs[2].set_data_from_numpy(mask_h.cpu().numpy())
        res = self._client.infer(
            self._model,
            inputs,
            outputs=[tc.InferRequestedOutput("v")],
            client_timeout=self._s.backend.triton_client_timeout_s,
        )
        arr = res.as_numpy("v")
        return torch.from_numpy(arr).to(self.device)

    def _infer_shm(self, x_cat, t_h, mask_h, B, T):
        import tritonclient.utils.cuda_shared_memory as cshm

        tc = self._tc
        specs = [
            ("x", x_cat, [B, T, N_MELS * 3]),
            ("t", t_h, [B]),
            ("padding_mask", mask_h, [B, T]),
        ]
        inputs = []
        for name, tensor, dims in specs:
            nbytes = tensor.numel() * tensor.element_size()
            key = f"dhv_{name}_{B}x{T}"
            handle = self._shm_region(key, nbytes)
            cshm.set_shared_memory_region_from_dlpack(handle, [tensor])
            inp = tc.InferInput(name, dims, "FP16")
            inp.set_shared_memory(key, nbytes)
            inputs.append(inp)

        out_bytes = B * T * N_MELS * 2
        out_key = f"dhv_v_{B}x{T}"
        out_handle = self._shm_region(out_key, out_bytes)
        req_out = tc.InferRequestedOutput("v")
        req_out.set_shared_memory(out_key, out_bytes)

        res = self._client.infer(
            self._model,
            inputs,
            outputs=[req_out],
            client_timeout=self._s.backend.triton_client_timeout_s,
        )
        arr = cshm.get_contents_as_numpy(
            out_handle, __import__("numpy").float16, [B, T, N_MELS]
        )
        return torch.as_tensor(arr, device=self.device)

    def warmup(self, buckets: Sequence[int], batch_sizes: Sequence[int]) -> None:
        if self.device.type != "cuda":
            return
        for T in buckets:
            for B in batch_sizes:
                x = torch.zeros((B, T, N_MELS), device=self.device, dtype=self.dtype)
                t = torch.full((B,), 0.5, device=self.device, dtype=torch.float32)
                m = torch.zeros((B, T), device=self.device, dtype=torch.bool)
                try:
                    self.fm_step(x, x, x, t, m)
                except Exception as e:
                    logger.warning("triton_warmup_failed", batch=B, frames=T, error=str(e)[:200])

    def close(self) -> None:
        try:
            import tritonclient.utils.cuda_shared_memory as cshm

            for key, (handle, _n) in list(self._shm.items()):
                try:
                    self._client.unregister_cuda_shared_memory(key)
                    cshm.destroy_shared_memory_region(handle)
                except Exception:
                    pass
        except Exception:
            pass
        self._shm.clear()
        super().close()
