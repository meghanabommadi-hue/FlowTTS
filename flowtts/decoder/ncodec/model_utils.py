import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file

from flowtts.decoder.ncodec.layers import (
    Snake1d,
    WNConv1d,
    ResidualUnit,
    WNConvTranspose1d,
    init_weights,
)

def remove_weight_norm_recursive(m):
    """
    Recursively removes weight normalization from a module.
    """
    try:
        if hasattr(m, 'weight_g') and hasattr(m, 'weight_v'):
            nn.utils.remove_weight_norm(m)
    except Exception as e:
        print(f"Could not remove weight norm from {m}: {e}")


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        kernel_size: int = 2,
        stride: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        kernel_sizes,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, (kernel_size, stride) in enumerate(zip(kernel_sizes, rates)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, kernel_size, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


# Derive site-packages from where torch is actually installed — correct for both
# system Python and venv Python without any hardcoded paths.
_SP = str(Path(torch.__file__).parents[1])


def _ensure_trt_libs() -> None:
    """Preload libnvinfer into the global link map before importing torch_tensorrt.

    LD_LIBRARY_PATH is read once at process startup — os.environ changes have no effect
    on dlopen. The venv's libtorchtrt.so RPATH points to tensorrt_libs/, but only after
    patchelf converted RUNPATH→RPATH on the torch_tensorrt .so files. libnvinfer must
    still be globally visible so libtorchtrt.so finds it when loaded transitively.
    """
    import ctypes
    for _lib in [
        f"{_SP}/tensorrt_libs/libnvinfer.so.10",
        f"{_SP}/tensorrt_libs/libnvinfer_plugin.so.10",
    ]:
        if os.path.isfile(_lib):
            ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)


class AudioTokenizer():
    def __init__(
        self,
        model_path: str,
        use_trt: bool = False,
        gpu_chunk_size: int = 40,
    ):
        model_config = {'input_channel': 1024, 'channels': 1536, 'rates': [8, 5, 4, 2], 'kernel_sizes': [16, 11, 8, 4]}
        self.detokenizer = Decoder(**model_config)
        self.detokenizer.apply(remove_weight_norm_recursive)

        state_dict = load_file(model_path)
        missing_keys, unexpected_keys = self.detokenizer.load_state_dict(state_dict, strict=False)
        self.detokenizer = self.detokenizer.eval().float().to("cuda:0").half()

        self._trt_max_t: int | None = None
        self._trt_min_t: int | None = None
        self._fp16_fallback = None
        if use_trt:
            trt_model, trt_min_t, trt_max_t = self._load_trt(gpu_chunk_size, model_path)
            if trt_model is not None:
                self._fp16_fallback = self.detokenizer  # keep FP16 for out-of-profile inputs
                self.detokenizer = trt_model
                self._trt_min_t = trt_min_t
                self._trt_max_t = trt_max_t
                return
            print("[TRT] Falling back to plain FP16 decoder.")

    def _load_trt(self, gpu_chunk_size: int, model_path: str):
        """Load a pre-compiled TRT .ep engine. Returns (model, min_T, max_T) or (None, None, None)."""
        cache_path = Path(model_path).parent / f"decoder_trt_b{gpu_chunk_size}.ep"
        if not cache_path.exists():
            print(f"[TRT] No cached engine at {cache_path}. Falling back.")
            return None, None, None

        print(f"[TRT] Loading cached engine: {cache_path}")
        try:
            _ensure_trt_libs()
            import torch_tensorrt
            loaded = torch_tensorrt.load(str(cache_path))
            if hasattr(loaded, "module"):
                loaded = loaded.module()

            # Read T profile range from companion JSON written by compile_trt.py.
            # The loaded .ep is a GraphModule with no .engine attribute, so we
            # cannot extract the profile at runtime.
            import json as _json
            meta_path = cache_path.with_suffix(".json")
            if meta_path.exists():
                meta = _json.loads(meta_path.read_text())
                min_t = int(meta.get("min_t", 50))
                max_t = int(meta.get("max_t", 350))
            else:
                # Conservative defaults matching the original compile_trt.py profile.
                min_t, max_t = 50, 350
            print(f"[TRT] Engine loaded successfully. T_range=[{min_t}, {max_t}]")
            return loaded, min_t, max_t
        except Exception as e:
            print(f"[TRT] Load failed ({e}). Falling back.")
            return None, None, None

    def decode(self, x):
        # Fall back to FP16 if T is outside the compiled TRT profile range.
        # Short streaming chunks (chunk_tokens=15) hit the lower bound; long
        # sentences hit the upper bound.
        t = x.shape[-1]
        if self._fp16_fallback is not None and (
            (self._trt_min_t is not None and t < self._trt_min_t)
            or (self._trt_max_t is not None and t > self._trt_max_t)
        ):
            return self._fp16_fallback(x)
        return self.detokenizer(x)
