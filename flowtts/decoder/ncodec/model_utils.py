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


# Site-packages of the active venv (resolved at import time).
_SP = str(Path(__file__).parents[3] / "llm/lib/python3.12/site-packages")


def _ensure_trt_libs() -> None:
    """Prepend the cu13 CUDA runtime and TRT libs to LD_LIBRARY_PATH so that
    libtorchtrt.so can find libcudart.so.13 and the TensorRT shared libs.
    Must be called before the first `import torch_tensorrt`.
    """
    ld_paths = [
        f"{_SP}/nvidia/cu13/lib",
        f"{_SP}/tensorrt_libs",
        f"{_SP}/torch/lib",
    ]
    new_ld = ":".join(p for p in ld_paths if os.path.isdir(p))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{new_ld}:{existing}" if existing else new_ld


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

        if use_trt:
            trt_model = self._load_trt(gpu_chunk_size, model_path)
            if trt_model is not None:
                self.detokenizer = trt_model
                return
            print("[TRT] Falling back to plain FP16 decoder.")

    def _load_trt(self, gpu_chunk_size: int, model_path: str):
        """Load a pre-compiled TRT .ep engine. Returns the loaded model or None."""
        cache_path = Path(model_path).parent / f"decoder_trt_b{gpu_chunk_size}.ep"
        if not cache_path.exists():
            print(f"[TRT] No cached engine at {cache_path}. Falling back.")
            return None

        print(f"[TRT] Loading cached engine: {cache_path}")
        try:
            _ensure_trt_libs()
            import torch_tensorrt
            loaded = torch_tensorrt.load(str(cache_path))
            if hasattr(loaded, "module"):
                loaded = loaded.module()
            print("[TRT] Engine loaded successfully.")
            return loaded
        except Exception as e:
            print(f"[TRT] Load failed ({e}). Falling back.")
            return None

    def decode(self, x):
        return self.detokenizer(x)
