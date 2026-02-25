import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
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
            # This is a good sign of weight_norm
            nn.utils.remove_weight_norm(m)
    except Exception as e:
        print(f"Could not remove weight norm from {m}: {e}")

def load_config(config_path: Path):

    # Load the initial configuration from the given path
    config = OmegaConf.load(config_path)

    # Check if there is a base configuration specified and merge if necessary
    if config.get("base_config", None) is not None:
        base_config = OmegaConf.load(config["base_config"])
        config = OmegaConf.merge(base_config, config)

    return config

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


class AudioTokenizer():
    def __init__(
        self,
        model_path: str,
        use_trt: bool = False,
        gpu_chunk_size: int = 40,
    ):
        """Loads the audio detokenizer.

        Args:
            model_path:      Path to detokenizer.safetensors.
            use_trt:         Compile with TensorRT FP16 for 3-5x faster decode.
                             Requires:  pip install torch-tensorrt
                             First call triggers a ~60 s one-time compilation.
            gpu_chunk_size:  Max batch size passed to the decoder (used to set
                             the TRT optimum/max shape profiles).
        """
        model_config = {'input_channel': 1024, 'channels': 1536, 'rates': [8, 5, 4, 2], 'kernel_sizes': [16, 11, 8, 4]}
        self.detokenizer = Decoder(**model_config)
        self.detokenizer.apply(remove_weight_norm_recursive)

        state_dict = load_file(model_path)
        missing_keys, unexpected_keys = self.detokenizer.load_state_dict(state_dict, strict=False)
        self.detokenizer = self.detokenizer.eval().float().to("cuda:0").half()

        if use_trt:
            result = self._load_or_compile_trt(gpu_chunk_size, model_path)
            if result is not None:
                self.detokenizer = result
            else:
                print("[TRT] Falling back to plain FP16 decoder.")

    # ------------------------------------------------------------------
    # TRT helpers — compile via venv2 (tensorrt-cu12, works on driver 570).
    # The main llmc venv has tensorrt-cu13 libs which require driver >= 575,
    # so TRT compilation and the cache .ep file are managed entirely through
    # the venv2 subprocess.  The main process always runs plain FP16.
    # ------------------------------------------------------------------

    _COMPILE_PYTHON = "/root/BatchBicodec/venv2/bin/python3"
    _COMPILE_SP     = "/root/BatchBicodec/venv2/lib/python3.12/site-packages"

    @classmethod
    def _trt_env(cls) -> dict:
        sp = cls._COMPILE_SP
        ld_paths = [
            f"{sp}/nvidia/cudnn/lib",
            f"{sp}/nvidia/cuda_runtime/lib",
            f"{sp}/tensorrt_libs",
            f"{sp}/torch/lib",
        ]
        new_ld = ":".join(p for p in ld_paths if os.path.isdir(p))
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        ld = f"{new_ld}:{existing_ld}" if existing_ld else new_ld
        env = {**os.environ, "LD_LIBRARY_PATH": ld}
        if not env.get("CUDA_VISIBLE_DEVICES"):
            try:
                import subprocess, re
                out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
                for line in out.splitlines():
                    m = re.search(r"(MIG-[0-9a-f\-]+)", line)
                    if m:
                        env["CUDA_VISIBLE_DEVICES"] = m.group(1)
                        break
            except Exception:
                pass
        return env

    def _load_or_compile_trt(self, gpu_chunk_size: int, model_path: str):
        """Compile TRT engine via the venv2 subprocess if not already cached.

        The venv2 (torch-tensorrt 2.8, tensorrt-cu12) is compatible with
        driver 570, unlike the main llmc venv's tensorrt-cu13 libs which
        require driver >= 575.  As a result:
          - Compilation happens in venv2 subprocess → writes .ep cache file.
          - Loading the .ep into the main process is not supported on this
            driver; _load_or_compile_trt always returns None so the caller
            falls back to plain FP16 PyTorch inference.

        Cache: decoder_trt_b{gpu_chunk_size}.ep next to model weights.
              Run compile_trt_mig.py in venv2 to pre-build this file.
        """
        import subprocess, textwrap

        cache_path = Path(model_path).parent / f"decoder_trt_b{gpu_chunk_size}.ep"

        if cache_path.exists():
            print(f"[TRT] Cache exists: {cache_path}")
            print("[TRT] In-process TRT load not available on this driver (570 < 575 required for cu13 TRT).")
            print("[TRT] Falling back to plain FP16 decoder.")
            return None

        print(f"[TRT] Compiling FP16 engine via venv2 "
              f"(opt_batch={gpu_chunk_size}) — ~60 s first time ...")

        flowtts_root = str(Path(__file__).parents[3])
        compile_script = textwrap.dedent(f"""
            import sys, torch, torch_tensorrt
            from pathlib import Path
            from safetensors.torch import load_file
            sys.path.insert(0, {flowtts_root!r})
            from flowtts.decoder.ncodec.model_utils import Decoder, remove_weight_norm_recursive

            model_config = {{'input_channel': 1024, 'channels': 1536,
                             'rates': [8, 5, 4, 2], 'kernel_sizes': [16, 11, 8, 4]}}
            det = Decoder(**model_config)
            det.apply(remove_weight_norm_recursive)
            det.load_state_dict(load_file({str(model_path)!r}), strict=False)
            det = det.eval().float().to('cuda:0').half()

            compiled = torch_tensorrt.compile(
                det,
                inputs=[torch_tensorrt.Input(
                    min_shape=(1, 1024, 50),
                    opt_shape=({gpu_chunk_size}, 1024, 172),
                    max_shape=({gpu_chunk_size}, 1024, 350),
                    dtype=torch.float16,
                )],
                enabled_precisions={{torch.float16}},
                truncate_long_and_double=True,
                require_full_compilation=False,
            )
            example = torch.zeros({gpu_chunk_size}, 1024, 172, dtype=torch.float16, device='cuda:0')
            torch_tensorrt.save(compiled, {str(cache_path)!r}, inputs=[example])
            print('TRT_COMPILE_OK')
        """)

        result = subprocess.run(
            [self._COMPILE_PYTHON, "-c", compile_script],
            env=self._trt_env(),
            timeout=1800,  # TRT dynamo compile can take 10-30 min first time
            capture_output=False,
        )
        if result.returncode != 0 or not cache_path.exists():
            print(f"[TRT] Subprocess compile failed (rc={result.returncode}). Using plain FP16.")
        else:
            print(f"[TRT] Engine cached at {cache_path}. Plain FP16 used in this process.")
        # In-process TRT load is not available on driver 570 (cu13 TRT requires >= 575).
        # The compiled .ep is available for future use when the driver is upgraded.
        return None

    def decode(self, x):
        return self.detokenizer(x)
