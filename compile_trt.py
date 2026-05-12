"""
Compile the TRT decoder engine for torch_tensorrt 2.8.0 / torch 2.8.0.
Run once: python compile_trt.py
Writes decoder_trt_b160.ep next to detokenizer.safetensors.
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch  # must come first — loads venv's libtorch_cuda.so into the dl map

# libnvinfer.so.10 must be in the global link map before libtorchtrt.so is loaded.
# LD_LIBRARY_PATH is read once at process start so os.environ changes won't help;
# ctypes.CDLL with the full venv path + RTLD_GLOBAL is the reliable approach.
import ctypes
_SP = str(Path(torch.__file__).parents[1])
for _lib in [
    f"{_SP}/tensorrt_libs/libnvinfer.so.10",
    f"{_SP}/tensorrt_libs/libnvinfer_plugin.so.10",
]:
    if os.path.isfile(_lib):
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
    else:
        raise RuntimeError(f"Required lib not found: {_lib}. Run with: .venv/bin/python compile_trt.py")

import torch_tensorrt
from safetensors.torch import load_file
from flowtts.decoder.ncodec.model_utils import Decoder, remove_weight_norm_recursive

DECODERS_PATH = Path.home() / ".cache/huggingface/hub/models--YatharthS--MiraTTS/snapshots/ff750bd74e7b2a2d9313873d583df34a91ed7d8a/decoders"
MODEL_PATH    = DECODERS_PATH / "detokenizer.safetensors"
GPU_CHUNK     = 150
CACHE_PATH    = DECODERS_PATH / f"decoder_trt_b{GPU_CHUNK}.ep"

print(f"Loading model from {MODEL_PATH} ...")
model_config = {'input_channel': 1024, 'channels': 1536, 'rates': [8, 5, 4, 2], 'kernel_sizes': [16, 11, 8, 4]}
det = Decoder(**model_config)
det.apply(remove_weight_norm_recursive)
det.load_state_dict(load_file(str(MODEL_PATH)), strict=False)
det = det.eval().float().to("cuda:0").half()

print(f"Compiling TRT engine (opt_batch={GPU_CHUNK}) — this takes ~5-15 min first time ...")
compiled = torch_tensorrt.compile(
    det,
    inputs=[torch_tensorrt.Input(
        min_shape=(1,    1024, 50),
        opt_shape=(GPU_CHUNK, 1024, 172),
        max_shape=(GPU_CHUNK, 1024, 350),
        dtype=torch.float16,
    )],
    enabled_precisions={torch.float16},
    truncate_long_and_double=True,
    require_full_compilation=False,
)

example = torch.zeros(GPU_CHUNK, 1024, 172, dtype=torch.float16, device="cuda:0")
print(f"Saving to {CACHE_PATH} ...")
torch_tensorrt.save(compiled, str(CACHE_PATH), inputs=[example])

import json
META_PATH = CACHE_PATH.with_suffix(".json")
META_PATH.write_text(json.dumps({"min_t": 50, "max_t": 350, "gpu_chunk": GPU_CHUNK}))
print(f"Done. Engine saved to {CACHE_PATH}")
