"""
Recompile the TRT decoder engine for torch_tensorrt 2.9.0 / torch 2.9.1.
Run once: python compile_trt.py
Writes decoder_trt_b90.ep next to detokenizer.safetensors.
"""
import os, sys
from pathlib import Path

# Ensure TRT shared libs are visible before importing torch_tensorrt.
SP = str(Path(__file__).parent / "llm/lib/python3.12/site-packages")
ld_paths = [f"{SP}/nvidia/cu13/lib", f"{SP}/tensorrt_libs", f"{SP}/torch/lib"]
existing = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in ld_paths if os.path.isdir(p)) + (f":{existing}" if existing else "")

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch_tensorrt
from safetensors.torch import load_file
from flowtts.decoder.ncodec.model_utils import Decoder, remove_weight_norm_recursive

DECODERS_PATH = Path.home() / ".cache/huggingface/hub/models--YatharthS--MiraTTS/snapshots/ff750bd74e7b2a2d9313873d583df34a91ed7d8a/decoders"
MODEL_PATH    = DECODERS_PATH / "detokenizer.safetensors"
GPU_CHUNK     = 90
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
print(f"Done. Engine saved to {CACHE_PATH}")
