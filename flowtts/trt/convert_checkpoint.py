"""Pipeline position: ACCELERATION (build time) — HF weights → TRT-LLM checkpoint.

Vendored from github.com/tlitech/omnivoice-trtllm (``scripts/convert_checkpoint.py``)
with no behavioural change, so the TRT-LLM engine this repo builds is the same
engine upstream builds. Driven by :mod:`flowtts.trt.build_trtllm`.

Only relevant on a box that has TensorRT-LLM installed; the default path on this
service is :mod:`flowtts.trt.build_trt`, which needs only TensorRT.

Original docstring follows.

Convert OmniVoice HuggingFace checkpoint to TRT-LLM format.

Only converts the Qwen3 transformer layers + final norm.
Audio embeddings, text embeddings, and audio heads stay in PyTorch
and are loaded directly from the original checkpoint at runtime.

Supports FP8 weight quantization (--fp8) for ~2x speedup on L4/H100.

Usage:
    # FP16 (default)
    python convert_checkpoint.py \
        --model_dir /path/to/OmniVoice \
        --output_dir /path/to/trtllm_ckpt

    # FP8 weight quantization
    python convert_checkpoint.py \
        --model_dir /path/to/OmniVoice \
        --output_dir /path/to/trtllm_ckpt_fp8 \
        --fp8
"""
import argparse
import json
import os
import re
import time

import safetensors.torch
import torch


def str_dtype_to_torch(s):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the OmniVoice HuggingFace model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tllm_checkpoint",
        help="Path to save TRT-LLM checkpoint",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Quantize linear layer weights to FP8 (E4M3). "
        "Requires GPU with FP8 support (L4, H100, L40S, RTX 4090).",
    )
    return parser.parse_args()


def load_hf_weights(model_dir):
    """Load weights from HuggingFace safetensors files."""
    weight_files = sorted(
        [
            f
            for f in os.listdir(model_dir)
            if f.endswith(".safetensors") and f != "model.safetensors.index.json"
        ]
    )

    if not weight_files:
        raise FileNotFoundError(
            f"No safetensors files found in {model_dir}"
        )

    all_weights = {}
    for wf in weight_files:
        path = os.path.join(model_dir, wf)
        print(f"Loading {path}")
        weights = safetensors.torch.load_file(path)
        all_weights.update(weights)

    return all_weights


# Weight name mapping: HuggingFace → TRT-LLM
#
# Actual HF format (from model.safetensors):
#   llm.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
#   llm.layers.{i}.self_attn.{q_norm,k_norm}.weight
#   llm.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight
#   llm.layers.{i}.input_layernorm.weight
#   llm.layers.{i}.post_attention_layernorm.weight
#   llm.norm.weight
#
# TRT-LLM format (matches patch/omnivoice/modules.py attribute names):
#   layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
#   layers.{i}.self_attn.{q_norm,k_norm}.weight
#   layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight
#   layers.{i}.input_layernorm.weight
#   layers.{i}.post_attention_layernorm.weight
#   final_norm.weight

NAME_MAP = {
    # Final norm
    r"^llm\.norm\.weight$": "final_norm.weight",
    # Attention projections + QK norms
    r"^llm\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$": r"layers.\1.self_attn.\2.weight",
    r"^llm\.layers\.(\d+)\.self_attn\.(q_norm|k_norm)\.weight$": r"layers.\1.self_attn.\2.weight",
    # MLP
    r"^llm\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$": r"layers.\1.mlp.\2.weight",
    # Layer norms
    r"^llm\.layers\.(\d+)\.input_layernorm\.weight$": r"layers.\1.input_layernorm.weight",
    r"^llm\.layers\.(\d+)\.post_attention_layernorm\.weight$": r"layers.\1.post_attention_layernorm.weight",
}

# Keys to skip (not part of the TRT engine)
SKIP_PREFIXES = [
    "llm.embed_tokens.",  # text embedding (PyTorch)
    "audio_embeddings.",  # audio embedding (PyTorch)
    "audio_heads.",  # audio output heads (PyTorch)
    "codebook_layer_offsets",  # buffer, not a weight
    "llm.lm_head.",  # LM head (unused)
]

# Weights that should be quantized to FP8 (linear layer weights only).
# Norms and biases stay in higher precision.
FP8_QUANTIZE_PATTERNS = [
    r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
    r"mlp\.(gate_proj|up_proj|down_proj)\.weight$",
]


def _should_quantize_fp8(name):
    """Check if a weight should be quantized to FP8."""
    return any(re.search(p, name) for p in FP8_QUANTIZE_PATTERNS)


def _quantize_to_fp8(weight):
    """Quantize a weight tensor to FP8 E4M3 with per-tensor scaling.

    Args:
        weight: FP16/BF16/FP32 tensor

    Returns:
        (fp8_weight, scale) where:
        - fp8_weight: torch.float8_e4m3fn tensor
        - scale: FP32 scalar (amax / fp8_max)
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    w_float = weight.float()
    amax = w_float.abs().amax()
    # Avoid division by zero
    scale = (amax / fp8_max).clamp(min=1e-12)
    fp8_weight = (w_float / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return fp8_weight, scale


def map_weight_name(hf_name):
    """Map HF weight name to TRT-LLM name. Returns None if skipped."""
    for prefix in SKIP_PREFIXES:
        if hf_name.startswith(prefix):
            return None

    for pattern, replacement in NAME_MAP.items():
        match = re.match(pattern, hf_name)
        if match:
            return re.sub(pattern, replacement, hf_name)

    return None


def convert_checkpoint(args):
    torch_dtype = str_dtype_to_torch(args.dtype)
    tik = time.time()

    # Load HF weights
    hf_weights = load_hf_weights(args.model_dir)
    print(f"Loaded {len(hf_weights)} weight tensors from HF checkpoint")

    # Print first 20 weight names for debugging
    print("Sample weight names:")
    for i, name in enumerate(sorted(hf_weights.keys())):
        if i < 20:
            print(f"  {name}")
    print("  ...")

    # Convert
    trtllm_weights = {}
    skipped = []
    unmapped = []
    fp8_quantized = []

    for name, param in hf_weights.items():
        trtllm_name = map_weight_name(name)
        if trtllm_name is None:
            skipped.append(name)
            continue

        if args.fp8 and _should_quantize_fp8(trtllm_name):
            # FP8 quantization: store quantized weight + scaling factors
            fp8_weight, w_scale = _quantize_to_fp8(param)
            trtllm_weights[trtllm_name] = fp8_weight
            # Weight scaling factor
            w_scale_name = trtllm_name.replace(".weight", ".weights_scaling_factor")
            trtllm_weights[w_scale_name] = w_scale.reshape(1)
            # Activation scaling factor (default 1.0 — no calibration data).
            # TRT-LLM requires this for FP8 W8A8. scale=1.0 means activations
            # are quantized with identity scaling, which is safe but not optimal.
            # For better quality, run calibration and replace these values.
            a_scale_name = trtllm_name.replace(".weight", ".activation_scaling_factor")
            trtllm_weights[a_scale_name] = torch.ones(1, dtype=torch.float32)
            fp8_quantized.append(trtllm_name)
        else:
            # Keep in original dtype (norms, etc.)
            trtllm_weights[trtllm_name] = param.contiguous().to(torch_dtype)

    # Report
    print(f"Converted {len(trtllm_weights)} weights to TRT-LLM format")
    print(f"Skipped {len(skipped)} weights (embeddings/heads, kept in PyTorch)")
    if args.fp8:
        print(f"FP8 quantized: {len(fp8_quantized)} linear layer weights")

    for name in hf_weights:
        if name not in skipped and map_weight_name(name) is None:
            unmapped.append(name)
    if unmapped:
        print(f"WARNING: {len(unmapped)} unmapped weights:")
        for n in unmapped[:10]:
            print(f"  {n}")

    tok = time.time()
    print(f"Conversion took {tok - tik:.1f}s")

    return trtllm_weights


def save_config(args, output_dir):
    """Save TRT-LLM config.json based on the HF model config."""
    hf_config_path = os.path.join(args.model_dir, "config.json")
    with open(hf_config_path) as f:
        hf_config = json.load(f)

    llm_config = hf_config.get("llm_config", {})

    config = {
        "architecture": "OmniVoice",
        "dtype": args.dtype,
        "hidden_size": llm_config.get("hidden_size", 1024),
        "num_hidden_layers": llm_config.get("num_hidden_layers", 28),
        "num_attention_heads": llm_config.get("num_attention_heads", 16),
        "num_kv_heads": llm_config.get("num_key_value_heads", 8),
        "head_dim": llm_config.get("head_dim", 128),
        "intermediate_size": llm_config.get("intermediate_size", 3072),
        "rms_norm_eps": llm_config.get("rms_norm_eps", 1e-6),
        "rope_theta": llm_config.get("rope_parameters", {}).get(
            "rope_theta", 1000000
        ),
        "mapping": {
            "world_size": 1,
            "cp_size": 1,
            "tp_size": 1,
            "pp_size": 1,
        },
    }

    if args.fp8:
        config["quantization"] = {
            "quant_algo": "FP8",
            "kv_cache_quant_algo": None,
        }

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Saved TRT-LLM config to {config_path}")


def main():
    args = parse_arguments()

    save_config(args, args.output_dir)
    weights = convert_checkpoint(args)

    output_path = os.path.join(args.output_dir, "rank0.safetensors")
    safetensors.torch.save_file(weights, output_path)
    print(f"Saved TRT-LLM weights to {output_path}")

    if args.fp8:
        print("\nFP8 quantization summary:")
        fp8_count = sum(1 for k in weights if k.endswith(".weights_scaling_factor"))
        print(f"  {fp8_count} layers quantized to FP8 E4M3")
        print(f"  Build with: trtllm-build --gemm_plugin fp8 ...")


if __name__ == "__main__":
    main()
