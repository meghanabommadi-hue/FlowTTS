"""Pipeline position: ACCELERATION — build a TensorRT engine for the Qwen3 backbone.

Role in pipeline (build time, run once per box):

    OmniVoice HF weights
      └─ Qwen3Backbone.from_llm()      [flowtts.trt.backbone]
           └─ torch.onnx.export()      → backbone.onnx
                └─ TensorRT builder    → backbone.plan  (+ manifest.json)
                     └─ flowtts.trt.runtime.TRTBackbone → patched llm.forward

Why this exists alongside the TRT-LLM path:
  github.com/tlitech/omnivoice-trtllm builds the same engine through
  TensorRT-LLM 0.18.2 — but that pins torch 2.6 and TensorRT 10.8, and the NGC
  container this service runs in ships torch 2.8 + TensorRT 10.11 with a global
  pip constraint holding torch there. Installing TRT-LLM into the serving
  environment would break the other services on the box.

  The engine upstream produces does not actually use any TensorRT-LLM *runtime*
  feature: no KV cache, no in-flight batching, no paged attention — it is run
  through a bare ``Session``, which is a thin wrapper over a TensorRT execution
  context. TRT-LLM is used only as a graph-authoring API. So this path builds
  the identical network, with the identical I/O contract (hidden_states,
  rope_cos, rope_sin, input_lengths → output), straight from PyTorch via ONNX
  using the TensorRT already installed. ``flowtts.trt.build_trtllm`` remains
  available for boxes that do have TensorRT-LLM.

Usage:
    python -m flowtts.trt.build_trt --precision fp16 --max-batch 64 --max-seq 2048
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# I/O tensor names — must match upstream's engine exactly so one runtime wrapper
# can drive either engine kind.
INPUT_NAMES = ("hidden_states", "rope_cos", "rope_sin", "input_lengths")
OUTPUT_NAME = "output"

DEFAULT_ENGINE_DIR = "engines/omnivoice-backbone"
MANIFEST_NAME = "manifest.json"
ENGINE_NAME = "backbone.plan"
ONNX_NAME = "backbone.onnx"


def export_onnx(
    backbone,
    onnx_path: str | Path,
    *,
    opt_batch: int = 8,
    opt_seq: int = 512,
    dtype=None,
    device: str = "cuda",
    opset: int = 17,
) -> Path:
    """Export a :class:`~flowtts.trt.backbone.Qwen3Backbone` to ONNX."""
    import torch

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = backbone.cfg
    dtype = dtype or torch.float16

    dummy = (
        torch.randn(opt_batch, opt_seq, cfg.hidden_size, device=device, dtype=dtype),
        torch.randn(opt_batch, opt_seq, cfg.head_dim, device=device, dtype=dtype),
        torch.randn(opt_batch, opt_seq, cfg.head_dim, device=device, dtype=dtype),
        torch.full((opt_batch,), opt_seq, dtype=torch.int32, device=device),
    )

    # The explicit attention form only for the export; see
    # Qwen3Attention._attention_explicit for why SDPA's subgraph cannot be built.
    if hasattr(backbone, "set_export_mode"):
        backbone.set_export_mode(True)

    with torch.no_grad():
        torch.onnx.export(
            backbone,
            dummy,
            str(onnx_path),
            input_names=list(INPUT_NAMES),
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                "hidden_states": {0: "batch", 1: "seq"},
                "rope_cos": {0: "batch", 1: "seq"},
                "rope_sin": {0: "batch", 1: "seq"},
                "input_lengths": {0: "batch"},
                OUTPUT_NAME: {0: "batch", 1: "seq"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    if hasattr(backbone, "set_export_mode"):
        backbone.set_export_mode(False)
    return onnx_path


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    hidden_size: int,
    head_dim: int,
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 64,
    min_seq: int = 16,
    opt_seq: int = 512,
    max_seq: int = 2048,
    precision: str = "fp16",
    workspace_gb: float = 8.0,
    builder_optimization_level: int = 3,
    verbose: bool = False,
) -> Path:
    """Compile *onnx_path* into a TensorRT engine with one dynamic-shape profile.

    ``precision`` names the dtype the ONNX was exported in (``fp16`` / ``bf16`` /
    ``fp32``), which a strongly-typed network then honours exactly. ``fp8``
    additionally needs an ONNX carrying Q/DQ nodes (from NVIDIA ModelOpt) and an
    Ada or Hopper GPU; without those nodes the layers stay at the exported
    precision and the manifest records what was asked for.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # STRONGLY_TYPED, not the FP16 builder flag. This is load-bearing.
    #
    # With the FP16 flag, TensorRT is free to re-fold the graph's precision and
    # it pushes the fp32 casts out of RMSNorm. The norm then squares its input
    # in fp16, and OmniVoice's hidden states reach absmax ~325 — 325^2 = 105625,
    # past fp16's 65504 — so the sum overflows to inf, rsqrt(inf) is 0, and the
    # whole layer outputs zeros. The engine builds cleanly and silently returns
    # silence; only the patcher's cosine check catches it.
    #
    # A strongly-typed network takes its precision from the ONNX graph verbatim,
    # exactly as onnxruntime does: fp16 matmuls where the weights are fp16, fp32
    # inside the norms where the casts say so. TensorRT's own log warns about
    # this ("exporting the model to use INormalizationLayer ... can help
    # preserving accuracy") — at opset 17 there is no RMSNormalization op to
    # emit, so this is the fix.
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = Path(onnx_path).read_bytes()
    if not parser.parse(onnx_bytes):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.builder_optimization_level = builder_optimization_level

    # Precision flags are rejected on a strongly-typed network — the ONNX dtypes
    # are the specification. `precision` therefore selects the dtype the backbone
    # is exported in (see build_from_model), not a builder flag.
    if precision == "fp8":
        config.set_flag(trt.BuilderFlag.FP8)

    profile = builder.create_optimization_profile()
    profile.set_shape("hidden_states",
                      (min_batch, min_seq, hidden_size),
                      (opt_batch, opt_seq, hidden_size),
                      (max_batch, max_seq, hidden_size))
    for name in ("rope_cos", "rope_sin"):
        profile.set_shape(name,
                          (min_batch, min_seq, head_dim),
                          (opt_batch, opt_seq, head_dim),
                          (max_batch, max_seq, head_dim))
    profile.set_shape("input_lengths", (min_batch,), (opt_batch,), (max_batch,))
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed (see TRT log above)")

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(plan))

    manifest = {
        "kind": "tensorrt",
        "tensorrt_version": trt.__version__,
        "precision": precision,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "min_batch": min_batch, "opt_batch": opt_batch, "max_batch": max_batch,
        "min_seq": min_seq, "opt_seq": opt_seq, "max_seq": max_seq,
        "inputs": list(INPUT_NAMES),
        "output": OUTPUT_NAME,
        "build_seconds": round(time.perf_counter() - t0, 1),
        "engine_bytes": engine_path.stat().st_size,
    }
    (engine_path.parent / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return engine_path


def build_from_model(
    model,
    engine_dir: str | Path = DEFAULT_ENGINE_DIR,
    *,
    precision: str = "fp16",
    keep_onnx: bool = False,
    **build_kwargs,
) -> Path:
    """End-to-end: loaded OmniVoice → ONNX → TensorRT engine on disk."""
    import torch

    from flowtts.trt.backbone import BackboneConfig, Qwen3Backbone

    engine_dir = Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    cfg = BackboneConfig.from_hf(model.llm.config)
    # The constant position buffer is baked into the engine, so it has to cover
    # the largest sequence the optimization profile allows.
    backbone = Qwen3Backbone.from_llm(
        model.llm, cfg, max_positions=max(build_kwargs.get("max_seq", 2048), 2048)
    )

    dtype = torch.float32 if precision == "fp32" else (
        torch.bfloat16 if precision == "bf16" else torch.float16
    )
    backbone = backbone.to(dtype=dtype)

    onnx_path = engine_dir / ONNX_NAME
    export_onnx(
        backbone, onnx_path,
        opt_batch=build_kwargs.get("opt_batch", 8),
        opt_seq=build_kwargs.get("opt_seq", 512),
        dtype=dtype,
        device=str(next(backbone.parameters()).device),
    )
    try:
        return build_engine(
            onnx_path, engine_dir / ENGINE_NAME,
            hidden_size=cfg.hidden_size, head_dim=cfg.head_dim,
            precision=precision, **build_kwargs,
        )
    finally:
        if not keep_onnx:
            onnx_path.unlink(missing_ok=True)
            # The exporter writes large initializers beside the graph when the
            # model exceeds the 2 GB protobuf limit; clean those up too.
            for extra in engine_dir.glob("*.onnx_data"):
                extra.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the OmniVoice Qwen3 TensorRT engine")
    ap.add_argument("--engine-dir", default=None,
                    help=f"output directory (default: settings value, else {DEFAULT_ENGINE_DIR})")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp8", "fp32"])
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--opt-batch", type=int, default=8)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--min-seq", type=int, default=16)
    ap.add_argument("--opt-seq", type=int, default=512)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--workspace-gb", type=float, default=8.0)
    ap.add_argument("--keep-onnx", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import torch

    from flowtts.core.config import resolve_model_source, settings

    engine_dir = args.engine_dir or settings.omnivoice.trt_engine_dir or DEFAULT_ENGINE_DIR

    from omnivoice import OmniVoice

    source = resolve_model_source()
    print(f"[build_trt] loading OmniVoice from {source}", flush=True)
    model = OmniVoice.from_pretrained(
        source,
        device_map=settings.omnivoice.device,
        dtype=torch.float16,     # engine weights are fp16 regardless of serve dtype
        load_asr=False,
        trust_remote_code=settings.omnivoice.trust_remote_code,
    )

    print(f"[build_trt] exporting + building ({args.precision})…", flush=True)
    path = build_from_model(
        model, engine_dir,
        precision=args.precision,
        keep_onnx=args.keep_onnx,
        min_batch=args.min_batch, opt_batch=args.opt_batch, max_batch=args.max_batch,
        min_seq=args.min_seq, opt_seq=args.opt_seq, max_seq=args.max_seq,
        workspace_gb=args.workspace_gb, verbose=args.verbose,
    )
    manifest = json.loads((path.parent / MANIFEST_NAME).read_text())
    print(f"[build_trt] engine → {path}  ({manifest['engine_bytes'] / 1e6:.0f} MB, "
          f"{manifest['build_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
