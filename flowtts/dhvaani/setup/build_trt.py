"""Pipeline position: SETUP — export the flow decoder to ONNX and build TensorRT.

Run once per machine (engines are hardware- and TensorRT-version-specific):

    python -m flowtts.dhvaani.setup.build_trt --max-batch 64
    python -m flowtts.dhvaani.setup.build_trt --validate

Engine contract (identical to upstream ZipVoice's exporter, on purpose):

    input   x             (N, T, 300)  fp16
    input   t             (N,)         fp16
    input   padding_mask  (N, T)       fp16   1.0 at padded positions
    output  v             (N, T, 100)  fp16

so an engine from `python -m zipvoice.bin.tensorrt_export` also loads in
`backends/trt_backend.py`, and vice versa.

Two upstream tricks are essential and reproduced here
------------------------------------------------------
1. `CompactRelPositionalEncoding.extend_pe` caches `self.pe` and returns early
   when the cached length is sufficient. Under tracing that bakes ONE sequence
   length into the graph. The patched version below always recomputes, which is
   what makes a dynamic `T` axis possible.

2. `convert_scaled_to_non_scaled(..., is_onnx=True)` folds the training-only
   modules (Balancer, Whiten, Dropout3) into identities, swaps the Swoosh
   activations for their ONNX-expressible forms, and `torch.jit.script`s the
   positional encoding so its length stays dynamic. Without it the export
   either fails or silently produces a fixed-shape graph.

The rest of the Zipformer is already export-safe: every stochastic branch is
guarded by `torch.jit.is_scripting() or torch.jit.is_tracing()`, so the traced
graph takes the deterministic path. That is why `dynamo=False` (the TorchScript
tracer) is used rather than the dynamo exporter -- it is the path those guards
were written for.

Batch sizing note
-----------------
DhVaani-0.5 is the NON-distilled ZipVoice, so classifier-free guidance is
applied by DOUBLING the batch (see `ops.cfg_expand`). The engine's max batch
must therefore be 2x the scheduler's `engine.max_batch_size`, which
`--max-batch` accounts for automatically.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
from pathlib import Path

from flowtts.dhvaani.config import N_MELS, dhv_settings

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("build_trt")


# ---------------------------------------------------------------------------
# Upstream patches (Apache-2.0, k2-fsa/ZipVoice)
# ---------------------------------------------------------------------------
def patch_positional_encoding() -> None:
    """Force `extend_pe` to recompute for every input length.

    The cached fast path returns early when `self.pe` is already long enough,
    which under `torch.jit.trace` freezes the encoding at the traced length.
    """
    import torch
    from zipvoice.models.modules.zipformer import CompactRelPositionalEncoding

    def extend_pe(self, x, left_context_len: int = 0) -> None:
        T = x.size(0) + left_context_len
        x = torch.arange(-(T - 1), T, device=x.device).to(torch.float32).unsqueeze(1)
        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)
        compression_length = self.embed_dim**0.5
        x_compressed = (
            compression_length
            * x.sign()
            * ((x.abs() + compression_length).log() - math.log(compression_length))
        )
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)
        x_atan = (x_compressed / length_scale).atan()
        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()
        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0
        self.pe = pe.to(dtype=x.dtype)

    CompactRelPositionalEncoding.extend_pe = extend_pe
    log.info("patched CompactRelPositionalEncoding.extend_pe (cache removed)")


def convert_scaled_to_non_scaled(model, inplace: bool = False, is_onnx: bool = True):
    """Vendored from `zipvoice.utils.scaling_converter` (Apache-2.0).

    DhVaani's trimmed `_backend/` omits this module, but every symbol it needs
    is present, so the 40 lines that matter are reproduced here rather than
    adding a dependency on a full upstream checkout.
    """
    import torch
    import torch.nn as nn
    from zipvoice.models.modules.scaling import (
        Balancer, Dropout3, SwooshL, SwooshLOnnx, SwooshR, SwooshROnnx, Whiten,
    )
    from zipvoice.models.modules.zipformer import CompactRelPositionalEncoding

    if not inplace:
        model = copy.deepcopy(model)

    def get_submodule(m, target):
        if target == "":
            return m
        for item in target.split("."):
            m = getattr(m, item)
        return m

    d = {}
    for name, m in model.named_modules():
        if isinstance(m, (Balancer, Dropout3, Whiten)):
            d[name] = nn.Identity()
        elif is_onnx and isinstance(m, SwooshR):
            d[name] = SwooshROnnx()
        elif is_onnx and isinstance(m, SwooshL):
            d[name] = SwooshLOnnx()
        elif is_onnx and isinstance(m, CompactRelPositionalEncoding):
            # Scripted (not traced) so the encoding length follows the input.
            d[name] = torch.jit.script(m)

    for k, v in d.items():
        if "." in k:
            parent, child = k.rsplit(".", maxsplit=1)
            setattr(get_submodule(model, parent), child, v)
        else:
            setattr(model, k, v)
    log.info("converted %d scaled/training modules for ONNX export", len(d))
    return model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(model, out_path: Path, trace_batch: int = 2, trace_frames: int = 256,
                opset: int = 18) -> Path:
    import torch

    fm = model.fm_decoder
    x = torch.randn(trace_batch, trace_frames, N_MELS * 3, dtype=torch.float32)
    t = torch.full((trace_batch,), 0.5, dtype=torch.float32)
    mask = torch.zeros(trace_batch, trace_frames, dtype=torch.bool)

    log.info("tracing fm_decoder at (B=%d, T=%d)", trace_batch, trace_frames)
    try:
        traced = torch.jit.trace(fm, (x, t, mask), strict=False)
    except Exception as e:
        raise RuntimeError(
            f"torch.jit.trace of fm_decoder failed: {e}\n"
            "This usually means a training-only module survived "
            "convert_scaled_to_non_scaled. Check which op is named in the error "
            "and add it to the conversion table above."
        ) from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("exporting ONNX -> %s (opset %d)", out_path, opset)
    torch.onnx.export(
        traced,
        (x, t, mask),
        str(out_path),
        opset_version=opset,
        input_names=["x", "t", "padding_mask"],
        output_names=["v"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "t": {0: "N"},
            "padding_mask": {0: "N", 1: "T"},
            "v": {0: "N", 1: "T"},
        },
        dynamo=False,  # the is_tracing() guards in the Zipformer target this path
    )
    log.info("ONNX written (%.1f MiB)", out_path.stat().st_size / 2**20)
    return out_path


# ---------------------------------------------------------------------------
# TensorRT build
# ---------------------------------------------------------------------------
def build_engine(onnx_path: Path, plan_path: Path, min_shape, opt_shape, max_shape,
                 fp16: bool = True, workspace_gb: int = 8,
                 timing_cache: Path | None = None) -> Path:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed:\n" + "\n".join(errs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # A persisted timing cache turns a 10-minute rebuild into seconds when only
    # the shape profile changed.
    cache_obj = None
    if timing_cache is not None:
        blob = timing_cache.read_bytes() if timing_cache.exists() else b""
        cache_obj = config.create_timing_cache(blob)
        config.set_timing_cache(cache_obj, ignore_mismatch=False)

    profile = builder.create_optimization_profile()
    present = {network.get_input(i).name for i in range(network.num_inputs)}
    shapes = {
        "x": ((min_shape[0], min_shape[1], N_MELS * 3),
              (opt_shape[0], opt_shape[1], N_MELS * 3),
              (max_shape[0], max_shape[1], N_MELS * 3)),
        "t": ((min_shape[0],), (opt_shape[0],), (max_shape[0],)),
        "padding_mask": ((min_shape[0], min_shape[1]),
                         (opt_shape[0], opt_shape[1]),
                         (max_shape[0], max_shape[1])),
        # Only ZipVoice-Distill has this input. Setting a profile for an input
        # the network does not declare is a build error -- upstream's script has
        # this bug for the non-distill model; we guard on `present` instead.
        "guidance_scale": ((min_shape[0],), (opt_shape[0],), (max_shape[0],)),
    }
    for name, (lo, opt, hi) in shapes.items():
        if name in present:
            profile.set_shape(name, lo, opt, hi)
    config.add_optimization_profile(profile)

    dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    for i in range(network.num_inputs):
        network.get_input(i).dtype = dtype
    for i in range(network.num_outputs):
        network.get_output(i).dtype = dtype

    log.info("building engine %s  min=%s opt=%s max=%s fp16=%s",
             plan_path.name, min_shape, opt_shape, max_shape, fp16)
    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT build returned no engine (see the log above)")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(blob)

    if timing_cache is not None and cache_obj is not None:
        timing_cache.write_bytes(memoryview(cache_obj.serialize()))

    meta = {
        "inputs": {
            "x": {"min": list(shapes["x"][0]), "opt": list(shapes["x"][1]),
                  "max": list(shapes["x"][2])},
            "t": {"min": list(shapes["t"][0]), "max": list(shapes["t"][2])},
            "padding_mask": {"min": list(shapes["padding_mask"][0]),
                             "max": list(shapes["padding_mask"][2])},
        },
        "outputs": {"v": [-1, -1, N_MELS]},
        "fp16": fp16,
        "trt_version": trt.__version__,
        "built_at": time.time(),
        "note": "compatible with zipvoice.bin.tensorrt_export engines",
    }
    plan_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    log.info("engine written in %.1fs (%.1f MiB)",
             time.perf_counter() - t0, plan_path.stat().st_size / 2**20)
    return plan_path


def plan_profiles(settings, max_batch: int) -> list[tuple]:
    """Split the bucket range into a few engines.

    One engine spanning 128..1536 frames would have TensorRT pick kernels for
    the midpoint and run poorly at both ends. Three overlapping profiles keep
    each specialised without exploding build time.
    """
    buckets = settings.buckets.buckets
    lo, hi = buckets[0], buckets[-1]
    mid1 = buckets[len(buckets) // 3]
    mid2 = buckets[2 * len(buckets) // 3]
    return [
        ((1, lo), (max_batch // 2 or 1, mid1), (max_batch, mid1)),
        ((1, mid1), (max_batch // 2 or 1, mid2), (max_batch, mid2)),
        ((1, mid2), (max(1, max_batch // 4), hi), (max_batch, hi)),
    ]


def build_engines(settings=None, max_batch: int | None = None,
                  fp16: bool | None = None, keep_onnx: bool = False) -> list[Path]:
    """Export + build every engine the scheduler will need."""
    import torch

    from flowtts.dhvaani.model.loader import load_model

    s = settings or dhv_settings
    out_dir = Path(s.backend.trt_engine_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    fp16 = s.backend.trt_fp16 if fp16 is None else fp16
    # CFG doubles the batch through the decoder for the non-distilled model.
    max_batch = max_batch or (s.engine.max_batch_size * 2)

    loaded = load_model(s)
    model = copy.deepcopy(loaded.zipvoice).to("cpu").float().eval()

    patch_positional_encoding()
    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)

    onnx_path = out_dir / "fm_decoder.onnx"
    with torch.no_grad():
        export_onnx(model, onnx_path)

    plans: list[Path] = []
    cache = out_dir / "timing.cache"
    for lo, opt, hi in plan_profiles(s, max_batch):
        name = f"fm_step_b{hi[0]}_t{lo[1]}-{hi[1]}.plan"
        plans.append(build_engine(onnx_path, out_dir / name, lo, opt, hi,
                                  fp16=fp16, workspace_gb=s.backend.trt_workspace_gb,
                                  timing_cache=cache))
    if not keep_onnx:
        onnx_path.unlink(missing_ok=True)
    log.info("built %d engine(s) in %s", len(plans), out_dir)
    return plans


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(settings=None, batch: int = 4, frames: int = 384) -> int:
    """Compare the TensorRT engine against eager PyTorch on identical inputs."""
    import torch

    from flowtts.dhvaani.backends.torch_backend import TorchFmBackend
    from flowtts.dhvaani.backends.trt_backend import TrtFmBackend
    from flowtts.dhvaani.model.loader import load_model

    s = settings or dhv_settings
    loaded = load_model(s)
    ref = TorchFmBackend(loaded, s)
    try:
        trt_be = TrtFmBackend(loaded, s)
    except Exception as e:
        print(f"cannot load TensorRT backend: {e}", file=sys.stderr)
        return 1

    dev, dt = loaded.device, loaded.dtype
    torch.manual_seed(0)
    x = torch.randn(batch, frames, N_MELS, device=dev, dtype=dt)
    tc = torch.randn(batch, frames, N_MELS, device=dev, dtype=dt)
    sc = torch.randn(batch, frames, N_MELS, device=dev, dtype=dt)
    t = torch.rand(batch, device=dev, dtype=torch.float32)
    mask = torch.zeros(batch, frames, device=dev, dtype=torch.bool)
    mask[:, frames - 32:] = True

    a = ref.fm_step(x, tc, sc, t, mask).float()
    b = trt_be.fm_step(x, tc, sc, t, mask).float()
    torch.cuda.synchronize(dev)

    diff = (a - b).abs()
    denom = a.abs().clamp(min=1e-3)
    print(f"shape          : {tuple(a.shape)}")
    print(f"max abs error  : {diff.max().item():.6f}")
    print(f"mean abs error : {diff.mean().item():.6f}")
    print(f"max rel error  : {(diff / denom).max().item():.6f}")
    # fp16 accumulation over a 16-layer network drifts; 2e-2 relative is the
    # band where output audio is indistinguishable.
    ok = diff.max().item() < 5e-2
    print("VERDICT        :", "OK" if ok else "SUSPICIOUS - inspect before serving")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Export + build DhVaani TensorRT engines")
    ap.add_argument("--max-batch", type=int, default=None,
                    help="Engine max batch (defaults to 2x engine.max_batch_size for CFG)")
    ap.add_argument("--fp32", action="store_true", help="Build in fp32 instead of fp16")
    ap.add_argument("--keep-onnx", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="Compare an existing engine against PyTorch and exit")
    ap.add_argument("--batch", type=int, default=4, help="validate: batch")
    ap.add_argument("--frames", type=int, default=384, help="validate: frames")
    args = ap.parse_args()

    if args.validate:
        return validate(dhv_settings, args.batch, args.frames)

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("tensorrt is not installed: pip install tensorrt-cu12", file=sys.stderr)
        return 1

    build_engines(dhv_settings, max_batch=args.max_batch,
                  fp16=not args.fp32, keep_onnx=args.keep_onnx)
    print("\nNow start the server with:\n"
          "  python -m flowtts.dhvaani.server --backend trt --ports 1\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
