"""Pipeline position: ACCELERATION (build time) — build the TensorRT-LLM engine.

Role in pipeline:
  The faithful upstream build, for boxes that have TensorRT-LLM installed:

      HF checkpoint (k2-fsa/OmniVoice)
        └─ convert_checkpoint.py     → rank0.safetensors  (+ FP8 scales with --fp8)
        └─ patch/omnivoice/          → registered into tensorrt_llm.models
        └─ trtllm-build              → rank0.engine
             └─ flowtts.trt.runtime.TRTLLMBackbone

  Same three steps as github.com/tlitech/omnivoice-trtllm's
  ``build_trtllm_engine`` / ``build_trtllm_fp8_engine``, minus Modal.

  On a box without TensorRT-LLM — which includes the NGC container this service
  normally runs in, where a global pip constraint pins torch — use
  ``python -m flowtts.trt.build_trt`` instead. It produces an engine with the
  same I/O contract using the plain TensorRT that is already installed.

Usage:
    python -m flowtts.trt.build_trtllm                  # FP16
    python -m flowtts.trt.build_trtllm --fp8            # FP8 (Ada/Hopper)
    python -m flowtts.trt.build_trtllm --max-batch 64 --max-seq 2048
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_PATCH_DIR = Path(__file__).parent / "patch" / "omnivoice"
_REGISTER_SNIPPET = (
    "\nfrom .omnivoice.model import OmniVoice\n"
    "MODEL_MAP['OmniVoice'] = OmniVoice\n"
)


def _tensorrt_llm_package_dir() -> Path:
    """Locate the installed tensorrt_llm package without importing it.

    Importing tensorrt_llm initializes CUDA and MPI, which is slow and can fail
    on a shared box; ``pip show`` tells us the location for free.
    """
    out = subprocess.run(
        [sys.executable, "-m", "pip", "show", "tensorrt_llm"],
        capture_output=True, text=True, check=True,
    ).stdout
    location = next(
        (line.split(": ", 1)[1] for line in out.splitlines() if line.startswith("Location:")),
        None,
    )
    if location is None:
        raise RuntimeError("tensorrt_llm is not installed")
    return Path(location) / "tensorrt_llm"


def register_model() -> Path:
    """Copy the OmniVoice network definition into tensorrt_llm and register it."""
    trtllm = _tensorrt_llm_package_dir()
    target = trtllm / "models" / "omnivoice"

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(_PATCH_DIR, target)
    # The vendored package ships this repo's own __init__; inside tensorrt_llm
    # it must be the upstream (empty) one or the relative imports break.
    (target / "__init__.py").write_text("")

    init_file = trtllm / "models" / "__init__.py"
    if "OmniVoice" not in init_file.read_text():
        with init_file.open("a") as fh:
            fh.write(_REGISTER_SNIPPET)
    return target


def convert(model_dir: str | Path, out_dir: str | Path, *, fp8: bool = False) -> Path:
    """Run the vendored checkpoint converter."""
    cmd = [
        sys.executable, str(Path(__file__).parent / "convert_checkpoint.py"),
        "--model_dir", str(model_dir),
        "--output_dir", str(out_dir),
    ]
    if fp8:
        cmd.append("--fp8")
    subprocess.run(cmd, check=True)
    return Path(out_dir)


def trtllm_build(
    checkpoint_dir: str | Path,
    engine_dir: str | Path,
    *,
    max_batch: int = 64,
    fp8: bool = False,
) -> Path:
    """Compile the converted checkpoint with ``trtllm-build``."""
    cmd = [
        "trtllm-build",
        "--checkpoint_dir", str(checkpoint_dir),
        "--output_dir", str(engine_dir),
        "--max_batch_size", str(max_batch),
        # The engine takes padded [B, S, H] batches straight from OmniVoice's
        # generation loop; packed input would need the loop rewritten.
        "--remove_input_padding", "disable",
    ]
    if fp8:
        cmd += ["--gemm_plugin", "fp8"]
    subprocess.run(cmd, check=True)
    return Path(engine_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the OmniVoice TensorRT-LLM engine")
    ap.add_argument("--model-dir", default=None, help="OmniVoice HF weights (default: config)")
    ap.add_argument("--engine-dir", default=None, help="output engine dir (default: config)")
    ap.add_argument("--checkpoint-dir", default=None, help="intermediate TRT-LLM checkpoint dir")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--fp8", action="store_true", help="FP8 E4M3 weights (Ada/Hopper)")
    ap.add_argument("--skip-convert", action="store_true", help="reuse an existing checkpoint")
    args = ap.parse_args()

    from flowtts.core.config import resolve_model_source, settings

    model_dir = args.model_dir or resolve_model_source()
    engine_dir = Path(args.engine_dir or settings.omnivoice.trtllm_engine_dir
                      or "engines/omnivoice-trtllm")
    ckpt_dir = Path(args.checkpoint_dir or (engine_dir.parent / f"{engine_dir.name}-ckpt"))

    print(f"[build_trtllm] model={model_dir}", flush=True)
    if not args.skip_convert:
        print(f"[build_trtllm] converting checkpoint → {ckpt_dir} (fp8={args.fp8})", flush=True)
        convert(model_dir, ckpt_dir, fp8=args.fp8)

    print("[build_trtllm] registering OmniVoice in tensorrt_llm.models", flush=True)
    print(f"[build_trtllm]   → {register_model()}", flush=True)

    print(f"[build_trtllm] building engine → {engine_dir}", flush=True)
    trtllm_build(ckpt_dir, engine_dir, max_batch=args.max_batch, fp8=args.fp8)
    print(f"[build_trtllm] done: {engine_dir / 'rank0.engine'}", flush=True)


if __name__ == "__main__":
    main()
