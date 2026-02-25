#!/usr/bin/env python3
"""
Batch decode stress-test for FlowTTS TTSCodec.

Initialises ONE TTSCodec instance (model load + ONNX session warm-up),
then runs R rounds of N concurrent decode_async() calls against it.

The codec is never re-initialised between rounds — only the first round
pays the cold-start cost.  All subsequent rounds use the warm, pre-loaded
sessions.

IMPORTANT: run via run.sh or with the correct LD_LIBRARY_PATH so that
onnxruntime uses GPU (CUDAExecutionProvider needs libcudnn.so.9):
    LD_LIBRARY_PATH=<venv>/nvidia/cudnn/lib:... python test_concurrent_decode.py
Without it, ONNX falls back to CPU and is ~15x slower.

Metrics reported per round:
  • per-request latency (wall time from send→result)
  • actual GPU batch sizes dispatched (via monkey-patch)
  • GPU forward pass count (ideally 1 per round, split by gpu_chunk)

Usage:
    # 30 concurrent requests, 3 rounds, default bench dir
    python flowtts/test/test_concurrent_decode.py

    # 90 requests, 5 rounds, GPU chunk 100, parallel ONNX workers 4
    python flowtts/test/test_concurrent_decode.py \\
        --n-requests 90 --rounds 5 --gpu-chunk 100 --onnx-workers 4

    # Don't save WAVs
    python flowtts/test/test_concurrent_decode.py --no-save
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Allow running directly from any working directory
sys.path.insert(0, str(Path(__file__).parents[2]))

from flowtts.decoder.ncodec.codec import TTSCodec
from flowtts.decoder.decoder import tensor_to_wav, SAMPLE_RATE


def _check_onnx_gpu() -> None:
    """Warn loudly if onnxruntime will fall back to CPU (missing libcudnn.so.9)."""
    try:
        import onnxruntime as ort
        import tempfile, numpy as np
        providers = [("CUDAExecutionProvider", {"device_id": 0})]
        # A trivial model load tells us whether CUDA provider is actually active.
        # We check via get_available_providers — if CUDA not available it lists only CPU.
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" not in avail:
            print(
                "\n  WARNING: onnxruntime CUDAExecutionProvider not available — ONNX will run on CPU!\n"
                "  Fix: export LD_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH\n"
                "  Or launch via run.sh which sets this automatically.\n",
                flush=True,
            )
    except Exception:
        pass

# Default context tokens (same as FlowTtsSynthesizer._default_context)
DEFAULT_CONTEXT = (
    "<|context_token_3991|><|context_token_1250|><|context_token_2828|>"
    "<|context_token_3303|><|context_token_1187|><|context_token_3021|>"
    "<|context_token_355|><|context_token_3767|><|context_token_3663|>"
    "<|context_token_837|><|context_token_731|><|context_token_3656|>"
    "<|context_token_757|><|context_token_3360|><|context_token_3250|>"
    "<|context_token_3626|><|context_token_1244|><|context_token_526|>"
    "<|context_token_3829|><|context_token_205|><|context_token_1619|>"
    "<|context_token_268|><|context_token_4024|><|context_token_3375|>"
    "<|context_token_3032|><|context_token_2180|><|context_token_3278|>"
    "<|context_token_1609|><|context_token_3685|><|context_token_1359|>"
    "<|context_token_2817|><|context_token_3999|>"
)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _load_json_inputs(bench_root: Path) -> list[dict]:
    """Load all JSON bench records that have a valid audio_tokens field."""
    records: list[dict] = []
    search_dirs = sorted(bench_root.glob("bench_*")) if bench_root.is_dir() else [bench_root]
    if not search_dirs:
        search_dirs = [bench_root]
    for d in search_dirs:
        for jf in sorted(d.glob("*.json")):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            tokens = data.get("audio_tokens", "")
            if tokens and "<|speech_token_" in tokens:
                records.append({"source": jf.name, **data})
    return records


# ---------------------------------------------------------------------------
# Batch instrumentation: count how many GPU batches were dispatched
# ---------------------------------------------------------------------------

class _PatchedDecoder:
    """Thin wrapper around AudioDecoder that counts detokenize_batch calls."""

    def __init__(self, real_decoder):
        self._real = real_decoder
        self.batch_calls: list[int] = []  # batch sizes per call

    def reset(self) -> None:
        self.batch_calls.clear()

    def detokenize_batch(self, requests):
        self.batch_calls.append(len(requests))
        return self._real.detokenize_batch(requests)

    def detokenize(self, *args, **kwargs):
        return self._real.detokenize(*args, **kwargs)


# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------

async def _decode_one(
    idx: int,
    codec: TTSCodec,
    audio_tokens: str,
    context_tokens: str,
    out_dir: Optional[Path],
    source: str,
    round_idx: int,
) -> dict:
    t0 = time.perf_counter()
    try:
        wav_tensor = await codec.decode_async(audio_tokens, context_tokens)
        elapsed = time.perf_counter() - t0
        decoded = tensor_to_wav(wav_tensor, sample_rate=SAMPLE_RATE)
        ok = True
        error = None
        wav_bytes = decoded.wav_bytes
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        ok = False
        error = str(exc)
        wav_bytes = b""

    wav_path = ""
    if ok and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(source).stem if source else f"req{idx:04d}"
        wav_path = str(out_dir / f"r{round_idx:02d}_req{idx:04d}_{stem}.wav")
        Path(wav_path).write_bytes(wav_bytes)

    status = "OK" if ok else f"FAIL: {error}"
    print(
        f"  [r{round_idx} #{idx:>3d}] {status}  decode={elapsed*1000:.0f}ms  "
        f"wav={len(wav_bytes)}B",
        flush=True,
    )
    return {
        "round": round_idx,
        "idx": idx,
        "source": source,
        "ok": ok,
        "decode_s": elapsed,
        "wav_bytes": len(wav_bytes),
        "error": error,
    }


# ---------------------------------------------------------------------------
# One round
# ---------------------------------------------------------------------------

async def _run_round(
    round_idx: int,
    n_requests: int,
    codec: TTSCodec,
    patcher: _PatchedDecoder,
    records: list[dict],
    out_dir: Optional[Path],
) -> tuple[list[dict], float]:
    chosen = [random.choice(records) for _ in range(n_requests)]
    patcher.reset()

    print(f"\n── Round {round_idx + 1}  ({n_requests} concurrent requests) {'─'*30}", flush=True)
    t_wall = time.perf_counter()

    tasks = [
        asyncio.create_task(
            _decode_one(
                idx=i,
                codec=codec,
                audio_tokens=chosen[i]["audio_tokens"],
                context_tokens=DEFAULT_CONTEXT,
                out_dir=out_dir,
                source=chosen[i].get("source", ""),
                round_idx=round_idx + 1,
            )
        )
        for i in range(n_requests)
    ]
    results = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - t_wall

    ok_count = sum(1 for r in results if r["ok"])
    print(
        f"  Round {round_idx + 1} done — {ok_count}/{n_requests} ok  "
        f"wall={wall_s*1000:.0f}ms  "
        f"GPU calls={len(patcher.batch_calls)}  "
        f"batch_sizes={patcher.batch_calls}",
        flush=True,
    )
    return list(results), wall_s


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(all_results: list[list[dict]], all_walls: list[float], n_requests: int) -> None:
    flat = [r for rnd in all_results for r in rnd]
    ok   = [r for r in flat if r["ok"]]
    fail = [r for r in flat if not r["ok"]]
    n_rounds = len(all_results)
    width = 72

    print()
    print("═" * width)
    print(
        f"  BATCH DECODE SUMMARY  "
        f"{len(ok)}/{len(flat)} ok   {len(fail)} failed   "
        f"{n_rounds} round(s)"
    )
    print("═" * width)

    if fail:
        print("\n  Failures:")
        for r in fail:
            print(f"    [r{r['round']} #{r['idx']:>3d}] {r.get('error', '?')}")

    if ok:
        vals = sorted(r["decode_s"] * 1000 for r in ok)

        def _pct(p: float) -> float:
            k = (len(vals) - 1) * p / 100
            lo = int(k)
            hi = min(lo + 1, len(vals) - 1)
            return vals[lo] + (k - lo) * (vals[hi] - vals[lo])

        avg = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(
            f"\n  Per-request latency across all rounds (ms):\n"
            f"    avg={avg:.0f}  min={min(vals):.0f}  max={max(vals):.0f}  "
            f"std={std:.0f}  p50={_pct(50):.0f}  p95={_pct(95):.0f}  p99={_pct(99):.0f}"
        )

        # Per-round breakdown
        print(f"\n  {'RND':>4}  {'OK':>4}  {'wall':>8}  {'avg':>8}  {'min':>6}  {'max':>6}  {'req/s':>7}")
        print("  " + "-" * 56)
        for rnd_idx, (rnd_results, wall_s) in enumerate(zip(all_results, all_walls)):
            rnd_ok = [r for r in rnd_results if r["ok"]]
            if rnd_ok:
                t_vals = [r["decode_s"] * 1000 for r in rnd_ok]
                rps = len(rnd_ok) / wall_s
                print(
                    f"  {rnd_idx+1:>4}  {len(rnd_ok):>4}  "
                    f"{wall_s*1000:>6.0f}ms  "
                    f"{statistics.mean(t_vals):>6.0f}ms  "
                    f"{min(t_vals):>4.0f}ms  "
                    f"{max(t_vals):>4.0f}ms  "
                    f"{rps:>6.1f}"
                )
            else:
                print(f"  {rnd_idx+1:>4}     0  (all failed)")

    total_wall = sum(all_walls)
    total_ok   = len(ok)
    if total_wall > 0:
        print(f"\n  Overall throughput: {total_ok/total_wall:.1f} req/s  ({total_ok} ok in {total_wall*1000:.0f}ms)")

    print("═" * width + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_test(
    n_requests: int,
    n_rounds: int,
    gpu_chunk: int,
    onnx_workers: int,
    use_trt: bool,
    bench_root: Path,
    out_dir: Optional[Path],
) -> None:
    _check_onnx_gpu()

    # Load inputs
    print(f"\nLoading JSON inputs from {bench_root} …")
    records = _load_json_inputs(bench_root)
    if not records:
        print(f"ERROR: no JSON files with audio_tokens found under {bench_root}")
        print("       Run benchmark.py first or adjust --bench-dir.")
        sys.exit(1)
    print(f"  {len(records)} records available")

    # ── Init codec ONCE ───────────────────────────────────────────────────────
    print(
        f"\nInitialising TTSCodec "
        f"(gpu_chunk={gpu_chunk}, onnx_workers={onnx_workers}, use_trt={use_trt}) …",
        flush=True,
    )
    t_init = time.perf_counter()
    codec = TTSCodec(
        max_batch_size=n_requests,
        batch_timeout_ms=50.0,   # generous window — lets all N requests arrive before dispatch
        gpu_chunk_size=gpu_chunk,
        onnx_workers=onnx_workers,
        use_trt=use_trt,
    )
    init_ms = (time.perf_counter() - t_init) * 1000
    print(f"  codec ready  ({init_ms:.0f}ms — not counted in round timings)\n", flush=True)

    # Instrument the inner decoder so we can count actual GPU batch dispatches
    patcher = _PatchedDecoder(codec.audio_decoder)
    codec.audio_decoder = patcher

    # ── Rounds ────────────────────────────────────────────────────────────────
    all_results: list[list[dict]] = []
    all_walls:   list[float]      = []

    for r in range(n_rounds):
        results, wall_s = await _run_round(r, n_requests, codec, patcher, records, out_dir)
        all_results.append(results)
        all_walls.append(wall_s)

    _print_summary(all_results, all_walls, n_requests)

    if out_dir:
        print(f"WAV files saved to: {out_dir}/\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concurrent decode_async() stress-test — codec initialised once, N rounds run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-requests", type=int, default=30,
        help="Concurrent decode requests per round (default: 30)",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Number of rounds to run against the warm codec (default: 3)",
    )
    parser.add_argument(
        "--gpu-chunk", type=int, default=50,
        help="GPU chunk size for AudioDecoder (default: 50)",
    )
    parser.add_argument(
        "--onnx-workers", type=int, default=2,
        help="Parallel ONNX worker threads (default: 2)",
    )
    parser.add_argument(
        "--use-trt", action="store_true", default=False,
        help="Compile decoder with TensorRT FP16 (requires torch-tensorrt)",
    )
    parser.add_argument(
        "--bench-dir", type=Path,
        default=Path(__file__).parents[2] / "test",
        help="Root directory containing bench_*/ subdirs (default: test/)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write decoded WAV files to disk",
    )
    args = parser.parse_args()

    out_dir: Optional[Path] = None
    if not args.no_save:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parents[2] / "test" / f"batch_decode_{tag}"
        print(f"WAV outputs → {out_dir}/")

    asyncio.run(run_test(
        n_requests=args.n_requests,
        n_rounds=args.rounds,
        gpu_chunk=args.gpu_chunk,
        onnx_workers=args.onnx_workers,
        use_trt=args.use_trt,
        bench_root=args.bench_dir,
        out_dir=out_dir,
    ))


if __name__ == "__main__":
    main()
