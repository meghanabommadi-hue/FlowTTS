#!/usr/bin/env python3
"""
Spawn N independent AudioDecoder instances and send exactly one request to
each, all in parallel.

codec i → request i, fired concurrently via ThreadPoolExecutor.
Inputs are drawn at random from the JSON benchmark files in test/bench_*/.

Usage:
    # 3 codec instances, 3 parallel requests (default)
    python flowtts/test/test_codec_instances.py

    # 10 instances / 10 parallel requests
    python flowtts/test/test_codec_instances.py --n-codecs 10

    # No WAV output
    python flowtts/test/test_codec_instances.py --n-codecs 5 --no-save

    # Custom bench directory
    python flowtts/test/test_codec_instances.py --bench-dir test/bench_20260220_160328
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

# Allow running directly from any working directory
sys.path.insert(0, str(Path(__file__).parents[2]))

from flowtts.decoder.decoder import AudioDecoder, CONTEXT_TOKENS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_json_inputs(bench_root: Path) -> list[dict]:
    """Return all JSON records with a valid audio_tokens field."""
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


def _decode_one(
    codec_id: int,
    codec: AudioDecoder,
    record: dict,
    out_dir: Optional[Path],
) -> dict:
    """Decode one record on its dedicated codec instance."""
    audio_tokens: str = record["audio_tokens"]
    source: str = record.get("source", "?")

    t0 = time.perf_counter()
    try:
        result = codec.decode_to_wav(audio_tokens, CONTEXT_TOKENS)
        elapsed = time.perf_counter() - t0
        wav_bytes = result.wav_bytes
        sample_rate = result.sample_rate
        ok = True
        error = None
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        wav_bytes = b""
        sample_rate = 0
        ok = False
        error = str(exc)

    wav_path = ""
    if ok and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(source).stem
        wav_path = str(out_dir / f"codec{codec_id:04d}_{stem}.wav")
        Path(wav_path).write_bytes(wav_bytes)

    return {
        "codec_id": codec_id,
        "source": source,
        "ok": ok,
        "decode_s": elapsed,
        "wav_bytes": len(wav_bytes),
        "sample_rate": sample_rate,
        "wav_path": wav_path,
        "error": error,
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def run(
    n_codecs: int,
    bench_root: Path,
    out_dir: Optional[Path],
) -> list[dict]:
    # 1. Collect inputs
    print(f"\nCollecting JSON inputs from {bench_root} …")
    all_records = _collect_json_inputs(bench_root)
    if not all_records:
        raise RuntimeError(
            f"No JSON benchmark files with audio_tokens found under {bench_root}. "
            "Run benchmark.py first or pass --bench-dir pointing at a bench_* folder."
        )
    print(f"  {len(all_records)} records available\n")

    # One random record per codec instance
    job_records = [random.choice(all_records) for _ in range(n_codecs)]

    # 2. Initialise N codec instances sequentially (model load is serial)
    print(f"Initialising {n_codecs} codec instance(s) …")
    codecs: list[AudioDecoder] = []
    for i in range(n_codecs):
        t0 = time.perf_counter()
        codecs.append(AudioDecoder())
        print(f"  [codec {i}] ready in {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)
    print()

    # 3. Fire all N requests in parallel — codec i handles request i exclusively
    print(f"Dispatching {n_codecs} requests in parallel (1 per codec instance) …\n")
    results: list[dict] = [{}] * n_codecs
    t_wall = time.perf_counter()
    completed = 0

    with ThreadPoolExecutor(max_workers=n_codecs) as executor:
        future_to_idx = {
            executor.submit(_decode_one, i, codecs[i], job_records[i], out_dir): i
            for i in range(n_codecs)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                res = future.result()
            except Exception as exc:
                res = {
                    "codec_id": i,
                    "source": job_records[i].get("source", "?"),
                    "ok": False,
                    "decode_s": 0.0,
                    "wav_bytes": 0,
                    "sample_rate": 0,
                    "wav_path": "",
                    "error": str(exc),
                }
            results[i] = res
            completed += 1
            status = "OK" if res["ok"] else f"FAIL: {res['error']}"
            print(
                f"  codec {res['codec_id']:>4d}  {status}  "
                f"decode={res['decode_s']*1000:.0f}ms  "
                f"({completed}/{n_codecs})",
                flush=True,
            )

    wall = time.perf_counter() - t_wall
    print(f"\n  Wall time: {wall*1000:.0f}ms  ({n_codecs/wall:.2f} jobs/s)\n")
    return results


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    ok = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]
    width = 68

    print("═" * width)
    print(f"  CODEC INSTANCE TEST  {len(ok)}/{len(results)} ok   {len(fail)} failed")
    print("═" * width)

    if fail:
        print("\n  Failures:")
        for r in fail:
            print(f"    codec {r['codec_id']:>4d}  {r.get('error', '?')}")

    if ok:
        vals = sorted(r["decode_s"] * 1000 for r in ok)
        avg = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(
            f"\n  decode latency (ms):  "
            f"avg={avg:.0f}  min={min(vals):.0f}  max={max(vals):.0f}  std={std:.0f}"
        )

        print(f"\n  {'CODEC':>6}  {'decode':>9}  {'bytes':>9}  SOURCE")
        print("  " + "-" * (width - 2))
        for r in sorted(ok, key=lambda x: x["codec_id"]):
            print(
                f"  {r['codec_id']:>6}  {r['decode_s']*1000:>7.0f}ms  "
                f"{r['wav_bytes']:>9}  {r['source']}"
            )

    print("═" * width + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Spawn N codec instances and send exactly one request to each, in parallel.\n"
            "codec i handles request i — no sharing between instances."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-codecs", type=int, default=3,
        help="Number of codec instances (= number of parallel requests) (default: 3)",
    )
    parser.add_argument(
        "--bench-dir", type=Path,
        default=Path(__file__).parents[2] / "test",
        help="Root directory containing bench_*/ subdirs with JSON files",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write decoded WAV files to disk",
    )
    args = parser.parse_args()

    out_dir: Optional[Path] = None
    if not args.no_save:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parents[2] / "test" / f"codec_instances_{tag}"
        print(f"WAV outputs → {out_dir}/")

    results = run(
        n_codecs=args.n_codecs,
        bench_root=args.bench_dir,
        out_dir=out_dir,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
