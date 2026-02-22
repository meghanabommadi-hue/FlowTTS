#!/usr/bin/env python3
"""
Concurrent codec decode stress-test: one codec instance → one request, strictly.
Codec pool stays alive across multiple runs — no re-initialisation between rounds.

Spawns exactly N AudioDecoder instances once, then fires R rounds of N parallel
decode jobs. codec i always handles request i within each round.
Inputs are drawn at random from the JSON benchmark files in test/bench_*/.

Usage:
    # 3 codec instances, 3 rounds (defaults)
    python flowtts/test/test_concurrent_codec.py

    # 10 instances, 5 rounds
    python flowtts/test/test_concurrent_codec.py --n-codecs 10 --rounds 5

    # No WAV output, specific bench directory
    python flowtts/test/test_concurrent_codec.py --no-save --bench-dir test/bench_20260220_160328

    # 30 instances, 10 rounds
    python flowtts/test/test_concurrent_codec.py --n-codecs 30 --rounds 10
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


# ── Input collection ──────────────────────────────────────────────────────────

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


# ── Codec pool ────────────────────────────────────────────────────────────────

class CodecPool:
    """
    N AudioDecoder instances initialised once and kept alive for all rounds.
    codec i is dedicated to request i within each round — no sharing.
    """

    def __init__(self, n: int) -> None:
        self._instances: list[AudioDecoder] = []
        print(f"Initialising {n} codec instance(s) …")
        for i in range(n):
            t0 = time.perf_counter()
            codec = AudioDecoder()
            elapsed = time.perf_counter() - t0
            self._instances.append(codec)
            print(f"  [codec {i}] ready in {elapsed*1000:.0f}ms", flush=True)
        print(f"  Pool of {n} live codec(s) ready — will reuse across all rounds.\n")

    def get(self, idx: int) -> AudioDecoder:
        return self._instances[idx]

    @property
    def size(self) -> int:
        return len(self._instances)


# ── Per-job worker ────────────────────────────────────────────────────────────

def _job(
    round_idx: int,
    codec_idx: int,
    codec: AudioDecoder,
    record: dict,
    out_dir: Optional[Path],
) -> dict:
    """Execute one decode job on its dedicated (live) codec instance."""
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
        wav_path = str(out_dir / f"r{round_idx:03d}_codec{codec_idx:04d}_{stem}.wav")
        Path(wav_path).write_bytes(wav_bytes)

    return {
        "round": round_idx,
        "codec_idx": codec_idx,
        "source": source,
        "ok": ok,
        "decode_s": elapsed,
        "wav_bytes": len(wav_bytes),
        "sample_rate": sample_rate,
        "wav_path": wav_path,
        "error": error,
    }


# ── Single round ──────────────────────────────────────────────────────────────

def _run_round(
    round_idx: int,
    pool: CodecPool,
    all_records: list[dict],
    out_dir: Optional[Path],
) -> tuple[list[dict], float]:
    """Fire one round: N parallel jobs on the N live codec instances."""
    n = pool.size
    job_records = [random.choice(all_records) for _ in range(n)]

    results: list[dict] = [{}] * n
    completed = 0
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n) as executor:
        future_to_idx = {
            executor.submit(_job, round_idx, i, pool.get(i), job_records[i], out_dir): i
            for i in range(n)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                res = future.result()
            except Exception as exc:
                res = {
                    "round": round_idx,
                    "codec_idx": i,
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
                f"    codec {res['codec_idx']:>4d}  {status}  "
                f"decode={res['decode_s']*1000:.0f}ms  "
                f"({completed}/{n})",
                flush=True,
            )

    wall = time.perf_counter() - t0
    return results, wall


# ── Main runner ───────────────────────────────────────────────────────────────

def run(
    n_codecs: int,
    n_rounds: int,
    bench_root: Path,
    out_dir: Optional[Path],
) -> list[list[dict]]:
    # 1. Collect inputs
    print(f"\nCollecting JSON inputs from {bench_root} …")
    all_records = _collect_json_inputs(bench_root)
    if not all_records:
        raise RuntimeError(
            f"No JSON benchmark files with audio_tokens found under {bench_root}. "
            "Run benchmark.py first or adjust --bench-dir."
        )
    print(f"  {len(all_records)} records available\n")

    # 2. Build pool ONCE — stays alive for all rounds
    pool = CodecPool(n_codecs)

    # 3. Run R rounds against the same live pool
    all_round_results: list[list[dict]] = []
    for r in range(n_rounds):
        print(f"── Round {r+1}/{n_rounds} {'─'*40}", flush=True)
        results, wall = _run_round(r, pool, all_records, out_dir)
        ok_count = sum(1 for res in results if res.get("ok"))
        print(
            f"  Round {r+1} done — {ok_count}/{n_codecs} ok  "
            f"wall={wall*1000:.0f}ms\n",
            flush=True,
        )
        all_round_results.append(results)

    return all_round_results


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_summary(all_round_results: list[list[dict]]) -> None:
    flat = [r for round_res in all_round_results for r in round_res]
    ok = [r for r in flat if r.get("ok")]
    fail = [r for r in flat if not r.get("ok")]
    n_rounds = len(all_round_results)
    width = 72

    print("═" * width)
    print(
        f"  CONCURRENT CODEC TEST  "
        f"{len(ok)}/{len(flat)} ok   {len(fail)} failed   "
        f"{n_rounds} round(s)"
    )
    print("═" * width)

    if fail:
        print("\n  Failures:")
        for r in fail:
            print(f"    round {r['round']+1}  codec {r['codec_idx']:>4d}  {r.get('error', '?')}")

    if ok:
        vals = sorted(r["decode_s"] * 1000 for r in ok)

        def _pct(p: float) -> float:
            k = (len(vals) - 1) * p / 100
            f = int(k)
            c = min(f + 1, len(vals) - 1)
            return vals[f] + (k - f) * (vals[c] - vals[f])

        avg = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(
            f"\n  decode latency (ms) across all rounds:\n"
            f"    avg={avg:.0f}  min={min(vals):.0f}  max={max(vals):.0f}  "
            f"std={std:.0f}  p50={_pct(50):.0f}  p95={_pct(95):.0f}  p99={_pct(99):.0f}"
        )

        # Per-round summary
        print(f"\n  {'ROUND':>6}  {'OK':>4}  {'avg decode':>11}  {'min':>7}  {'max':>7}")
        print("  " + "-" * 44)
        for r_idx, round_res in enumerate(all_round_results):
            r_ok = [r for r in round_res if r.get("ok")]
            if r_ok:
                t_vals = [r["decode_s"] * 1000 for r in r_ok]
                print(
                    f"  {r_idx+1:>6}  {len(r_ok):>4}  "
                    f"{statistics.mean(t_vals):>9.0f}ms  "
                    f"{min(t_vals):>5.0f}ms  "
                    f"{max(t_vals):>5.0f}ms"
                )
            else:
                print(f"  {r_idx+1:>6}     0  (all failed)")

        # Per-codec summary across all rounds
        by_codec: dict[int, list[dict]] = {}
        for r in ok:
            by_codec.setdefault(r["codec_idx"], []).append(r)

        print(f"\n  {'CODEC':>6}  {'RUNS':>5}  {'avg decode':>11}  {'min':>7}  {'max':>7}")
        print("  " + "-" * 44)
        for cid in sorted(by_codec):
            reqs = by_codec[cid]
            t_vals = [r["decode_s"] * 1000 for r in reqs]
            print(
                f"  {cid:>6}  {len(reqs):>5}  "
                f"{statistics.mean(t_vals):>9.0f}ms  "
                f"{min(t_vals):>5.0f}ms  "
                f"{max(t_vals):>5.0f}ms"
            )

    print("═" * width + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Spawn N codec instances once, then run R rounds of N parallel requests.\n"
            "The pool stays alive — no re-init between rounds.\n"
            "codec i handles request i in every round."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-codecs", type=int, default=3,
        help="Number of codec instances (= requests per round) (default: 3)",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Number of rounds to run against the live pool (default: 3)",
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
        out_dir = Path(__file__).parents[2] / "test" / f"concurrent_codec_{tag}"
        print(f"WAV outputs → {out_dir}/")

    all_results = run(
        n_codecs=args.n_codecs,
        n_rounds=args.rounds,
        bench_root=args.bench_dir,
        out_dir=out_dir,
    )
    print_summary(all_results)


if __name__ == "__main__":
    main()
