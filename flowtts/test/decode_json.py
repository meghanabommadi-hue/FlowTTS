#!/usr/bin/env python3
"""
Decode audio_tokens from saved JSON benchmark files using multiple
AudioDecoder instances in parallel — one codec per request, all at once.

Mirrors exactly what TTSIntegration/decoder.py does:
    tts = TTSCodec()
    wav = tts.decode(audio_tokens, context_tokens)

Usage:
    python flowtts/test/decode_json.py                        # 3 random JSONs
    python flowtts/test/decode_json.py --n 8                  # 8 in parallel
    python flowtts/test/decode_json.py --file path/to/x.json  # specific file
    python flowtts/test/decode_json.py --no-save              # skip WAV output
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from flowtts.decoder.decoder import CONTEXT_TOKENS


# ── Worker (runs in a subprocess — own CUDA context) ─────────────────────────

def _decode_one(job: dict) -> dict:
    """
    Executed in a dedicated subprocess.
    Imports AudioDecoder locally so each process gets its own CUDA context,
    exactly like running TTSIntegration/decoder.py N times in parallel.
    """
    from flowtts.decoder.decoder import AudioDecoder

    codec  = AudioDecoder()
    source = job["source"]
    audio_tokens   = job["audio_tokens"]
    context_tokens = job.get("context_tokens") or CONTEXT_TOKENS
    out_path       = job.get("out_path")

    t0 = time.perf_counter()
    try:
        result   = codec.decode_to_wav(audio_tokens, context_tokens)
        elapsed  = time.perf_counter() - t0

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(result.wav_bytes)

        return {
            "source":   source,
            "ok":       True,
            "decode_s": elapsed,
            "wav_bytes": len(result.wav_bytes),
            "out_path":  out_path or "",
        }
    except Exception as exc:
        return {
            "source":   source,
            "ok":       False,
            "decode_s": time.perf_counter() - t0,
            "error":    str(exc),
        }


# ── Input collection ──────────────────────────────────────────────────────────

def _collect(bench_root: Path, limit: int) -> list[dict]:
    records: list[dict] = []
    dirs = sorted(bench_root.glob("bench_*")) if bench_root.is_dir() else [bench_root]
    if not dirs:
        dirs = [bench_root]
    for d in dirs:
        for jf in sorted(d.glob("*.json")):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            tok = data.get("audio_tokens", "")
            if tok and "<|speech_token_" in tok:
                records.append({
                    "source":         jf.name,
                    "audio_tokens":   tok,
                    "context_tokens": data.get("context_tokens") or CONTEXT_TOKENS,
                    "text":           data.get("text", ""),
                    "_path":          str(jf),
                })
    random.shuffle(records)
    return records[:limit]


# ── Runner ────────────────────────────────────────────────────────────────────

def run(jobs: list[dict]) -> list[dict]:
    n = len(jobs)
    print(f"Spawning {n} subprocess(es) — one per decode job, all in parallel …\n")

    results: list[dict] = []
    t_wall = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n) as pool:
        future_to_job = {pool.submit(_decode_one, j): j for j in jobs}
        for future in as_completed(future_to_job):
            try:
                res = future.result()
            except Exception as exc:
                j = future_to_job[future]
                res = {"source": j["source"], "ok": False, "error": str(exc), "decode_s": 0}
            results.append(res)
            status = "OK" if res["ok"] else f"FAIL: {res.get('error')}"
            print(
                f"  {res['source']:40s}  {status}"
                + (f"  {res['decode_s']*1000:.0f}ms  {res.get('wav_bytes',0)} bytes" if res["ok"] else ""),
                flush=True,
            )

    wall = time.perf_counter() - t_wall
    print(f"\n  Wall time: {wall*1000:.0f}ms\n")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode audio_tokens from JSON files using parallel subprocesses."
    )
    parser.add_argument("--n", type=int, default=3,
                        help="Number of JSON files to decode in parallel (default: 3)")
    parser.add_argument("--file", type=Path, default=None,
                        help="Decode a single specific JSON file")
    parser.add_argument("--bench-dir", type=Path,
                        default=Path(__file__).parents[2] / "test",
                        help="Root dir containing bench_*/ subdirs (default: test/)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip writing WAV files")
    args = parser.parse_args()

    out_dir: Path | None = None
    if not args.no_save:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parents[2] / "test" / f"decoded_{tag}"
        print(f"WAV outputs → {out_dir}/\n")

    # Build job list
    if args.file:
        data = json.loads(args.file.read_text(encoding="utf-8"))
        records = [{
            "source":         args.file.name,
            "audio_tokens":   data["audio_tokens"],
            "context_tokens": data.get("context_tokens") or CONTEXT_TOKENS,
            "text":           data.get("text", ""),
        }]
    else:
        records = _collect(args.bench_dir, args.n)
        if not records:
            print(f"No JSON files found under {args.bench_dir}")
            sys.exit(1)

    # Attach output paths
    jobs = []
    for i, rec in enumerate(records):
        out_path = None
        if out_dir:
            stem = Path(rec["source"]).stem
            out_path = str(out_dir / f"{i:04d}_{stem}.wav")
        jobs.append({**rec, "out_path": out_path})

    # Print what we're about to decode
    print(f"Decoding {len(jobs)} file(s):")
    for j in jobs:
        preview = j["text"][:60] if j.get("text") else j["source"]
        print(f"  {j['source']}  {preview!r}")
    print()

    results = run(jobs)

    ok   = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    print(f"Done — {len(ok)}/{len(results)} ok" + (f"  {len(fail)} failed" if fail else ""))
    if fail:
        for r in fail:
            print(f"  FAIL: {r['source']}  {r.get('error')}")


if __name__ == "__main__":
    main()
