#!/usr/bin/env python3
"""
Generate WAV files for sentences from a parquet file.

Writes WAVs to <voice>_audio/ named by sha256(text).wav.
Writes generate_progress.json after every sentence so push_to_hf.py
can start in parallel while this is still running.

Resume-safe: skips sentences whose sha256.wav already exists on disk.

Usage:
    python generate_wavs.py --voice simran
    python generate_wavs.py --voice tara --port 8766 --concurrency 8
    python generate_wavs.py --voice simran --min-freq 3 --parquet custom.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import websockets

HERE = Path(__file__).parent

DEFAULT_PARQUET = str(HERE / "normalized_sentences.parquet")
PROGRESS_FILE_TMPL = "{voice}_generate_progress.json"


def sha256_filename(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() + ".wav"


def load_sentences(parquet_path: str, min_freq: int) -> list[str]:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    filtered = df[df["frequency"] > min_freq].sort_values("frequency", ascending=False)
    sentences = filtered["text"].tolist()
    print(f"[INFO] Total unique sentences          : {len(df):,}")
    print(f"[INFO] Sentences with frequency > {min_freq}   : {len(sentences):,}")
    return sentences


def load_progress(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"done": 0, "skipped": 0, "errors": 0, "total": 0, "finished": False}


def save_progress(path: Path, done: int, skipped: int, errors: int, total: int, finished: bool = False) -> None:
    path.write_text(json.dumps({
        "done":     done,
        "skipped":  skipped,
        "errors":   errors,
        "total":    total,
        "finished": finished,
        "updated":  time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))


async def synthesize_one(text: str, port: int, semaphore: asyncio.Semaphore) -> bytes:
    call_id = str(uuid.uuid4())
    url = f"ws://localhost:{port}/ws/{call_id}"
    async with semaphore:
        async with websockets.connect(url, open_timeout=10, max_size=100 * 1024 * 1024) as ws:
            await ws.send(json.dumps({
                "type":           "synthesize",
                "call_id":        call_id,
                "text_id":        str(uuid.uuid4()),
                "text":           text,
                "pre_normalized": True,
            }))
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("error", "server error"))
            wav_data = await ws.recv()
            if isinstance(wav_data, str):
                wav_data = wav_data.encode()
            return wav_data


async def run_generation(
    sentences: list[str],
    port: int,
    concurrency: int,
    out_dir: Path,
    progress_path: Path,
) -> None:
    total = len(sentences)
    semaphore = asyncio.Semaphore(concurrency)

    done = 0
    skipped = 0
    errors = 0
    t_start = time.perf_counter()

    # Pre-count already-done to resume counters accurately
    for text in sentences:
        if (out_dir / sha256_filename(text)).exists():
            skipped += 1

    save_progress(progress_path, done, skipped, errors, total)

    async def process(idx: int, text: str) -> None:
        nonlocal done, skipped, errors

        wav_path = out_dir / sha256_filename(text)
        if wav_path.exists():
            # Already counted above — just log silently
            return

        try:
            t0 = time.perf_counter()
            wav_bytes = await synthesize_one(text, port, semaphore)
            if not wav_bytes:
                raise RuntimeError("empty WAV")
            wav_path.write_bytes(wav_bytes)
            done += 1
            elapsed = time.perf_counter() - t0
            total_done = done + skipped + errors
            rate = done / max(time.perf_counter() - t_start, 1)
            remaining = total - total_done
            eta_s = int(remaining / rate) if rate > 0 else 0
            print(
                f"  [{total_done}/{total}] OK {elapsed:.1f}s  "
                f"rate={rate:.2f}/s  ETA={eta_s//3600}h{(eta_s%3600)//60}m  "
                f"{text[:60]!r}",
                flush=True,
            )
        except Exception as exc:
            errors += 1
            total_done = done + skipped + errors
            print(f"  [{total_done}/{total}] ERROR {exc}  {text[:50]!r}", flush=True)

        # Save progress after every completed sentence
        save_progress(progress_path, done, skipped, errors, total)

    await asyncio.gather(*[process(i, t) for i, t in enumerate(sentences, 1)])

    save_progress(progress_path, done, skipped, errors, total, finished=True)
    elapsed = time.perf_counter() - t_start
    print(f"\n[DONE] done={done:,}  skipped={skipped:,}  errors={errors:,}  time={elapsed:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WAV files for sentences")
    parser.add_argument("--voice",       choices=["simran", "tara"], required=True)
    parser.add_argument("--parquet",     default=DEFAULT_PARQUET)
    parser.add_argument("--min-freq",    type=int, default=5)
    parser.add_argument("--port",        type=int, default=8765)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out-dir",     default=None,
                        help="Output directory for WAVs (default: <voice>_audio/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else HERE / f"{args.voice}_audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = HERE / PROGRESS_FILE_TMPL.format(voice=args.voice)

    print(f"[INFO] Voice       : {args.voice}")
    print(f"[INFO] Audio dir   : {out_dir.resolve()}")
    print(f"[INFO] Progress    : {progress_path}")
    print(f"[INFO] Port        : {args.port}")
    print(f"[INFO] Concurrency : {args.concurrency}")

    sentences = load_sentences(args.parquet, args.min_freq)
    asyncio.run(run_generation(sentences, args.port, args.concurrency, out_dir, progress_path))


if __name__ == "__main__":
    main()
