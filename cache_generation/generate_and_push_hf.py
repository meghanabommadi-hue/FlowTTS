#!/usr/bin/env python3
"""
Generate audio for sentences with frequency > 5 and stream-push to HuggingFace.

Resume logic (two layers):
  1. WAV files: if sha256.wav already exists on disk it is skipped — sentence-level resume.
  2. progress.json: tracks last successfully pushed HF batch — batch-level resume.
     On restart, batches already pushed are skipped entirely (no re-synthesis, no re-push).

Uses the same encoding as push_to_hf.py:
  - datasets.Audio(sampling_rate=16000)
  - Dataset.from_list + push_to_hub

Usage:
    export HF_TOKEN=<your_hf_token>
    python generate_and_push_hf.py --voice simran
    python generate_and_push_hf.py --voice tara --port 8766
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

HF_REPOS = {
    "simran": "MeghanaKap/simran_counseling_cache",
    "tara":   "MeghanaKap/tara_counseling_cache",
}

DEFAULT_TOKEN   = os.environ.get("HF_TOKEN", "")
DEFAULT_PARQUET = str(HERE / "normalized_sentences.parquet")
SAMPLE_RATE     = 16000
BATCH_SIZE      = 200


def sha256_filename(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() + ".wav"


def load_sentences(parquet_path: str, min_freq: int = 5) -> list[str]:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    filtered = df[df["frequency"] > min_freq].sort_values("frequency", ascending=False)
    sentences = filtered["text"].tolist()
    print(f"[INFO] Total unique sentences          : {len(df):,}")
    print(f"[INFO] Sentences with frequency > {min_freq}   : {len(sentences):,}")
    return sentences


def load_progress(progress_path: Path) -> int:
    """Return the last successfully pushed batch index (0 = none pushed yet)."""
    if progress_path.exists():
        try:
            data = json.loads(progress_path.read_text())
            last = int(data.get("last_pushed_batch", 0))
            print(f"[RESUME] Found progress file — last pushed batch: {last}")
            return last
        except Exception:
            pass
    return 0


def save_progress(progress_path: Path, batch_idx: int, total_pushed: int) -> None:
    progress_path.write_text(json.dumps({
        "last_pushed_batch": batch_idx,
        "total_pushed":      total_pushed,
    }, indent=2))


# ── WebSocket synthesis ────────────────────────────────────────────────────────

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


async def generate_batch(
    sentences: list[str],
    port: int,
    concurrency: int,
    out_dir: Path,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    counters = {"ok": 0, "skipped": 0, "error": 0}

    total = len(sentences)

    async def process(idx: int, text: str) -> dict | None:
        wav_path = out_dir / sha256_filename(text)
        if wav_path.exists():
            counters["skipped"] += 1
            done = counters["ok"] + counters["skipped"] + counters["error"]
            print(f"  [{done}/{total}] SKIP  {text[:70]!r}", flush=True)
            return {"text": text, "audio": str(wav_path)}
        try:
            t0 = time.perf_counter()
            wav_bytes = await synthesize_one(text, port, semaphore)
            if not wav_bytes:
                raise RuntimeError("empty WAV")
            wav_path.write_bytes(wav_bytes)
            counters["ok"] += 1
            elapsed = time.perf_counter() - t0
            done = counters["ok"] + counters["skipped"] + counters["error"]
            print(f"  [{done}/{total}] OK {elapsed:.1f}s  {text[:70]!r}", flush=True)
            return {"text": text, "audio": str(wav_path)}
        except Exception as exc:
            counters["error"] += 1
            done = counters["ok"] + counters["skipped"] + counters["error"]
            print(f"  [{done}/{total}] ERROR {exc}  {text[:60]!r}", flush=True)
            return None

    raw = await asyncio.gather(*[process(i, t) for i, t in enumerate(sentences, 1)])
    records = [r for r in raw if r is not None]
    print(f"  generated={counters['ok']}  skipped={counters['skipped']}  errors={counters['error']}", flush=True)
    return records


# ── HuggingFace push — streaming via upload_large_folder ──────────────────────
# Avoids loading all wav bytes into RAM (which caused OOM with datasets.Audio).
# Each batch gets its own subfolder under audio_dir/hf_upload/batchNNN/.
# A parquet shard (text + wav filename) is written alongside the wavs so the
# dataset viewer can match text ↔ audio.

def push_batch_to_hf(
    records: list[dict],
    repo_id: str,
    token: str,
    batch_idx: int,
    audio_dir: Path,
) -> None:
    import pandas as pd
    from huggingface_hub import HfApi

    # Build a parquet shard: columns text + file_name (relative path)
    rows = [
        {"text": r["text"], "file_name": Path(r["audio"]).name}
        for r in records
    ]
    shard_dir = audio_dir / f"batch_{batch_idx:04d}"
    shard_dir.mkdir(exist_ok=True)

    # Symlink (or copy) wavs into the shard dir so upload_large_folder picks them up
    for r in records:
        src = Path(r["audio"])
        dst = shard_dir / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())

    # Write parquet shard into the same folder
    parquet_path = shard_dir / "metadata.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(shard_dir),
        path_in_repo=f"data/batch_{batch_idx:04d}",
    )
    print(f"  [HF] Pushed batch {batch_idx} ({len(records)} samples) → {repo_id}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio and stream-push to HuggingFace")
    parser.add_argument("--voice",       choices=["simran", "tara"], required=True,
                        help="Voice to generate: simran or tara")
    parser.add_argument("--repo",        default=None,
                        help="Override HF repo (default: from HF_REPOS dict)")
    parser.add_argument("--token",       default=DEFAULT_TOKEN,   help="HuggingFace write token")
    parser.add_argument("--parquet",     default=DEFAULT_PARQUET, help="Input parquet with text+frequency")
    parser.add_argument("--min-freq",    type=int, default=5,     help="Minimum frequency threshold (default: 5)")
    parser.add_argument("--port",        type=int, default=8765,  help="FlowTTS WebSocket port")
    parser.add_argument("--concurrency", type=int, default=4,     help="Parallel WS requests")
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE, help="Samples per HF push")
    parser.add_argument("--no-push",     action="store_true",     help="Generate only, skip HF push")
    args = parser.parse_args()

    repo_id      = args.repo or HF_REPOS[args.voice]
    audio_dir    = HERE / f"{args.voice}_counseling_audio"
    progress_path = HERE / f"{args.voice}_progress.json"

    if not args.token and not args.no_push:
        print("[ERROR] HF_TOKEN not set. Export it or pass --token.", file=sys.stderr)
        sys.exit(1)

    audio_dir.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences(args.parquet, min_freq=args.min_freq)
    total = len(sentences)
    batches = [sentences[i:i + args.batch_size] for i in range(0, total, args.batch_size)]

    last_pushed = load_progress(progress_path)
    resume_from = last_pushed + 1  # 1-indexed batch

    print(f"[INFO] Voice      → {args.voice}")
    print(f"[INFO] HF repo    → {repo_id}")
    print(f"[INFO] Audio dir  → {audio_dir.resolve()}")
    print(f"[INFO] Batch size={args.batch_size}  total batches={len(batches)}")
    print(f"[INFO] Resuming from batch {resume_from}/{len(batches)}\n")

    t_start = time.perf_counter()
    total_pushed = last_pushed * args.batch_size  # approximate carried-over count

    for batch_idx, batch in enumerate(batches, 1):
        if batch_idx < resume_from:
            print(f"[Batch {batch_idx}/{len(batches)}] already pushed — skipping")
            continue

        print(f"[Batch {batch_idx}/{len(batches)}] generating {len(batch)} sentences...")
        records = asyncio.run(generate_batch(batch, args.port, args.concurrency, audio_dir))

        if not records:
            print(f"  [Batch {batch_idx}] nothing to push, skipping")
            continue

        if not args.no_push:
            try:
                push_batch_to_hf(records, repo_id, args.token, batch_idx, audio_dir)
                total_pushed += len(records)
                save_progress(progress_path, batch_idx, total_pushed)
            except Exception as exc:
                print(f"  [HF ERROR batch {batch_idx}] {exc}")
                print(f"  [RESUME] Progress saved up to batch {batch_idx - 1}. Re-run to retry from here.")
                sys.exit(1)
        else:
            total_pushed += len(records)

        elapsed = time.perf_counter() - t_start
        done    = (batch_idx - resume_from + 1) * args.batch_size
        rate    = done / elapsed if elapsed > 0 else 0
        remaining = (len(batches) - batch_idx) * args.batch_size
        eta     = remaining / rate if rate > 0 else 0
        print(f"  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  ETA={eta:.0f}s\n")

    elapsed = time.perf_counter() - t_start
    print(f"[DONE] total_pushed={total_pushed:,}  total_time={elapsed:.1f}s")
    if not args.no_push:
        print(f"[DONE] https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
