#!/usr/bin/env python3
"""Batch TTS audio generation using an already-running FlowTTS server.

Connects to the server via WebSocket (no model reload).
Output files are named <SHA256-of-original-text>.wav.
Already-existing files are skipped — safe to resume.

Usage:
    python generate_audio_batch.py sentences.txt
    python generate_audio_batch.py sentences.txt --port 8765
    python generate_audio_batch.py sentences.txt --concurrency 4
    python generate_audio_batch.py sentences.txt --output-dir /data/audio
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import websockets


def sha256_filename(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() + ".wav"


def load_sentences(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


async def synthesize_one(
    text: str,
    port: int,
    semaphore: asyncio.Semaphore,
) -> bytes:
    """Send one synthesis request; return raw WAV bytes."""
    call_id = str(uuid.uuid4())
    url = f"ws://localhost:{port}/ws/{call_id}"

    async with semaphore:
        async with websockets.connect(url, open_timeout=10, max_size=100 * 1024 * 1024) as ws:
            await ws.send(json.dumps({
                "type":    "synthesize",
                "call_id": call_id,
                "text_id": str(uuid.uuid4()),
                "text":    text,
            }))

            # Frame 1: JSON metadata
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("error", "server error"))

            # Frame 2: WAV bytes
            wav_data = await ws.recv()
            if isinstance(wav_data, str):
                wav_data = wav_data.encode()
            return wav_data


async def process_sentence(
    idx: int,
    total: int,
    raw_text: str,
    out_dir: Path,
    port: int,
    semaphore: asyncio.Semaphore,
    counters: dict,
) -> None:
    out_path = out_dir / sha256_filename(raw_text)

    if out_path.exists():
        counters["skipped"] += 1
        print(f"[{idx}/{total}] SKIP  {out_path.name}  \"{raw_text}\"")
        return

    t0 = time.perf_counter()
    try:
        wav_bytes = await synthesize_one(raw_text, port, semaphore)
        if not wav_bytes:
            raise RuntimeError("empty WAV response")
        out_path.write_bytes(wav_bytes)
        elapsed = time.perf_counter() - t0
        counters["generated"] += 1
        print(f"[{idx}/{total}] OK    {out_path.name}  {elapsed:.2f}s  \"{raw_text}\"")
    except Exception as exc:
        counters["errors"] += 1
        print(f"[{idx}/{total}] ERROR {exc}  \"{raw_text}\"", file=sys.stderr)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Batch TTS via running FlowTTS server")
    parser.add_argument("input", type=Path, help=".txt file — one sentence per line")
    parser.add_argument("--port", type=int, default=8765,
                        help="FlowTTS WebSocket port (default: 8765)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Parallel WS requests (default: 1)")
    parser.add_argument("--output-dir", type=Path, default=Path("cached_audio_files"),
                        help="Output directory (default: ./cached_data_vikram)")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences(args.input)
    total = len(sentences)
    if total == 0:
        print("[ERROR] No sentences found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] {total} sentences  port={args.port}  concurrency={args.concurrency}")
    print(f"[INFO] Output → {args.output_dir.resolve()}\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    counters = {"generated": 0, "skipped": 0, "errors": 0}

    t_start = time.perf_counter()
    tasks = [
        process_sentence(idx, total, text, args.output_dir, args.port,
                         semaphore, counters)
        for idx, text in enumerate(sentences, 1)
    ]
    await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t_start
    print(f"\n[DONE] generated={counters['generated']}  skipped={counters['skipped']}"
          f"  errors={counters['errors']}  total_time={elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
