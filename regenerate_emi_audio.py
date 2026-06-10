#!/usr/bin/env python3
"""
Regenerate audio for all sentences containing "EMI" from bajaj_3_lakh_sentences.txt.

- Sends text to TTS with "EMI" replaced by "E M I"
- Keeps filenames based on SHA256 of the ORIGINAL text (same as existing files)
- Saves new audio to a diff folder (emi_regen_diff/)
- Replaces existing files in simran_3_lakh/

Usage:
    python regenerate_emi_audio.py
    python regenerate_emi_audio.py --port 8765 --concurrency 4
    python regenerate_emi_audio.py --sentences bajaj_3_lakh_sentences.txt \
        --source-dir simran_3_lakh --diff-dir emi_regen_diff
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

import websockets


def sha256_filename(text: str) -> str:
    """SHA256 of text — matches the naming used by generate_audio_batch.py."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest() + ".wav"


def replace_emi(text: str) -> str:
    """Replace standalone 'EMI' with 'E M I' (word boundary aware)."""
    return re.sub(r'\bEMI\b', 'E M I', text)


def load_emi_sentences(path: Path) -> list[str]:
    """Return lines that contain the word EMI."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and re.search(r'\bEMI\b', ln)]


async def synthesize_one(text: str, port: int, semaphore: asyncio.Semaphore) -> bytes:
    """Send one synthesis request; return raw WAV bytes."""
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


async def process_sentence(
    idx: int,
    total: int,
    original_text: str,
    source_dir: Path,
    diff_dir: Path,
    port: int,
    semaphore: asyncio.Semaphore,
    counters: dict,
    report_rows: list,
) -> None:
    label = f"{idx}/{total}"

    # Filename is always based on the ORIGINAL text (to match existing files)
    filename = sha256_filename(original_text)
    diff_path = diff_dir / filename
    source_path = source_dir / filename

    # Text sent to TTS has EMI → E M I
    synthesis_text = replace_emi(original_text)

    t0 = time.perf_counter()
    try:
        wav_bytes = await synthesize_one(synthesis_text, port, semaphore)
        if not wav_bytes:
            raise RuntimeError("empty WAV response")

        diff_path.write_bytes(wav_bytes)

        if source_path.exists():
            shutil.copy2(diff_path, source_path)
            replace_status = "replaced"
        else:
            print(f"[{label}] WARN  {filename} not found in {source_dir}, skipping replace")
            replace_status = "not_in_source"

        elapsed = time.perf_counter() - t0
        counters["generated"] += 1
        report_rows.append({
            "status": "OK",
            "idx": idx,
            "filename": filename,
            "replace_status": replace_status,
            "elapsed_s": round(elapsed, 2),
            "original": original_text,
            "synthesis": synthesis_text,
        })
        print(f"[{label}] OK    {filename}  {elapsed:.2f}s  ({replace_status})")
        print(f"          original : {original_text!r}")
        print(f"          synthesis: {synthesis_text!r}")

    except Exception as exc:
        counters["errors"] += 1
        report_rows.append({
            "status": "ERROR",
            "idx": idx,
            "filename": filename,
            "replace_status": "n/a",
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "original": original_text,
            "synthesis": synthesis_text,
            "error": str(exc),
        })
        print(f"[{label}] ERROR {exc}")
        print(f"          original : {original_text!r}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate EMI audio with E M I pronunciation")
    parser.add_argument("--sentences", type=Path, default=Path("bajaj_3_lakh_sentences.txt"),
                        help="Input sentences file (default: bajaj_3_lakh_sentences.txt)")
    parser.add_argument("--source-dir", type=Path, default=Path("simran_3_lakh"),
                        help="Folder whose files will be replaced (default: simran_3_lakh)")
    parser.add_argument("--diff-dir", type=Path, default=Path("emi_regen_diff"),
                        help="Folder to save new generations (default: emi_regen_diff)")
    parser.add_argument("--port", type=int, default=8765,
                        help="FlowTTS WebSocket port (default: 8765)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Parallel WS requests (default: 1)")
    args = parser.parse_args()

    if not args.sentences.is_file():
        print(f"[ERROR] Sentences file not found: {args.sentences}", file=sys.stderr)
        sys.exit(1)

    if not args.source_dir.is_dir():
        print(f"[ERROR] Source dir not found: {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    args.diff_dir.mkdir(parents=True, exist_ok=True)

    sentences = load_emi_sentences(args.sentences)
    total = len(sentences)
    if total == 0:
        print("[ERROR] No sentences with 'EMI' found.")
        sys.exit(1)

    print(f"[INFO] Found {total} sentences containing EMI")
    print(f"[INFO] Port={args.port}  Concurrency={args.concurrency}")
    print(f"[INFO] Source dir → {args.source_dir.resolve()}")
    print(f"[INFO] Diff dir   → {args.diff_dir.resolve()}\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    counters = {"generated": 0, "errors": 0}
    report_rows: list = []

    t_start = time.perf_counter()
    tasks = [
        process_sentence(idx, total, text, args.source_dir, args.diff_dir,
                         args.port, semaphore, counters, report_rows)
        for idx, text in enumerate(sentences, 1)
    ]
    await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t_start

    # ── Summary report ────────────────────────────────────────────────────────
    replaced      = [r for r in report_rows if r["replace_status"] == "replaced"]
    not_in_source = [r for r in report_rows if r["replace_status"] == "not_in_source"]
    errors        = [r for r in report_rows if r["status"] == "ERROR"]
    ok_rows       = [r for r in report_rows if r["status"] == "OK"]
    avg_time      = (sum(r["elapsed_s"] for r in ok_rows) / len(ok_rows)) if ok_rows else 0.0

    sep = "=" * 70
    print(f"\n{sep}")
    print("  SUMMARY REPORT")
    print(sep)
    print(f"  Input file       : {args.sentences.resolve()}")
    print(f"  Source dir       : {args.source_dir.resolve()}")
    print(f"  Diff dir         : {args.diff_dir.resolve()}")
    print(f"  Total EMI sentences found : {total}")
    print(f"  Successfully generated    : {counters['generated']}")
    print(f"    ↳ replaced in source    : {len(replaced)}")
    print(f"    ↳ not found in source   : {len(not_in_source)}")
    print(f"  Errors                    : {counters['errors']}")
    print(f"  Total time                : {elapsed:.1f}s")
    print(f"  Avg time per sentence     : {avg_time:.2f}s")

    if not_in_source:
        print(f"\n  [WARN] {len(not_in_source)} file(s) were generated but had no match in source dir:")
        for r in not_in_source:
            print(f"    [{r['idx']}] {r['filename']}")
            print(f"         {r['original']!r}")

    if errors:
        print(f"\n  [ERRORS] {len(errors)} sentence(s) failed:")
        for r in errors:
            print(f"    [{r['idx']}] {r.get('error', 'unknown error')}")
            print(f"         {r['original']!r}")

    print(f"\n  Processed sentences (original → synthesis):")
    for r in sorted(report_rows, key=lambda x: x["idx"]):
        status_tag = "OK   " if r["status"] == "OK" else "ERROR"
        print(f"    [{r['idx']:>4}] [{status_tag}] {r['original']!r}")
        if r["status"] == "OK":
            print(f"           → {r['synthesis']!r}")

    print(sep)
    print(f"  New files saved to : {args.diff_dir.resolve()}")
    print(f"  Files replaced in  : {args.source_dir.resolve()}")
    print(sep)

    # ── Audio filenames txt ───────────────────────────────────────────────────
    filenames_path = args.diff_dir / "emi_audio_filenames.txt"
    with filenames_path.open("w", encoding="utf-8") as f:
        for r in sorted(report_rows, key=lambda x: x["idx"]):
            f.write(r["filename"] + "\n")
    print(f"\n  Audio filenames list saved to: {filenames_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
