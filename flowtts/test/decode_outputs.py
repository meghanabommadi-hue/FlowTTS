#!/usr/bin/env python3
"""
Decode JSON benchmark outputs to WAV files.

Reads JSON files produced by benchmark.py (each containing audio_tokens
and context tokens), decodes them with ncodec TTSCodec, and writes WAV
files to a sibling `audio/` folder next to the input directory.

Usage:
    # Decode all JSONs in a specific bench run
    python flowtts/test/decode_outputs.py FlowTTS/test/bench_20260220_123456

    # Decode all bench runs under FlowTTS/test/
    python flowtts/test/decode_outputs.py FlowTTS/test/

    # Custom output folder
    python flowtts/test/decode_outputs.py FlowTTS/test/bench_20260220_123456 --out-dir /tmp/wavs
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from ncodec.codec import TTSCodec

CONTEXT_TOKENS = (
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


def decode_tokens(tts: TTSCodec, audio_tokens: str, context: str) -> bytes:
    """Decode a token string to raw WAV bytes at 48 kHz PCM_16."""
    audio = tts.decode(audio_tokens, context)
    audio = np.asarray(audio)
    if audio.dtype == np.float16:
        audio = audio.astype(np.float32)
    audio = audio.squeeze()

    wav_io = io.BytesIO()
    sf.write(wav_io, audio, samplerate=48000, subtype="PCM_16", format="WAV")
    wav_io.seek(0)
    return wav_io.read()


def decode_directory(json_dir: Path, out_dir: Path, tts: TTSCodec) -> None:
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"  No JSON files in {json_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  {json_dir.name}  →  {out_dir}/")
    print(f"  {len(json_files)} file(s) to decode\n")

    ok = fail = 0
    for jf in json_files:
        try:
            record = json.loads(jf.read_text(encoding="utf-8"))
            audio_tokens = record.get("audio_tokens", "")
            if not audio_tokens:
                print(f"  SKIP  {jf.name}  (no audio_tokens)")
                fail += 1
                continue

            context = record.get("context_tokens", CONTEXT_TOKENS)
            text    = record.get("text", "")
            llm_s   = record.get("llm_s")

            t0 = time.perf_counter()
            wav_bytes = decode_tokens(tts, audio_tokens, context)
            dec_ms = (time.perf_counter() - t0) * 1000

            wav_name = jf.stem + ".wav"
            (out_dir / wav_name).write_bytes(wav_bytes)

            print(
                f"  OK    {wav_name}  "
                f"decode={dec_ms:.0f}ms  "
                + (f"llm={llm_s*1000:.0f}ms  " if llm_s is not None else "")
                + f"{len(wav_bytes)//1024}KB  {text[:40]!r}"
            )
            ok += 1

        except Exception as e:
            print(f"  FAIL  {jf.name}  {e}")
            fail += 1

    print(f"\n  Done: {ok} ok, {fail} failed  →  {out_dir}/\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode FlowTTS JSON outputs to WAV files"
    )
    parser.add_argument(
        "input",
        help="Path to a bench_* directory (or parent dir containing multiple bench_* dirs)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for WAVs (default: <input>/audio/ or <input>/../audio/<run>/)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()

    print("Loading TTSCodec...", flush=True)
    tts = TTSCodec()
    print("Codec ready.\n", flush=True)

    # Collect directories to process
    if input_path.is_dir():
        # Check if it directly contains JSON files (single run)
        if list(input_path.glob("*.json")):
            dirs_to_process = [input_path]
        else:
            # Treat as parent: process all bench_* subdirs
            dirs_to_process = sorted(d for d in input_path.iterdir()
                                     if d.is_dir() and list(d.glob("*.json")))
    else:
        parser.error(f"Not a directory: {input_path}")

    if not dirs_to_process:
        print("No JSON files found.")
        return

    for json_dir in dirs_to_process:
        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            # Place audio/ next to the bench_* dir, with same name
            out_dir = json_dir.parent / "audio" / json_dir.name

        decode_directory(json_dir, out_dir, tts)


if __name__ == "__main__":
    main()
