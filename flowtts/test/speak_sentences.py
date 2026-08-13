"""
Split text on full stops, normalize each sentence, synthesize it with the
FlowTTS WebSocket server one sentence at a time, and play each one back
through speakers as soon as it's generated (before moving to the next).

Usage
-----
    python flowtts/test/speak_sentences.py "First sentence. Second one. Third!"
    python flowtts/test/speak_sentences.py --file mytext.txt
    python flowtts/test/speak_sentences.py --url ws://localhost:8765 "Hello. World."

Requires `ffplay` (from ffmpeg) on PATH for playback.
"""

import argparse
import asyncio
import re
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowtts.flowSynthesizer import FlowSynthesizer  # noqa: E402


# ---------------------------------------------------------------------------
# Sentence splitting + normalization
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split text into sentences on full stops ('.')."""
    parts = text.split(".")
    return [p for p in (normalize(p) for p in parts) if p]


def normalize(sentence: str) -> str:
    """Basic text cleanup: collapse whitespace, trim, restore a trailing full stop."""
    cleaned = re.sub(r"\s+", " ", sentence).strip()
    if not cleaned:
        return ""
    if cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned + "."


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def play_wav_bytes(wav_bytes: bytes) -> None:
    """Write WAV bytes to a temp file and play them synchronously via ffplay."""
    tmp_path = Path(f"/tmp/flowtts_play_{uuid.uuid4().hex[:8]}.wav")
    tmp_path.write_bytes(wav_bytes)
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(tmp_path)],
            check=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

async def speak_all(sentences: list[str], url: str) -> None:
    synth = FlowSynthesizer(url=url, request_timeout=60.0)
    synth.start()

    # Wait for connection
    for _ in range(50):
        if synth.is_ready:
            break
        await asyncio.sleep(0.1)

    for i, sentence in enumerate(sentences, 1):
        print(f"[{i}/{len(sentences)}] Synthesizing: {sentence!r}")
        try:
            result = await synth.synthesize(sentence)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        print(f"  → {len(result.audio_bytes)} bytes, playing...")
        play_wav_bytes(result.audio_bytes)

    synth.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split text on full stops and speak each sentence one by one."
    )
    parser.add_argument("text", nargs="?", help="Text to synthesize.")
    parser.add_argument("--file", help="Path to a text file to read input from.")
    parser.add_argument(
        "--url", default="ws://localhost:8765", help="FlowTTS WebSocket server URL."
    )
    args = parser.parse_args()

    if args.file:
        raw_text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        raw_text = args.text
    else:
        parser.error("Provide text as an argument or via --file.")
        return

    sentences = split_sentences(raw_text)
    if not sentences:
        print("No sentences found (nothing before a '.').")
        return

    asyncio.run(speak_all(sentences, args.url))


if __name__ == "__main__":
    main()
