
#!/usr/bin/env python3
"""Generate a ~1 minute demo: N sentences, each with an explicit emotion tag
whose voice matches the sentiment of its text, synthesized individually via
the live FlowTTS /tts endpoint, saved as separate WAVs, then concatenated
into one merged track.

Usage:
    python3 scripts/generate_emotion_demo.py
    python3 scripts/generate_emotion_demo.py --url http://127.0.0.1:8081/tts
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import requests

# Each entry: (tag, text). All lines are spoken by the debt-collection agent
# (never the borrower), and read as ONE continuous call transcript — each
# line follows on from the previous one rather than jumping between
# disconnected fragments. Arc: neutral opening → firm reminder → escalating
# anger → empathetic beat once the reason for missed payment comes up →
# negotiated resolution → happy/relieved close. Kept to 5-7 words each.
SENTENCES = [
    ("",         "This call is about your loan account."),
    ("",         "मैं बजाज finance से बोल रही हूं।"),
    ("angry",    "Your EMI payment is still pending."),
    ("angry",    "यह तीसरी बार payment miss हुआ है।"),
    ("angry",    "आपने बिना बताए payment टाल दिया है।"),
    ("angry",    "देरी से आपका credit score गिरेगा।"),
    ("",         "क्या payment में देरी की कोई वजह है?"),
    ("sad",      "मुझे पता है आपकी नौकरी चली गई।"),
    ("sad",      "I understand this has been very difficult."),
    ("sad",      "यह वाकई एक मुश्किल समय होगा।"),
    ("sad",      "I am sorry to hear that, sir."),
    ("",         "फिर भी हमें कोई समाधान निकालना होगा।"),
    ("",         "क्या आप आज पचास प्रतिशत जमा कर सकते हैं?"),
    ("",         "बाकी राशि के लिए एक हफ्ते का समय मिलेगा।"),
    ("happy",    "बहुत अच्छा, धन्यवाद आपके सहयोग के लिए।"),
    ("happy",    "Thank you for agreeing to pay today."),
    ("",         "अगली EMI की तारीख पंद्रह होगी।"),
    ("happy",    "Have a great day, thank you."),
]

SAMPLE_RATE = 16000


def synthesize(url: str, tag: str, text: str) -> bytes:
    full_text = f"[{tag}] {text}" if tag else text
    resp = requests.post(url, json={"text": full_text, "auto_emotion": False}, timeout=60)
    resp.raise_for_status()
    return resp.content


def merge_wavs(paths: list[Path], out_path: Path, gap_ms: int = 250) -> float:
    """Concatenate WAV files (same format) into one, with a short silence gap between clips."""
    with wave.open(str(paths[0]), "rb") as w0:
        params = w0.getparams()

    gap_frames = int(params.framerate * gap_ms / 1000)
    silence = b"\x00" * (gap_frames * params.sampwidth * params.nchannels)

    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        for i, p in enumerate(paths):
            with wave.open(str(p), "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))
            if i < len(paths) - 1:
                out.writeframes(silence)

    with wave.open(str(out_path), "rb") as w:
        duration = w.getnframes() / w.getframerate()
    return duration


def main():
    parser = argparse.ArgumentParser(description="Generate FlowTTS emotion demo audio.")
    parser.add_argument("--url", default="http://127.0.0.1:8081/tts", help="FlowTTS /tts endpoint")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "sample_files" / "emotion_demo"),
        help="Output directory for individual + merged WAVs",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    print(f"Synthesizing {len(SENTENCES)} sentences -> {out_dir}/")
    for i, (tag, text) in enumerate(SENTENCES, start=1):
        clip_path = out_dir / f"{i:02d}_{tag or 'neutral'}.wav"
        print(f"  [{i:2d}/{len(SENTENCES)}] tag={tag or '(neutral)':<8} {text[:60]!r}")
        wav_bytes = synthesize(args.url, tag, text)
        clip_path.write_bytes(wav_bytes)
        clip_paths.append(clip_path)

    merged_path = out_dir / "merged_demo.wav"
    duration = merge_wavs(clip_paths, merged_path)
    print(f"\nSaved {len(clip_paths)} individual clips to {out_dir}/")
    print(f"Merged -> {merged_path}  ({duration:.1f}s)")


if __name__ == "__main__":
    main()
