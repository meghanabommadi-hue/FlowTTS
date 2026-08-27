#!/usr/bin/env python3
"""Measure whether a stitched utterance still sounds like one speaker.

"It sounds broken and stitched back together" is a real, measurable complaint.
This compares text synthesized through the streaming path (chunked, then
stitched) against the same text synthesized as a single ``generate()`` call, and
reports the three quantities that carry the artifact:

  **Level drift** — the spread of voiced RMS across the utterance. A single
  generate() holds roughly one level throughout; independently generated chunks
  wander, and the wander is heard as the voice changing character mid-sentence.

  **Seam steps** — the largest sample-to-sample jump. A join that clicks shows
  up here far above the waveform's own slope.

  **Silence profile** — how much of the utterance is pause, and how long the
  longest one is. Too little says the sentences were run together; too much says
  the model's edge padding was left in.

Run it against the deployed service's own settings:

    python -m flowtts.test.check_seams --voice anika
    python -m flowtts.test.check_seams --voice anika --save /tmp/seams
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LONG_HINDI = (
    "नमस्ते, मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से। "
    "आपकी EMI तीन हज़ार सात सौ पचास रुपये की due date निकल चुकी है। "
    "कृपया आज ही भुगतान करें ताकि late charge से बचा जा सके। "
    "आप हमारे mobile app के माध्यम से payment कर सकते हैं। "
    "किसी भी सहायता के लिए हमारी customer care team से संपर्क करें। धन्यवाद।"
)


def voiced_profile(wav: np.ndarray, sr: int, window_ms: float = 50.0):
    """Voiced RMS per window, and the windows that count as silence."""
    win = max(1, int(sr * window_ms / 1000))
    n = wav.size // win
    if n < 2:
        return np.zeros(0), np.zeros(0, dtype=bool)
    frames = wav[: n * win].reshape(n, win)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    silent = rms < (rms.max() * 0.02 if rms.max() > 0 else 1.0)
    return rms, silent


def describe(name: str, wav: np.ndarray, sr: int) -> dict:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    rms, silent = voiced_profile(wav, sr)
    voiced = rms[~silent]
    if voiced.size == 0:
        print(f"  {name:26} EMPTY")
        return {}

    # Level drift measured over half-second stretches of voiced audio, which is
    # the timescale a listener perceives as "the voice changed".
    per_half_second = []
    step = max(1, int(0.5 * sr / (sr * 0.05)))
    for i in range(0, len(rms) - step, step):
        window = rms[i:i + step]
        alive = window[window > rms.max() * 0.02]
        if alive.size:
            per_half_second.append(alive.mean())

    drift = (max(per_half_second) / min(per_half_second)) if per_half_second else 1.0
    largest_step = float(np.abs(np.diff(wav)).max())
    silence_fraction = float(silent.mean())

    # Longest continuous silent run, in ms.
    longest = best = 0
    for s in silent:
        longest = longest + 1 if s else 0
        best = max(best, longest)

    stats = {
        "seconds": len(wav) / sr,
        "level_drift": drift,
        "largest_step": largest_step,
        "silence_pct": silence_fraction * 100,
        "longest_pause_ms": best * 50,
        "peak": float(np.abs(wav).max()),
    }
    print(f"  {name:26} {stats['seconds']:5.2f}s  drift={drift:4.2f}x  "
          f"step={largest_step:.4f}  silence={stats['silence_pct']:4.1f}%  "
          f"longest_pause={stats['longest_pause_ms']:4d}ms  peak={stats['peak']:.3f}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="anika")
    ap.add_argument("--language", default="hi")
    ap.add_argument("--text", default=LONG_HINDI)
    ap.add_argument("--num-step", type=int, default=8)
    ap.add_argument("--save", default=None, help="write the wavs to this directory")
    args = ap.parse_args()

    import asyncio

    from flowtts.core.config import settings
    from flowtts.synthesis.chunker import split_for_streaming
    from flowtts.synthesis.models import OmniVoiceSynthesizer
    from flowtts.synthesis.omnivoice_engine import GenParams

    async def run() -> None:
        synth = OmniVoiceSynthesizer()
        await synth.initialize()
        sr = synth.sampling_rate
        params = GenParams.build({"num_step": args.num_step})

        clean, lang = synth.prepare(args.text, args.language)
        chunks = split_for_streaming(
            clean,
            target_chars=settings.streaming.target_chars,
            tolerance_chars=settings.streaming.tolerance_chars,
            split_on_clause=settings.streaming.split_on_clause,
        )
        print(f"\ntext: {len(clean)} chars → {len(chunks)} chunk(s)")
        for c in chunks:
            print(f"   [{c.index}] {len(c):3d}ch  end={c.boundary:8} {c.text[:62]}")

        print("\nmeasurements:")
        # Reference: the whole thing in one generate() — no seams by construction.
        whole = await synth.synthesize(args.text, voice_id=args.voice,
                                       language=args.language, params=params,
                                       chunked=False)
        ref = describe("one generate() [reference]", whole, sr)

        # The streaming path, exactly as a client receives it.
        pieces = []
        async for chunk in synth.synthesize_stream(
            args.text, voice_id=args.voice, language=args.language, params=params
        ):
            if chunk.audio.size:
                pieces.append(chunk.audio)
        streamed = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        got = describe("streamed + stitched", streamed, sr)

        if ref and got:
            print()
            verdict = "OK" if got["level_drift"] <= ref["level_drift"] * 1.35 else "WORSE"
            print(f"  level drift vs reference: {got['level_drift']:.2f}x against "
                  f"{ref['level_drift']:.2f}x  → {verdict}")
            if got["longest_pause_ms"] < 120 and len(chunks) > 1:
                print("  WARNING: no real pause anywhere — sentences may be running together")

        if args.save:
            import soundfile as sf
            out = Path(args.save)
            out.mkdir(parents=True, exist_ok=True)
            sf.write(out / "whole.wav", whole, sr)
            sf.write(out / "streamed.wav", streamed, sr)
            print(f"\n  wrote {out}/whole.wav and {out}/streamed.wav")

    asyncio.run(run())


if __name__ == "__main__":
    main()
