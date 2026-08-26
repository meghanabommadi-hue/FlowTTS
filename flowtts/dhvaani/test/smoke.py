#!/usr/bin/env python3
"""End-to-end smoke test — the first thing to run after deploying to a GPU box.

    # create a voice from a clip + its transcript, then synthesize in 4 languages
    python -m flowtts.dhvaani.test.smoke \
        --ref sample_files/simran.wav \
        --ref-text "नमस्ते, मैं वाणी बोल रही हूं" \
        --voice-id simran --out /tmp/dhvaani_smoke

    # reuse an existing voice
    python -m flowtts.dhvaani.test.smoke --voice-id simran --out /tmp/smoke

Checks, in order: model loads, a voice can be cloned, synthesis produces
non-silent audio of a plausible duration for the character count, and the
per-stage timings are sane. Exits non-zero on the first real failure.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from pathlib import Path

SENTENCES = [
    ("hi", "नमस्ते, आपकी ईएमआई दो हज़ार पांच सौ रुपये बकाया है।"),
    ("en", "Hello, your payment of twelve hundred rupees is due today."),
    ("ta", "வணக்கம், உங்கள் கட்டணம் நிலுவையில் உள்ளது."),
    ("te", "నమస్కారం, మీ చెల్లింపు పెండింగ్‌లో ఉంది."),
    ("bn", "নমস্কার, আপনার পেমেন্ট বাকি আছে।"),
    ("mr", "नमस्कार, तुमचे पेमेंट प्रलंबित आहे."),
]


def write_wav(path: Path, pcm: bytes, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    return ok


async def run(args) -> int:
    import numpy as np

    from flowtts.dhvaani.config import apply_profile, dhv_settings
    from flowtts.dhvaani.engine.engine import DhvaaniEngine

    s = dhv_settings
    if args.profile:
        apply_profile(s, args.profile)
    if args.backend:
        s.backend.kind = args.backend

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    print("\n== 1. engine startup ==")
    eng = DhvaaniEngine(s)
    await eng.start()
    st = eng.stats()
    failures += not check("engine ready", eng.ready)
    print(f"       backend={st['backend']}  arenas={st['scheduler']['arenas']['total_mib']} MiB"
          f"  vram={st['vram'].get('allocated', 0) / 2**30:.2f} GiB")

    print("\n== 2. voice ==")
    if args.ref:
        ref = Path(args.ref)
        if not ref.exists():
            print(f"  reference audio not found: {ref}", file=sys.stderr)
            return 1
        if not args.ref_text:
            print("  --ref-text is required with --ref (DhVaani derives its speaking "
                  "rate from prompt_frames / prompt_tokens)", file=sys.stderr)
            return 1
        v = eng.voices.create(
            voice_id=args.voice_id, audio=ref, transcript=args.ref_text,
            language=args.language or "", overwrite=True,
            source_filename=ref.name,
        )
        print(f"       created '{v.voice_id}': {v.duration_s:.2f}s, {v.mel_frames} frames, "
              f"{len(v.token_ids)} tokens, {v.frames_per_token:.2f} frames/token")
    else:
        v = eng.voices.resolve(args.voice_id)
        print(f"       using existing '{v.voice_id}' ({v.duration_s:.2f}s)")
    failures += not check("voice has a prompt", v.mel_frames > 0)
    failures += not check(
        "frames/token plausible", 1.0 < v.frames_per_token < 30.0,
        f"{v.frames_per_token:.2f} -- far outside means the transcript does not "
        f"match the audio, and the voice will speak at the wrong rate",
    )

    print("\n== 3. synthesis ==")
    print(f"  {'lang':5s} {'chars':>6s} {'spans':>6s} {'TTFB':>8s} {'total':>8s} "
          f"{'audio':>7s} {'RTF':>6s} {'peak':>6s}")
    for lang, text in SENTENCES[: args.languages]:
        try:
            pcm, m = await eng.synthesize(text, v.voice_id, lang)
        except Exception as e:
            print(f"  {lang:5s} FAILED: {e}")
            failures += 1
            continue

        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        peak = float(np.abs(arr).max()) if arr.size else 0.0
        dur = arr.size / s.audio.output_sample_rate
        print(f"  {lang:5s} {len(text):6d} {m.n_spans:6d} {m.ttfb_ms:7.1f}m "
              f"{m.total_ms:7.1f}m {dur:6.2f}s {m.rtf:6.3f} {peak:6.3f}")

        write_wav(out_dir / f"smoke_{lang}.wav", pcm, s.audio.output_sample_rate)
        if not check(f"{lang}: audio is not silent", peak > 0.01, f"peak={peak:.4f}"):
            failures += 1
        # A character maps to roughly frames_per_token/93.75 seconds; allow a
        # wide band since scripts differ enormously in characters per phoneme.
        expected = len(text) * v.frames_per_token / 93.75
        if not check(f"{lang}: duration plausible",
                     0.35 * expected < dur < 2.5 * expected,
                     f"{dur:.2f}s vs ~{expected:.2f}s expected"):
            failures += 1

    print("\n== 4. streaming ==")
    chunks, first_ms = 0, None
    import time as _t

    t0 = _t.perf_counter()
    async for ch in eng.synthesize_stream(SENTENCES[0][1], v.voice_id, "hi"):
        if ch.audio and first_ms is None:
            first_ms = (_t.perf_counter() - t0) * 1000
        chunks += 1
    failures += not check("streamed more than one chunk", chunks > 1, f"{chunks} chunks")
    failures += not check("first chunk arrived", first_ms is not None,
                          f"{first_ms:.1f} ms" if first_ms else "never")

    print("\n== 5. VRAM stability ==")
    before = eng.stats()["vram"].get("allocated", 0)
    for _ in range(args.stability_iters):
        await eng.synthesize(SENTENCES[0][1], v.voice_id, "hi")
    after = eng.stats()["vram"].get("allocated", 0)
    growth = (after - before) / 2**20
    failures += not check(
        f"VRAM flat over {args.stability_iters} requests", abs(growth) < 256,
        f"{growth:+.1f} MiB",
    )

    await eng.stop()
    print(f"\nwrote {args.languages} wav file(s) to {out_dir}/")
    print("SMOKE TEST:", "PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="DhVaani end-to-end smoke test")
    ap.add_argument("--ref", default=None, help="Reference clip to clone")
    ap.add_argument("--ref-text", default=None, help="Exact transcript of --ref")
    ap.add_argument("--voice-id", default="smoke")
    ap.add_argument("--language", default=None, help="Language of the reference clip")
    ap.add_argument("--out", default="/tmp/dhvaani_smoke")
    ap.add_argument("--profile", default=None, choices=["fast", "balanced", "quality"])
    ap.add_argument("--backend", default=None, choices=["torch", "trt", "triton"])
    ap.add_argument("--languages", type=int, default=4, help="How many sentences to try")
    ap.add_argument("--stability-iters", type=int, default=30)
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
