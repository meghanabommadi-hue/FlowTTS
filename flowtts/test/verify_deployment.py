#!/usr/bin/env python3
"""End-to-end acceptance check against a running server.

Exercises the things a deployment is actually promised to do, over HTTP and
WebSocket, and reports pass/fail per item:

  * synthesis in all 22 scheduled languages of India, verifying real audio comes
    back and that the text preprocessor left no bare numerals for the model to
    guess at,
  * voice cloning: create, use, list, delete — and the one-shot preview,
  * the streaming API's time-to-first-byte,
  * the WebSocket protocol's framing,
  * OmniVoice's three synthesis modes (clone / design / auto) and its inline
    control tags,
  * the OpenAI-compatible endpoint.

    python -m flowtts.test.verify_deployment --url http://127.0.0.1:9000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import struct
import time
import urllib.error
import urllib.request
import wave

# One realistic sentence per scheduled language, each carrying a number and
# currency so the normalizer is exercised too, not just the model.
INDIC_SAMPLES = {
    "hi": "नमस्ते, आपका बकाया ₹2,500 है, कृपया आज ही भुगतान करें।",
    "bn": "নমস্কার, আপনার ব্যালেন্স ৯,৫০০ টাকা।",
    "mr": "नमस्कार, तुमच्या खात्यात ₹1,250 आहेत.",
    "te": "నమస్కారం, మీ ఖాతాలో ₹1,250 ఉంది.",
    "ta": "வணக்கம், உங்கள் கணக்கில் ₹1,250 உள்ளது.",
    "gu": "નમસ્તે, તમારા ખાતામાં ₹1,250 છે.",
    "ur": "السلام علیکم، آپ کا بیلنس ₹2,500 ہے۔",
    "kn": "ನಮಸ್ಕಾರ, ನಿಮ್ಮ ಖಾತೆಯಲ್ಲಿ ₹1,250 ಇದೆ.",
    "or": "ନମସ୍କାର, ଆପଣଙ୍କ ବାକି ₹1,200 ଅଛି।",
    "ml": "നമസ്കാരം, നിങ്ങളുടെ അക്കൗണ്ടിൽ ₹1,250 ഉണ്ട്.",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਹਾਡਾ ਬਕਾਇਆ ₹5,000 ਹੈ।",
    "as": "নমস্কাৰ, আপোনাৰ বেলেঞ্চ ₹1,500 আছে।",
    "mai": "प्रणाम, अहाँक बकाया ₹2,000 अछि।",
    "sat": "ᱡᱚᱦᱟᱨ, ᱟᱢᱟᱜ ᱵᱮᱞᱮᱱᱥ ₹1,200 ᱢᱮᱱᱟᱜᱼᱟ।",
    "ks": "آداب، توہُنٛد بیلنس ₹2,500 چھُ۔",
    "ne": "नमस्ते, तपाईंको बाँकी ₹2,000 छ।",
    "sd": "سلام، توهانجو بيلنس ₹2,500 آهي.",
    "kok": "नमस्कार, तुमच्या खात्यांत ₹1,250 आसात.",
    "doi": "नमस्कार, तुंदा बकाया ₹2,000 ऐ।",
    "mni": "খুরুমজরি, নখোয়গী বেলেন্স ₹1,200 লৈ।",
    "brx": "नमस्कार, नोंथांनि बाकिया ₹2,000 दं।",
    "sa": "नमस्ते, भवतः शेषम् ₹2,000 अस्ति।",
}


def post(url: str, path: str, payload: dict, timeout: float = 120.0):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url + path, body, {"content-type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
    return data, (time.perf_counter() - started) * 1000, response


def wav_seconds(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data)) as handle:
            return handle.getnframes() / handle.getframerate()
    except Exception:  # noqa: BLE001
        return 0.0


def wav_peak(data: bytes) -> float:
    """Peak amplitude, so "returned a WAV" is not confused with "made a sound"."""
    try:
        with wave.open(io.BytesIO(data)) as handle:
            frames = handle.readframes(handle.getnframes())
        if not frames:
            return 0.0
        samples = struct.unpack(f"<{len(frames) // 2}h", frames[: len(frames) // 2 * 2])
        return max(abs(s) for s in samples) / 32768.0
    except Exception:  # noqa: BLE001
        return 0.0


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def add(self, ok: bool, name: str, detail: str = "") -> None:
        self.rows.append((ok, name, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:42} {detail}", flush=True)

    def summary(self) -> bool:
        passed = sum(1 for ok, _, _ in self.rows if ok)
        print(f"\n  {passed}/{len(self.rows)} checks passed")
        for ok, name, detail in self.rows:
            if not ok:
                print(f"    FAILED: {name}  {detail}")
        return passed == len(self.rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9000")
    ap.add_argument("--voice", default="anika")
    ap.add_argument("--num-step", type=int, default=8)
    ap.add_argument("--skip-languages", action="store_true")
    args = ap.parse_args()

    url = args.url.rstrip("/")
    report = Report()
    gen = {"num_step": args.num_step}

    print(f"\nverifying {url}\n")

    # ---- readiness ----
    try:
        with urllib.request.urlopen(url + "/readyz", timeout=15) as response:
            ready = json.loads(response.read())
        report.add(ready.get("ready") is True, "service ready",
                   f"{len(ready.get('voices', []))} voices, {ready.get('sample_rate')} Hz")
    except Exception as exc:  # noqa: BLE001
        report.add(False, "service ready", str(exc))
        return 1

    # ---- all 22 scheduled languages ----
    if not args.skip_languages:
        print("\n  -- 22 scheduled languages of India --")
        for code, text in INDIC_SAMPLES.items():
            try:
                norm, _, _ = post(url, "/v1/normalize", {"text": text, "language": code})
                normalized = json.loads(norm)["normalized"]
                digits_left = any(ch.isdigit() for ch in normalized)

                audio, ms, _ = post(url, "/v1/tts", {
                    "text": text, "language": code, "voice_id": args.voice,
                    "generation": gen,
                })
                seconds, peak = wav_seconds(audio), wav_peak(audio)
                ok = seconds > 0.3 and peak > 0.02 and not digits_left
                detail = f"{seconds:5.2f}s peak={peak:.3f} {ms:6.0f}ms"
                if digits_left:
                    detail += "  DIGITS NOT NORMALIZED"
                report.add(ok, f"synthesize {code}", detail)
            except Exception as exc:  # noqa: BLE001
                report.add(False, f"synthesize {code}", str(exc)[:70])

    # ---- streaming TTFB ----
    print("\n  -- streaming --")
    try:
        body = json.dumps({"text": INDIC_SAMPLES["hi"], "language": "hi",
                           "voice_id": args.voice, "format": "pcm",
                           "generation": gen}).encode()
        req = urllib.request.Request(url + "/v1/tts/stream", body,
                                     {"content-type": "application/json"})
        started = time.perf_counter()
        first = None
        total = 0
        with urllib.request.urlopen(req, timeout=120) as response:
            while True:
                chunk = response.read(4096)
                if not chunk:
                    break
                if first is None:
                    first = (time.perf_counter() - started) * 1000
                total += len(chunk)
        report.add(first is not None and total > 0, "streaming synthesis",
                   f"ttfb={first:.0f}ms  audio={total / 2 / 24000:.2f}s")
    except Exception as exc:  # noqa: BLE001
        report.add(False, "streaming synthesis", str(exc)[:70])

    # ---- OmniVoice's three modes + control tags ----
    print("\n  -- synthesis modes --")
    for name, payload in [
        ("voice clone (by alias)", {"text": "Hello, this is a cloned voice.",
                                    "voice_id": args.voice, "language": "en"}),
        ("voice design (instruct)", {"text": "The world is full of wonders.",
                                     "instruct": "Female, Elderly, British Accent",
                                     "language": "en"}),
        ("auto voice", {"text": "The world is full of wonders.", "language": "en"}),
        ("control tag [laughter]", {"text": "[laughter] You really got me.",
                                    "voice_id": args.voice, "language": "en"}),
        ("ARPAbet [B EY1 S]", {"text": "He plays the [B EY1 S] guitar.",
                               "voice_id": args.voice, "language": "en"}),
        ("speed 1.3x", {"text": "This should sound faster than usual.",
                        "voice_id": args.voice, "language": "en", "speed": 1.3}),
    ]:
        try:
            audio, ms, _ = post(url, "/v1/tts", {**payload, "generation": gen})
            seconds, peak = wav_seconds(audio), wav_peak(audio)
            report.add(seconds > 0.3 and peak > 0.02, name,
                       f"{seconds:5.2f}s peak={peak:.3f} {ms:6.0f}ms")
        except Exception as exc:  # noqa: BLE001
            report.add(False, name, str(exc)[:70])

    # ---- voice cloning ----
    print("\n  -- voice cloning --")
    clone_id = "verify-clone"
    try:
        seed, _, _ = post(url, "/v1/tts", {
            "text": "यह एक परीक्षण संदर्भ ऑडियो है, जिससे आवाज़ की नकल की जाएगी।",
            "language": "hi", "voice_id": args.voice, "generation": gen,
        })
        report.add(wav_seconds(seed) > 1.0, "reference clip generated",
                   f"{wav_seconds(seed):.2f}s")

        created, ms, _ = post(url, "/v1/voices", {
            "voice_id": clone_id,
            "reference_text": "यह एक परीक्षण संदर्भ ऑडियो है, जिससे आवाज़ की नकल की जाएगी।",
            "audio_base64": base64.b64encode(seed).decode(),
            "language": "hi", "overwrite": True,
        })
        info = json.loads(created)
        report.add(info.get("voice_id") == clone_id, "POST /v1/voices (clone)",
                   f"{info.get('reference_frames')} frames, {ms:.0f}ms")

        with urllib.request.urlopen(url + "/v1/voices", timeout=15) as response:
            voices = json.loads(response.read())["voices"]
        report.add(any(v["voice_id"] == clone_id for v in voices),
                   "GET /v1/voices lists it", f"{len(voices)} voices")

        audio, ms, _ = post(url, "/v1/tts", {
            "text": "अब मैं क्लोन की गई आवाज़ में बोल रही हूं।",
            "language": "hi", "voice_id": clone_id, "generation": gen,
        })
        report.add(wav_seconds(audio) > 0.5 and wav_peak(audio) > 0.02,
                   "synthesize with the cloned voice",
                   f"{wav_seconds(audio):.2f}s peak={wav_peak(audio):.3f}")

        req = urllib.request.Request(f"{url}/v1/voices/{clone_id}", method="DELETE")
        with urllib.request.urlopen(req, timeout=15) as response:
            deleted = json.loads(response.read())
        report.add(deleted.get("status") == "ok", "DELETE /v1/voices/{id}")
    except Exception as exc:  # noqa: BLE001
        report.add(False, "voice cloning", str(exc)[:90])

    # ---- inline one-shot clone ----
    try:
        audio, ms, _ = post(url, "/v1/tts", {
            "text": "This is a one-shot clone with no voice registered.",
            "language": "en",
            "reference_audio": base64.b64encode(seed).decode(),
            "reference_text": "यह एक परीक्षण संदर्भ ऑडियो है, जिससे आवाज़ की नकल की जाएगी।",
            "generation": gen,
        })
        report.add(wav_seconds(audio) > 0.5, "inline reference_audio clone",
                   f"{wav_seconds(audio):.2f}s {ms:.0f}ms")
    except Exception as exc:  # noqa: BLE001
        report.add(False, "inline reference_audio clone", str(exc)[:70])

    # ---- OpenAI-compatible ----
    print("\n  -- compatibility --")
    try:
        audio, ms, _ = post(url, "/v1/audio/speech", {
            "model": "omnivoice", "input": "Testing the OpenAI compatible endpoint.",
            "voice": args.voice, "response_format": "wav",
        })
        report.add(wav_seconds(audio) > 0.3, "POST /v1/audio/speech",
                   f"{wav_seconds(audio):.2f}s {ms:.0f}ms")
    except Exception as exc:  # noqa: BLE001
        report.add(False, "POST /v1/audio/speech", str(exc)[:70])

    # ---- WebSocket ----
    try:
        import asyncio

        import websockets

        async def _ws_check() -> tuple[int, int, int]:
            ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/verify"
            async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
                await ws.send(json.dumps({
                    "type": "synthesize", "text": INDIC_SAMPLES["hi"],
                    "language": "hi", "voice_id": args.voice, "text_id": "v1",
                    "generation": gen,
                }))
                frames, audio_bytes = 0, 0
                while True:
                    message = await ws.recv()
                    if isinstance(message, bytes):
                        split = message.index(b"}") + 1
                        json.loads(message[:split])
                        frames += 1
                        audio_bytes += len(message) - split
                    else:
                        done = json.loads(message)
                        return frames, audio_bytes, done.get("llm_ttft_ms") or 0

        frames, audio_bytes, ttfb = asyncio.run(_ws_check())
        report.add(frames > 0 and audio_bytes > 0, "WebSocket /ws streaming",
                   f"{frames} frames, {audio_bytes / 2 / 24000:.2f}s, ttfb={ttfb}ms")
    except Exception as exc:  # noqa: BLE001
        report.add(False, "WebSocket /ws streaming", str(exc)[:70])

    return 0 if report.summary() else 1


if __name__ == "__main__":
    raise SystemExit(main())
