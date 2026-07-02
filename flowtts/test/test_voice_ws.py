#!/usr/bin/env python3
"""Test a (cloned) voice over the FlowTTS WebSocket — streams audio to a WAV.

Sends one `synthesize` request with a given voice_id, receives the streamed
audio_chunk frames (JSON header + raw int16 PCM in one binary frame), concatenates
the PCM, writes a WAV, and prints TTFB / total / RTF. Dependency-light: only needs
`websockets` (no numpy/torch) — the WAV is written with the stdlib `wave` module.

Usage (inside the container — write the WAV to the bind-mounted voices/ so it shows
up on the host):
    docker compose exec omnivoice-tts python -m flowtts.test.test_voice_ws \
        --voice niharika --lang hi \
        --text "नमस्ते, मैं प्रिया बोल रही हूँ, आपकी कैसे मदद कर सकती हूँ?" \
        --out voices/niharika_test.wav

From a host that has Python+websockets:
    python -m flowtts.test.test_voice_ws --host <vm-ip> --port 8080 --voice niharika \
        --text "…" --out niharika_test.wav

List loaded voices first (control API):
    curl -s http://localhost:8764/voices
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
import wave
from pathlib import Path

import websockets

_DEFAULT_TEXT = "नमस्ते, मैं प्रिया बोल रही हूँ, बजाज फाइनेंस से। आपकी कैसे मदद कर सकती हूँ?"


def _split_frame(raw) -> tuple[dict, bytes]:
    """Split a combined binary frame (JSON header {…} + appended PCM) → (msg, pcm).

    Text frames (pure JSON) return (msg, b"")."""
    if isinstance(raw, str):
        return json.loads(raw), b""
    depth = 0
    end = 0
    for i, b in enumerate(raw):
        if b == 0x7B:          # '{'
            depth += 1
        elif b == 0x7D:        # '}'
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return json.loads(raw[:end]), raw[end:]


async def run(args: argparse.Namespace) -> int:
    call_id = f"test-{uuid.uuid4().hex[:8]}"
    text_id = uuid.uuid4().hex[:8]
    prefix = args.path_prefix.rstrip("/")
    url = f"ws://{args.host}:{args.port}{prefix}/ws/{call_id}"
    print(f"[test] connecting {url}", flush=True)
    print(f"[test] voice={args.voice!r}  lang={args.lang!r}  text={args.text[:60]!r}", flush=True)

    pcm = bytearray()
    sample_rate = 24000
    ttfb: float | None = None

    async with websockets.connect(
        url, max_size=100 * 1024 * 1024, open_timeout=10,
        close_timeout=3, ping_interval=None,   # avoid post-stream close/keepalive hangs
    ) as ws:
        req = {
            "type": "synthesize",
            "call_id": call_id,
            "text_id": text_id,
            "text": args.text,
            "streaming": True,
        }
        if args.voice:
            req["voice_id"] = args.voice
        if args.lang:
            req["language"] = args.lang
        if args.speed:
            req["speed"] = args.speed

        t0 = time.perf_counter()
        await ws.send(json.dumps(req))

        saw_final = False
        done_msg: dict = {}
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=(5 if saw_final else 120))
            except asyncio.TimeoutError:
                if saw_final:
                    print("[test] audio_done frame not received (proxy dropped it?) — finalizing", flush=True)
                    break
                print("[test] TIMEOUT waiting for audio", flush=True)
                return 1

            msg, audio = _split_frame(raw)
            mtype = msg.get("type")

            if mtype == "audio_chunk":
                if not audio:  # fallback: PCM came as a separate binary frame
                    audio = await ws.recv()
                    if isinstance(audio, str):
                        audio = audio.encode()
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
                    print(f"[test] first chunk in {ttfb*1000:.0f} ms  "
                          f"(cache_hit={msg.get('cache_hit')})", flush=True)
                sample_rate = msg.get("sample_rate", sample_rate)
                pcm += audio
                if msg.get("is_final"):
                    saw_final = True

            elif mtype == "audio_done":
                done_msg = msg
                break

            elif mtype == "error":
                print(f"[test] SERVER ERROR: {msg.get('error')}", flush=True)
                return 1

    # Connection is now closed (bounded by close_timeout) — safe to finalize.
    total = time.perf_counter() - t0
    audio_s = (len(pcm) // 2) / sample_rate if sample_rate else 0.0
    print(
        f"[test] done  chunks={done_msg.get('chunks')}  audio={audio_s:.2f}s  "
        f"ttfb={(ttfb*1000 if ttfb else 0):.0f}ms  total={total*1000:.0f}ms  "
        f"rtf={done_msg.get('rtf')}  pcm_bytes={len(pcm)}  sr={sample_rate}",
        flush=True,
    )

    if not pcm:
        print("[test] no audio received", flush=True)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)          # int16
        w.setframerate(sample_rate)
        w.writeframes(bytes(pcm))
    print(f"[test] wrote {out}  ({out.stat().st_size} bytes, {sample_rate} Hz mono)", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Test a voice over the FlowTTS WebSocket")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--voice", default=None, help="voice_id alias (omit for default/auto voice)")
    ap.add_argument("--text", default=_DEFAULT_TEXT)
    ap.add_argument("--lang", default=None, help="language id (e.g. hi, en); omit to auto/voice-default")
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--path-prefix", default="", help="URL path prefix if behind a proxy, e.g. /omnivoice-tts")
    ap.add_argument("--out", default="voice_test.wav", help="output WAV path")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
