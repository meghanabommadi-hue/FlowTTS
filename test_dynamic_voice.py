#!/usr/bin/env python3
"""
Test dynamic voice registration.

1. Sends vanita.wav under the name "rama" (unknown → triggers binary frame read)
2. Sends a second request with "rama" (already encoded → no binary frame)
3. Sends a request with known voice "tara" (no binary frame)
"""

import asyncio
import json
import struct
import wave
import websockets
from pathlib import Path

WS_URL = "ws://localhost:8765"
AUDIO_FILE = "/home/ubuntu/FlowTTS/sample_files/vanita.wav"
OUT_DIR = Path("/home/ubuntu/FlowTTS/test_dynamic_voice_out")
SAMPLE_RATE = 16000

OUT_DIR.mkdir(exist_ok=True)


def save_pcm_as_wav(pcm_bytes: bytes, path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    print(f"  saved → {path}  ({len(pcm_bytes)} bytes PCM)")


async def recv_until_done(ws, label: str):
    """Drain messages until audio_done, saving PCM chunks to WAV."""
    pcm_chunks = []
    while True:
        msg = await ws.recv()
        if isinstance(msg, bytes):
            # server sends: JSON-header bytes + raw PCM concatenated in one frame
            # find the end of the JSON prefix by scanning for the first '{'...'}'
            brace_depth = 0
            json_end = 0
            for i, b in enumerate(msg):
                if b == ord('{'):
                    brace_depth += 1
                elif b == ord('}'):
                    brace_depth -= 1
                    if brace_depth == 0:
                        json_end = i + 1
                        break
            pcm = msg[json_end:]
            if pcm:
                pcm_chunks.append(pcm)
            print(f"  [{label}] chunk: {len(pcm)} bytes PCM")
            continue
        data = json.loads(msg)
        t = data.get("type")
        print(f"  [{label}] {t}", {k: v for k, v in data.items() if k != "type"})
        if t == "audio_done":
            if pcm_chunks:
                out_path = OUT_DIR / f"{label}.wav"
                save_pcm_as_wav(b"".join(pcm_chunks), out_path)
            break
        if t == "error":
            break


async def main():
    with open(AUDIO_FILE, "rb") as f:
        audio_bytes = f.read()
    print(f"Loaded {AUDIO_FILE}: {len(audio_bytes)} bytes\n")

    async with websockets.connect(WS_URL, max_size=100 * 1024 * 1024) as ws:

        # --- 1. Dynamic voice "rama" — first time, send audio bytes ---
        print("=== Request 1: voice_id='rama' (new — sending audio bytes) ===")
        await ws.send(json.dumps({
            "type":        "synthesize",
            "call_id":     "call-abc",
            "text_id":     "t1",
            "text":        "Hello how are you",
            "voice_id":    "rama",
            "sample_rate": 16000,
            "streaming":   False,
        }))
        await ws.send(audio_bytes)  # binary frame with voice audio
        await recv_until_done(ws, "rama-first")

        # --- 2. Same voice "rama" — no audio bytes this time ---
        print("\n=== Request 2: voice_id='rama' (already encoded — no bytes) ===")
        await ws.send(json.dumps({
            "type":      "synthesize",
            "call_id":   "call-abc",
            "text_id":   "t2",
            "text":      "This is a second sentence",
            "voice_id":  "rama",
            "streaming": False,
        }))
        await recv_until_done(ws, "rama-second")

        # --- 3. Known voice "tara" — no audio bytes ---
        print("\n=== Request 3: voice_id='tara' (known — no bytes) ===")
        await ws.send(json.dumps({
            "type":      "synthesize",
            "call_id":   "call-abc",
            "text_id":   "t3",
            "text":      "Namaste",
            "voice_id":  "tara",
            "streaming": False,
        }))
        await recv_until_done(ws, "tara")


asyncio.run(main())
