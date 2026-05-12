"""
End-to-end FlowTTS demo client.

This script:
- Connects to the FlowTTS WebSocket gateway.
- Sends a sequence of text chunks over a single call_id (simulated streaming).
- Logs timestamps for each stage so you can verify the full pipeline:
  client → gateway → Redis queue → TTS worker (vLLM) → Redis pub/sub →
  decoder + processing → gateway → client.

Assumptions:
- Redis, the FlowTTS FastAPI app, and the FlowTTS worker are already running.
- FlowTtsSynthesizer is implemented to call your vLLM-backed model.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable, List

import websockets


WS_URL = "ws://localhost:8080/ws"


async def streaming_demo(text_chunks: Iterable[str]) -> None:
    call_id = str(uuid.uuid4())
    url = f"{WS_URL}/{call_id}"

    print(f"[{time.time():.3f}] CALL {call_id} – connecting to {url}")
    async with websockets.connect(url) as ws:
        print(f"[{time.time():.3f}] WebSocket connected")

        for chunk in text_chunks:
            text_id = str(uuid.uuid4())
            payload = {
                "type": "synthesize",
                "call_id": call_id,
                "text_id": text_id,
                "text": chunk,
            }

            t_send = time.time()
            await ws.send(json.dumps(payload))
            print(f"[{t_send:.3f}] → Sent chunk text_id={text_id!s}: {chunk!r}")

            raw = await ws.recv()
            t_recv = time.time()
            msg = json.loads(raw)

            if msg.get("type") == "error":
                raise RuntimeError(msg.get("error", "Unknown error"))
            if msg.get("type") != "audio":
                raise RuntimeError(f"Unexpected message type: {msg.get('type')}")

            audio_b64 = msg["audio_base64"]
            audio_bytes = base64.b64decode(audio_b64)

            out_path = Path(f"flowtts_fullflow_{call_id}_{text_id}.wav")
            out_path.write_bytes(audio_bytes)

            print(
                f"[{t_recv:.3f}] ← Received audio for text_id={text_id} "
                f"(end-to-end latency={t_recv - t_send:.3f}s) → {out_path}"
            )

        print(f"[{time.time():.3f}] CALL {call_id} – done, closing WebSocket")


async def main(argv: List[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    if argv:
        # Use CLI args as a single joined prompt, then chunk it by sentences.
        text = " ".join(argv)
        chunks = [s.strip() for s in text.split(".") if s.strip()]
    else:
        chunks = [
            "This is a FlowTTS end-to-end demo",
            "It sends multiple text chunks over a single call",
            "Each chunk is synthesized by the vLLM-backed model",
        ]

    await streaming_demo(chunks)


if __name__ == "__main__":
    asyncio.run(main())

