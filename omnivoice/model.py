"""
Hosts OmniVoice behind an HTTP API.

Loads the model once at startup, wraps it in MicroBatchServer (batched
codec-decode + request microbatching, see microbatch_server.py), and
exposes a single synchronous TTS endpoint. Requests made while the model
is generating for other callers get folded into the same generate() batch
where possible instead of queueing serially.

Run:
    python model.py --host 0.0.0.0 --port 8000

Then hit POST /generate (see infer_client.py for a client, or commands.md
for curl examples).
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from microbatch_server import MicroBatchServer, TTSRequest

MODEL_ID = os.environ.get("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
DEVICE = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")
NUM_STEP = int(os.environ.get("OMNIVOICE_NUM_STEP", "16"))
MAX_BATCH_SIZE = int(os.environ.get("OMNIVOICE_MAX_BATCH_SIZE", "24"))

app = FastAPI(title="OmniVoice TTS server")

model = None
server: Optional[MicroBatchServer] = None


class GenerateRequest(BaseModel):
    text: str
    language: Optional[str] = None
    ref_audio: Optional[str] = None  # path to a reference wav/mp3 on the server
    ref_text: Optional[str] = None
    instruct: Optional[str] = None
    speed: Optional[float] = None


class GenerateResponse(BaseModel):
    audio_b64: str  # base64-encoded WAV bytes
    sample_rate: int
    queue_wait_s: float
    generate_s: float
    total_s: float
    batch_size: int


@app.on_event("startup")
async def load_model() -> None:
    global model, server
    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(MODEL_ID, device_map=DEVICE, dtype=torch.float16)
    server = MicroBatchServer(model, max_batch_size=MAX_BATCH_SIZE, num_step=NUM_STEP)
    await server.start()


@app.on_event("shutdown")
async def stop_server() -> None:
    if server is not None:
        await server.stop()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    if server is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    result = await server.submit(
        TTSRequest(
            text=req.text,
            language=req.language,
            ref_audio=req.ref_audio,
            ref_text=req.ref_text,
            instruct=req.instruct,
            speed=req.speed,
        )
    )

    buf = io.BytesIO()
    sf.write(buf, result.audio, result.sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return GenerateResponse(
        audio_b64=audio_b64,
        sample_rate=result.sample_rate,
        queue_wait_s=result.queue_wait_s,
        generate_s=result.generate_s,
        total_s=result.total_s,
        batch_size=result.batch_size,
    )


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Host OmniVoice as an HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
