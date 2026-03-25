"""
FlowTTS WebSocket server.

One sglang engine, many concurrent clients across N ports. Decodes tokens
to WAV and streams base64 audio back. Clients can request disconnection.

Usage:
    # Single port (default 8765)
    python3 -m flowtts.test.ws_server

    # Spin up 3 ports starting at 8765  →  8765, 8766, 8767
    python3 -m flowtts.test.ws_server --ports 3

    # Explicit base port
    python3 -m flowtts.test.ws_server --ports 3 --port 9000   →  9000, 9001, 9002

    # Run parallel test against N ports (server must already be running)
    python3 -m flowtts.test.ws_server --test --ports 3
    python3 -m flowtts.test.ws_server --test --ports 3 --port 9000 --host 192.168.1.10
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

# ── Config ────────────────────────────────────────────────────────────────────
WS_HOST   = "0.0.0.0"
WS_PORT   = 8765

MODEL_DIR = "/root/CleanTTSData/inference/models/MeghanaKap-MiraTTSTelugu"
REF_AUDIO = "/root/CleanTTSData/data/cropped_20260206output.wav"

SAMPLING_PARAMS = {
    "max_new_tokens":     1024,
    "temperature":        0.0,
    "top_p":              0.95,
    "top_k":              50,
    "repetition_penalty": 1.2,
    "min_p":              0.05,
    "ignore_eos":         False,
    "skip_special_tokens": False,
}

DEFAULT_CONTEXT = (
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

TEST_TEXTS = [
    "నమస్కారం, మీరు ఎలా ఉన్నారు?",
    "This is a FlowTTS sample test sentence.",
    "తెలుగు భాషలో మాట్లాడటం చాలా అందంగా ఉంటుంది.",
]
# ─────────────────────────────────────────────────────────────────────────────

# Globals set once at startup — shared across all ports
engine = None
tts_codec = None
context_tokens: str = DEFAULT_CONTEXT
ref_speech_tokens = None

# Active connections: conn_id → asyncio.Task
_active: dict[str, asyncio.Task] = {}


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models() -> None:
    global engine, tts_codec, context_tokens, ref_speech_tokens

    print("Loading TTSCodec...", flush=True)
    from ncodec.codec import TTSCodec
    codec = TTSCodec()

    if REF_AUDIO and os.path.isfile(REF_AUDIO):
        try:
            import librosa
            ref, _ = librosa.load(REF_AUDIO, sr=16000, dtype=np.float32)
            ref, _ = librosa.effects.trim(ref, top_db=20)
            if len(ref) > 5 * 16000:
                ref = ref[:5 * 16000]
            enc = codec.encode(ref)
            if isinstance(enc, tuple) and len(enc) == 2:
                ref_speech_tokens, context_tokens = enc[0], enc[1]
            else:
                context_tokens = enc
            print(f"  Ref audio loaded: {REF_AUDIO}", flush=True)
        except Exception as e:
            print(f"  Ref audio failed ({e}), using default context", flush=True)
    else:
        print(f"  Ref audio not found ({REF_AUDIO}), using default context", flush=True)

    tts_codec = codec

    print("Loading sgl.Engine...", flush=True)
    import sglang as sgl
    engine = sgl.Engine(
        model_path=MODEL_DIR,
        tokenizer_path=MODEL_DIR,
        mem_fraction_static=0.8,
        trust_remote_code=True,
        dtype="bfloat16",
        attention_backend="flashinfer",
        chunked_prefill_size=-1,
    )
    print("Engine ready.\n", flush=True)


# ── Audio decoding ────────────────────────────────────────────────────────────

def _tokens_to_wav_b64(audio_tokens: str) -> str:
    audio = tts_codec.decode(audio_tokens, context_tokens)
    audio = np.asarray(audio)
    if audio.dtype == np.float16:
        audio = audio.astype(np.float32)
    audio = audio.squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=48000, subtype="PCM_16", format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ── Prompt building ───────────────────────────────────────────────────────────

def _build_prompt(text: str) -> str:
    return tts_codec.format_prompt(text, context_tokens, ref_speech_tokens)


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def _handle(websocket) -> None:
    conn_id = str(uuid.uuid4())
    addr = websocket.remote_address
    print(f"[+] {addr}  conn={conn_id}", flush=True)

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                await websocket.send(json.dumps({"error": f"JSON parse error: {e}"}))
                continue

            if msg.get("type") == "kill":
                print(f"[x] Kill requested  conn={conn_id}", flush=True)
                await websocket.send(json.dumps({"type": "killed", "conn_id": conn_id}))
                return

            if msg.get("type") != "synthesize":
                await websocket.send(json.dumps({"error": "Unknown type. Use 'synthesize' or 'kill'"}))
                continue

            text    = msg.get("text", "").strip()
            text_id = msg.get("text_id", str(uuid.uuid4()))

            if not text:
                await websocket.send(json.dumps({"type": "error", "text_id": text_id, "error": "Empty text"}))
                continue

            print(f"  [{addr}] text_id={text_id}  text={text!r:.60}", flush=True)
            t0 = time.perf_counter()

            try:
                prompt = _build_prompt(text)
                result = await engine.async_generate(prompt, SAMPLING_PARAMS)
                audio_tokens = result["text"]
            except Exception as e:
                await websocket.send(json.dumps({"type": "error", "text_id": text_id, "error": str(e)}))
                continue

            t1 = time.perf_counter()

            try:
                audio_b64 = _tokens_to_wav_b64(audio_tokens)
            except Exception as e:
                await websocket.send(json.dumps({"type": "error", "text_id": text_id, "error": f"Decode failed: {e}"}))
                continue

            t2 = time.perf_counter()
            print(
                f"  [{addr}] text_id={text_id}  "
                f"gen={t1-t0:.2f}s  dec={t2-t1:.2f}s  total={t2-t0:.2f}s",
                flush=True,
            )

            await websocket.send(json.dumps({
                "type":         "audio",
                "text_id":      text_id,
                "audio_base64": audio_b64,
                "sample_rate":  48000,
            }))

    except (ConnectionClosedOK, ConnectionClosedError):
        pass
    finally:
        _active.pop(conn_id, None)
        print(f"[-] {addr}  conn={conn_id}", flush=True)


async def _handle_tracked(websocket) -> None:
    conn_id = str(uuid.uuid4())
    task = asyncio.current_task()
    _active[conn_id] = task
    if task:
        task.set_name(f"ws-{conn_id[:8]}")
    await _handle(websocket)


# ── Multi-port server ─────────────────────────────────────────────────────────

async def _serve(base_port: int, n_ports: int) -> None:
    """Load model once, then open n_ports listeners starting at base_port."""
    _load_models()

    ports = [base_port + i for i in range(n_ports)]

    # Open all listeners concurrently
    servers = await asyncio.gather(
        *[websockets.serve(_handle_tracked, WS_HOST, p) for p in ports]
    )

    port_list = "  ".join(f"ws://{WS_HOST}:{p}" for p in ports)
    print(f"Listening on {n_ports} port(s):", flush=True)
    for p in ports:
        print(f"  ws://{WS_HOST}:{p}", flush=True)
    print(flush=True)

    try:
        await asyncio.Future()  # run forever
    finally:
        for s in servers:
            s.close()
        await asyncio.gather(*[s.wait_closed() for s in servers])


# ── Parallel test client ──────────────────────────────────────────────────────

async def _test_one_port(host: str, port: int, out_dir: Path) -> None:
    """Connect to one port, send all TEST_TEXTS sequentially, save WAVs."""
    url = f"ws://{host}:{port}"
    print(f"[port {port}] Connecting to {url}", flush=True)

    try:
        async with websockets.connect(url) as ws:
            print(f"[port {port}] Connected", flush=True)

            for i, text in enumerate(TEST_TEXTS):
                text_id = f"p{port}-{i}"
                t_send = time.perf_counter()
                await ws.send(json.dumps({"type": "synthesize", "text_id": text_id, "text": text}))

                raw = await ws.recv()
                t_recv = time.perf_counter()
                msg = json.loads(raw)

                if msg.get("type") == "error":
                    print(f"[port {port}][{i}] ERROR: {msg['error']}", flush=True)
                    continue

                audio_bytes = base64.b64decode(msg["audio_base64"])
                out_path = out_dir / f"port{port}_text{i}.wav"
                out_path.write_bytes(audio_bytes)
                print(
                    f"[port {port}][{i}] OK  latency={t_recv-t_send:.2f}s  "
                    f"{len(audio_bytes)//1024}KB  → {out_path.name}",
                    flush=True,
                )

            # Kill connection when done
            await ws.send(json.dumps({"type": "kill"}))
            await ws.recv()
            print(f"[port {port}] Done + killed", flush=True)

    except Exception as e:
        print(f"[port {port}] FAILED: {e}", flush=True)


async def _run_test(host: str, base_port: int, n_ports: int) -> None:
    """Fire requests to all ports in parallel, save results to test/."""
    out_dir = Path(__file__).parent / "test"
    out_dir.mkdir(exist_ok=True)

    ports = [base_port + i for i in range(n_ports)]
    print(f"Running parallel test on {n_ports} port(s): {ports}", flush=True)
    print(f"Saving WAVs to {out_dir}/\n", flush=True)

    t0 = time.perf_counter()
    await asyncio.gather(*[_test_one_port(host, p, out_dir) for p in ports])
    elapsed = time.perf_counter() - t0

    wavs = sorted(out_dir.glob("*.wav"))
    print(f"\nDone in {elapsed:.2f}s. {len(wavs)} WAV file(s) in {out_dir}/", flush=True)
    for w in wavs:
        print(f"  {w.name}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FlowTTS WebSocket server")
    parser.add_argument("--test",  action="store_true",
                        help="Run parallel test client instead of server")
    parser.add_argument("--host",  default="localhost",
                        help="Host for test client (default: localhost)")
    parser.add_argument("--port",  type=int, default=WS_PORT,
                        help=f"Base port (default: {WS_PORT})")
    parser.add_argument("--ports", type=int, default=1,
                        help="Number of consecutive ports to open/test (default: 1)")
    args = parser.parse_args()

    if args.test:
        asyncio.run(_run_test(args.host, args.port, args.ports))
    else:
        asyncio.run(_serve(args.port, args.ports))


if __name__ == "__main__":
    main()
