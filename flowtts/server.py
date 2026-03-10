#!/usr/bin/env python3
"""
Single-process FlowTTS WebSocket server.
Loads sglang + ncodec once; handles all ports in one asyncio event loop.

Usage:
    ./run.sh --ports 100
    python -m flowtts.server --ports 3 --base-port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import datetime
import json
import time
import uuid
from pathlib import Path

import logging
from aiohttp import web

import websockets
from websockets.exceptions import WebSocketException

from flowtts.core.config import settings
from flowtts.decoder.decoder import tensor_to_wav, SAMPLE_RATE
from flowtts.monitoring.metrics import record_call, record_ws_connection_open, record_ws_connection_close, record_ws_error, record_ws_done, ws_log_snapshot, record_port_change
from flowtts.processing.text_normalize import normalize_text
from flowtts.synthesis.models import FlowTtsSynthesizer

# Silence websockets' own logger — we do our own prints
logging.getLogger("websockets").setLevel(logging.CRITICAL)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

_synthesizer: FlowTtsSynthesizer | None = None
_synthesizer_lock = asyncio.Lock()
# 16-worker norm pool: all concurrent requests normalize in parallel so LLM dispatch stays bunched.
# Separate wav pool: WAV encoding never queues behind normalization.
_norm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="norm")
_wav_executor  = concurrent.futures.ThreadPoolExecutor(max_workers=4,  thread_name_prefix="wav_enc")
_audio_out_dir: Path | None = None
_open_ports: set[int] = set()  # tracks all bound WS ports
_llm_log: Path = Path(__file__).parents[1] / "llm.log"
_llm_log_file = None  # opened once in main()
_llm_out_log: Path = Path(__file__).parents[1] / "monitoring" / "llm_outputs.jsonl"
_llm_out_log_file = None  # opened once in main()


def _ts() -> str:
    return time.strftime("%H:%M:%S")

def _tsms() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

def _log(line: str) -> None:
    if _llm_log_file is not None:
        _llm_log_file.write(line + "\n")
        _llm_log_file.flush()


async def _get_synthesizer() -> FlowTtsSynthesizer:
    global _synthesizer
    if _synthesizer is not None:
        return _synthesizer
    async with _synthesizer_lock:
        if _synthesizer is None:  # re-check after acquiring lock
            s = FlowTtsSynthesizer()
            print(f"[{_ts()}] loading model...", flush=True)
            await s.initialize()
            print(f"[{_ts()}] model ready", flush=True)
            _synthesizer = s
    return _synthesizer


async def _handle_request(
    ws: websockets.ServerConnection,
    send_lock: asyncio.Lock,
    data: dict,
    conn_id: str,
    port: int,
    ts_text_recv: str,
) -> None:
    """Run one TTS request; send_lock keeps JSON+WAV frame pairs atomic."""
    call_id = data.get("call_id") or conn_id
    text_id = data.get("text_id") or str(uuid.uuid4())
    raw_text = (data.get("text") or "").strip()

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(_norm_executor, normalize_text, raw_text)

    _log(f"{ts_text_recv}  RECV port={port}  text_id={text_id}  call_id={call_id}  text={text[:60]!r}")
    print(f"[{ts_text_recv}] :{port} {call_id}  req  {text[:60]!r}", flush=True)

    try:
        synth = await _get_synthesizer()
        t0 = time.perf_counter()
        ts_llm_start = _tsms()
        _log(f"{ts_llm_start}  IN   port={port}  text_id={text_id}  call_id={call_id}  text={text}")
        # No wait_for wrapper — the extra Task it creates adds scheduling overhead
        # when 50 coroutines are in-flight simultaneously.
        audio_tokens = await synth.synthesize(text)
        llm_s = round(time.perf_counter() - t0, 4)
        llm_ms = round(llm_s * 1000)
        ts_tokens_ready = _tsms()
        _log(f"{ts_tokens_ready}  OUT  port={port}  text_id={text_id}  call_id={call_id}  llm_ms={llm_ms}")

        token_count = audio_tokens.count("<|speech_token_")

        if _llm_out_log_file is not None:
            _llm_out_log_file.write(json.dumps({
                "ts": ts_tokens_ready,
                "call_id": call_id,
                "text_id": text_id,
                "port": port,
                "text": text,
                "audio_tokens": audio_tokens,
                "token_count": token_count,
                "llm_ms": llm_ms,
            }, ensure_ascii=False) + "\n")

        # Batch decode: all concurrent requests across all connections are
        # coalesced by TTSCodec's internal batch queue into one GPU forward pass.
        codec = synth._tts_codec
        ctx = synth._context_tokens
        td = time.perf_counter()
        wav_tensor = await codec.decode_async(audio_tokens, ctx)
        decode_s = round(time.perf_counter() - td, 4)

        tw = time.perf_counter()
        decoded = await asyncio.get_event_loop().run_in_executor(
            _wav_executor, tensor_to_wav, wav_tensor
        )
        wav_s = round(time.perf_counter() - tw, 4)

        record_call(
            call_id=call_id,
            text_id=text_id,
            port=port,
            text=text,
            token_count=token_count,
            llm_s=llm_s,
            decode_s=decode_s,
            wav_bytes=len(decoded.wav_bytes),
            ts=ts_tokens_ready,
        )

        if _audio_out_dir is not None:
            wav_file = _audio_out_dir / f"{text_id}.wav"
            wav_file.write_bytes(decoded.wav_bytes)
            print(f"[{_ts()}] :{port}  saved → {wav_file}", flush=True)

        async with send_lock:
            await ws.send(json.dumps({
                "type": "audio",
                "call_id": call_id,
                "text_id": text_id,
                "text": text,
                "audio_tokens": audio_tokens,
                "sample_rate": SAMPLE_RATE,
                "wav_bytes": len(decoded.wav_bytes),
                "is_final": True,
                "llm_s": llm_s,
                "decode_s": decode_s,
            }))
            await ws.send(decoded.wav_bytes)

        ts_audio_sent = _tsms()
        total_s = llm_s + decode_s + wav_s
        print(
            f"[{ts_audio_sent}] :{port} {call_id}  done"
            f"  llm={llm_ms}ms"
            f"  decode={round(decode_s*1000)}ms"
            f"  wav_enc={round(wav_s*1000)}ms"
            f"  total={round(total_s*1000)}ms"
            f"  tokens={token_count}"
            f"  wav={len(decoded.wav_bytes)}B",
            flush=True,
        )

        record_ws_done(
            call_id,
            port=port,
            text_id=text_id,
            token_count=token_count,
            llm_ms=llm_ms,
            decode_ms=round(decode_s * 1000),
            total_ms=round(total_s * 1000),
            wav_bytes=len(decoded.wav_bytes),
            ts_text_recv=ts_text_recv,
            ts_llm_start=ts_llm_start,
            ts_tokens_ready=ts_tokens_ready,
            ts_audio_sent=ts_audio_sent,
        )

    except Exception as e:
        ts_err = _tsms()
        print(f"[{ts_err}] :{port} {call_id}  ERROR: {e}", flush=True)
        record_ws_error(call_id, port=port, text_id=text_id, error=str(e))
        async with send_lock:
            try:
                await ws.send(json.dumps({
                    "type": "error", "call_id": call_id, "text_id": text_id,
                    "error": str(e),
                }))
            except Exception:
                pass


async def handle_connection(ws: websockets.ServerConnection, port: int) -> None:
    """Dispatch each incoming message as an independent asyncio Task."""
    peer = ws.remote_address
    conn_id = f"{peer[0]}:{peer[1]}"
    print(f"[{_ts()}] :{port} connected  peer={conn_id}", flush=True)
    record_ws_connection_open(conn_id, port=port)

    send_lock = asyncio.Lock()
    active_tasks: set[asyncio.Task] = set()

    try:
        async for raw in ws:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                async with send_lock:
                    await ws.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
                continue

            text = (data.get("text") or "").strip()
            if not text:
                call_id = data.get("call_id") or conn_id
                text_id = data.get("text_id") or str(uuid.uuid4())
                async with send_lock:
                    await ws.send(json.dumps({
                        "type": "error", "call_id": call_id, "text_id": text_id,
                        "error": "Missing text",
                    }))
                continue

            ts_text_recv = _tsms()
            task = asyncio.create_task(
                _handle_request(ws, send_lock, data, conn_id, port, ts_text_recv)
            )
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

    except WebSocketException:
        pass  # client disconnected — normal
    except Exception as e:
        print(f"[{_ts()}] :{port} connection error: {e}", flush=True)
    finally:
        for task in list(active_tasks):
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        record_ws_connection_close(conn_id, port=port)
        print(f"[{_ts()}] :{port} disconnected  peer={conn_id}", flush=True)


_WARMUP_SENTENCES = [
    # short
    "नमस्ते, मैं आपकी कैसे मदद कर सकती हूं?",
    "क्या आप अपना नाम बता सकते हैं?",
    "कृपया थोड़ा इंतज़ार करें.",
    "आपकी समस्या हल हो गई है.",
    "हम जल्द ही आपसे संपर्क करेंगे.",
    "आपका भुगतान सफलतापूर्वक हो गया है.",
    "क्या आप मुझे और जानकारी दे सकते हैं?",
    "आपका account number क्या है?",
    "क्या यह सही समय है बात करने का?",
    "मैं आपकी पूरी मदद करने के लिए यहाँ हूं.",
    # short with numbers
    "आपका खाता नंबर ९८७६५४३२१० है, कृपया confirm करें.",
    "आपका बकाया Rs. 2500 है, कृपया आज ही जमा करें.",
    "आपकी EMI Rs. 3750 हर महीने देय है.",
    "आपका बकाया ₹२,५०० है, कृपया आज ही जमा करें.",
    "आपका भुगतान ₹१०,००० सफलतापूर्वक हो गया है.",
    # medium
    "नमस्ते. मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से. क्या मैं customer name से बात कर रही हूं?",
    "आपके loan की किस्त अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "हमारे रिकॉर्ड के अनुसार आपका बकाया amount ₹५,००० है, कृपया जल्द से जल्द इसे जमा करें.",
    "आपकी EMI की due date निकल चुकी है, late charge से बचने के लिए आज ही payment करें.",
    "क्या आप अपनी personal details verify कर सकते हैं ताकि हम आपके account की जानकारी दे सकें?",
    "आपके loan की किस्त ₹३,७५० अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "आपकी EMI की due date ३० अप्रैल निकल चुकी है, late charge से बचने के लिए आज ही payment करें.",
    "आपके account नंबर ४५६७८९०१२३ पर ₹१५,००० का loan approve हुआ है, क्या आप details verify करेंगे?",
    "आपके loan की किस्त Rs. 3750 अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "आपके account number 4567890123 पर Rs. 15000 का loan approve हुआ है, क्या आप details verify करेंगे?",
    # medium — extra variety
    "आपकी loan application हमें मिल गई है और हम उसे process कर रहे हैं.",
    "क्या आप अपना registered mobile number confirm कर सकते हैं?",
    "आपकी next EMI की date 15 तारीख है, कृपया समय पर payment करें.",
    "हम आपको एक OTP भेज रहे हैं, कृपया उसे verify करें.",
    "आपका loan account successfully activate हो गया है.",
    # long
    "आपकी loan application approve हो गई है और ₹५०,००० सीधे आपके bank account ७८९०१२३४५६ में transfer कर दिए जाएंगे, जिसमें २ से ३ कार्य दिवस लग सकते हैं.",
    "हमारी company की policy के अनुसार अगर payment ३० दिनों के अंदर नहीं होती तो आपके credit score पर असर पड़ सकता है, इसलिए कृपया समय पर ₹८,२५० का भुगतान करें.",
    "आप हमारे mobile app के माध्यम से अपनी ₹४,५०० की EMI pay कर सकते हैं, इसके अलावा NEFT, IMPS, या UPI का भी उपयोग किया जा सकता है.",
    "आपकी loan application approve हो गई है और Rs. 50000 सीधे आपके bank account 7890123456 में transfer कर दिए जाएंगे, जिसमें 2 से 3 कार्य दिवस लग सकते हैं.",
    "हमारी company की policy के अनुसार अगर payment 30 दिनों के अंदर नहीं होती तो आपके credit score पर असर पड़ सकता है, इसलिए कृपया समय पर Rs. 8250 का भुगतान करें.",
    # long — extra variety
    "आपकी loan application approve हो गई है और loan amount सीधे आपके bank account में transfer कर दी जाएगी, जिसमें दो से तीन कार्य दिवस लग सकते हैं.",
    "आप हमारे mobile app के माध्यम से अपनी EMI pay कर सकते हैं, इसके अलावा NEFT, IMPS, या UPI का भी उपयोग किया जा सकता है, और किसी भी समस्या के लिए हमारी customer care team हमेशा available है.",
    "हमारे records के अनुसार आपका loan account number 1234567890 है और आपकी monthly EMI Rs. 4500 है जो हर महीने की 5 तारीख को deduct होती है.",
    "आपकी payment successfully receive हो गई है और आपका account अब up to date है, अगर आपको कोई और जानकारी चाहिए तो हमें call करें.",
]


async def _warmup(synth: FlowTtsSynthesizer) -> None:
    if not _WARMUP_SENTENCES:
        return
    # Build a batch of exactly 40 sentences by cycling through the list.
    batch_size = 40
    sentences = [_WARMUP_SENTENCES[i % len(_WARMUP_SENTENCES)] for i in range(batch_size)]
    print(f"[{_ts()}] warmup: running {batch_size} sentences concurrently...", flush=True)
    t0 = time.perf_counter()

    async def _one(sentence: str) -> bool:
        try:
            await synth.synthesize(normalize_text(sentence))
            return True
        except Exception as e:
            print(f"[{_ts()}] warmup sentence failed: {e}", flush=True)
            return False

    results = await asyncio.gather(*[_one(s) for s in sentences])
    ok = sum(results)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[{_ts()}] warmup done  {ok}/{batch_size} ok  ({elapsed:.0f}ms total)", flush=True)


async def _warmup_port(port: int, sentence: str) -> None:
    """Send one synthesize request through the WS handler on *port* to prime it."""
    url = f"ws://127.0.0.1:{port}/ws/warmup"
    try:
        async with websockets.connect(
            url,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
            open_timeout=10,
        ) as ws:
            await ws.send(json.dumps({
                "type": "synthesize",
                "call_id": "warmup",
                "text_id": "warmup",
                "text": sentence,
            }))
            await ws.recv()
    except Exception as e:
        print(f"[{_ts()}] warmup port {port} failed: {e}", flush=True)


async def _warmup_all_ports(ports: list[int]) -> None:
    """Warm up every bound WS port concurrently.

    All warmup decode requests are coalesced by TTSCodec's internal batch queue
    into one GPU forward pass, so firing all ports simultaneously is safe.
    """
    sentence = settings.tts_model.warmup_sentence
    if not sentence or not ports:
        return
    print(f"[{_ts()}] warming up {len(ports)} port(s) concurrently...", flush=True)
    t0 = time.perf_counter()
    await asyncio.gather(*[_warmup_port(p, sentence) for p in ports])
    print(f"[{_ts()}] all ports warmed up  ({(time.perf_counter()-t0)*1000:.0f}ms)", flush=True)


async def _bind_ws_port(port: int) -> bool:
    """Bind a new WebSocket listener on *port*. Returns False if already bound."""
    if port in _open_ports:
        return False
    async def handler(ws: websockets.ServerConnection, p: int = port) -> None:
        await handle_connection(ws, p)
    await websockets.serve(
        handler, "0.0.0.0", port,
        ping_interval=30, ping_timeout=30,
        max_size=100 * 1024 * 1024,
    )
    _open_ports.add(port)
    record_port_change(_open_ports)
    print(f"[{_ts()}] opened ws://0.0.0.0:{port}", flush=True)
    return True


# ---------------------------------------------------------------------------
# HTTP control API  (aiohttp)
# ---------------------------------------------------------------------------
async def _http_add_port(req: web.Request) -> web.Response:
    """POST /ports/add?port=N  — bind a new WS port while the server is running."""
    try:
        port = int(req.rel_url.query["port"])
    except (KeyError, ValueError):
        return web.Response(status=400, text="missing or invalid ?port=N")
    if port < 1024 or port > 65535:
        return web.Response(status=400, text="port out of range")
    already = port in _open_ports
    if not already:
        await _bind_ws_port(port)
    return web.json_response({"port": port, "opened": not already})


async def _http_list_ports(req: web.Request) -> web.Response:
    """GET /ports  — list all currently open WS ports."""
    return web.json_response({"ports": sorted(_open_ports)})


async def _http_ready(req: web.Request) -> web.Response:
    """GET /ready  — 200 once the model is loaded."""
    if _synthesizer is None:
        return web.Response(status=503, text="loading")
    return web.json_response({"ready": True, "ports": sorted(_open_ports)})


async def _http_ws_log(req: web.Request) -> web.Response:
    """GET /ws/log  — last 20 WS events (open/done/error/close)."""
    return web.json_response(ws_log_snapshot())


async def _http_metrics(req: web.Request) -> web.Response:
    """GET /metrics  — Prometheus scrape endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    # aiohttp rejects content_type values that include charset (e.g. "text/plain; charset=utf-8")
    ct = CONTENT_TYPE_LATEST.split(";")[0].strip()
    return web.Response(body=generate_latest(), content_type=ct)


async def _run_control_api(ctrl_port: int) -> None:
    app = web.Application()
    app.router.add_post("/ports/add", _http_add_port)
    app.router.add_get("/ports",      _http_list_ports)
    app.router.add_get("/ready",      _http_ready)
    app.router.add_get("/metrics",    _http_metrics)
    app.router.add_get("/ws/log",     _http_ws_log)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", ctrl_port)
    await site.start()
    print(f"[{_ts()}] control API  http://127.0.0.1:{ctrl_port}", flush=True)


async def run_server(base_port: int, n_ports: int, ctrl_port: int | None = None) -> None:
    # Load model once before binding ports
    synth = await _get_synthesizer()
    await _warmup(synth)

    # Start HTTP control API if requested
    if ctrl_port:
        await _run_control_api(ctrl_port)

    # Bind initial WS ports
    initial_ports = [base_port + i for i in range(n_ports)]
    for p in initial_ports:
        await _bind_ws_port(p)

    # Warmup each port — primes connection handling concurrently
    await _warmup_all_ports(initial_ports)

    print(f"\n[{_ts()}] FlowTTS  {len(_open_ports)} port(s) ready:", flush=True)
    for p in sorted(_open_ports):
        print(f"  ws://0.0.0.0:{p}", flush=True)
    print(flush=True)

    await asyncio.Future()  # run forever


def main() -> None:
    global _audio_out_dir

    parser = argparse.ArgumentParser(description="FlowTTS single-process WebSocket server")
    parser.add_argument("--base-port", type=int, default=settings.ws.port,
                        help=f"First port to bind (default: {settings.ws.port})")
    parser.add_argument("--ports", type=int, default=1,
                        help="Number of WebSocket ports to open (default: 1)")
    parser.add_argument("--save-audio", type=str, default=None, metavar="DIR",
                        help="Directory to save decoded WAV files (one per request)")
    parser.add_argument("--ctrl-port", type=int, default=None, metavar="PORT",
                        help="HTTP control API port for on-demand WS port binding (e.g. 8764)")
    args = parser.parse_args()

    global _llm_log_file, _llm_out_log_file
    _llm_log_file = open(_llm_log, "w", buffering=1)  # line-buffered, overwrites each run
    _llm_out_log.parent.mkdir(parents=True, exist_ok=True)
    _llm_out_log_file = open(_llm_out_log, "a", buffering=1)  # line-buffered, append across restarts

    if args.save_audio:
        _audio_out_dir = Path(args.save_audio)
        _audio_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[FlowTTS] Saving audio to {_audio_out_dir}/", flush=True)

    try:
        asyncio.run(run_server(args.base_port, args.ports, args.ctrl_port))
    except KeyboardInterrupt:
        print("\n[FlowTTS] Stopped.")


if __name__ == "__main__":
    main()
