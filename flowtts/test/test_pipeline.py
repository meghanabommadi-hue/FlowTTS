"""End-to-end pipeline smoke test (no model required).

Tests Redis ↔ WebSocket plumbing by injecting real audio_tokens directly into
Redis instead of running the LLM. Each request runs on its own WebSocket
connection; requests are distributed round-robin across the given ports.

Port selection (mirrors run.sh convention):
  --base-port P --n-ports N   Sequential: P, P+1, …, P+N-1  (run.sh default)
  --ports 8765,8766,8780      Explicit comma-separated list
  If neither given, auto-discovers live ports starting from 8765.

Modes:
  --mode tokens   No-decode path (decoder.enabled=False):
                    inject audio_tokens → flowtts:audio:{call_id}
                    → gateway forwards raw tokens back over WebSocket.

  --mode decoded  Pre-decoded inject (decoder.enabled=True on gateway):
                    inject audio_base64 → flowtts:decoded:{call_id}
                    → gateway forwards audio_base64 back over WebSocket.

  --mode worker   Full DecoderWorker end-to-end (decoder.enabled=True on gateway):
                    inject audio_tokens → flowtts:audio:{call_id}
                    → DecoderWorker decodes → flowtts:decoded:{call_id}
                    → gateway forwards WAV back over WebSocket.

Output:
  WAV files (modes decoded/worker) are saved to
  /root/FlowTTS/test/pipeline_test_YYYYMMDD_HHMMSS/req{N:04d}_port{port}.wav
  A summary table is printed and written to that directory as summary.txt.

Usage:
    # mirror run.sh: 3 ports from 8765 → 8765, 8766, 8767
    python -m flowtts.test.test_pipeline --mode tokens --requests 9 --base-port 8765 --n-ports 3

    # explicit port list
    python -m flowtts.test.test_pipeline --mode worker --requests 10 --ports 8780,8781

    # auto-discover live ports
    python -m flowtts.test.test_pipeline --mode tokens --requests 5
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime
import io
import json
import socket
import struct
import sys
import time
import uuid
from pathlib import Path
from typing import List, NamedTuple, Optional

import redis.asyncio as aioredis
import websockets

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_TEST_ROOT = Path("/root/FlowTTS/test")


def _make_out_dir() -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = _TEST_ROOT / f"pipeline_test_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Sample audio_tokens
# ---------------------------------------------------------------------------

def _load_sample_tokens() -> str:
    # Only search bench_* directories — never pipeline_test_* or other outputs
    jsons = sorted(_TEST_ROOT.glob("bench_*/*.json"))
    for p in jsons:
        try:
            data = json.loads(p.read_text())
            tokens = data.get("audio_tokens", "")
            if tokens and "<|speech_token_" in tokens:
                print(f"[sample] audio_tokens from {p.relative_to(_TEST_ROOT)} ({len(tokens)} chars)")
                return tokens
        except Exception:
            continue
    print("[sample] no bench_* JSON found, using minimal token fallback")
    return "<|speech_token_0|>" * 50


def _load_bench_texts() -> List[str]:
    """Load Telugu text sentences from bench_* JSON files."""
    texts = []
    for p in sorted(_TEST_ROOT.glob("bench_*/*.json")):
        try:
            data = json.loads(p.read_text())
            t = (data.get("text") or "").strip()
            if t:
                texts.append(t)
        except Exception:
            continue
    return texts


SAMPLE_TOKENS = _load_sample_tokens()
_BENCH_TEXTS: List[str] = []  # loaded lazily on first use


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
class RequestResult(NamedTuple):
    req_id: int
    port: int
    passed: bool
    latency_s: float
    wav_path: Optional[Path]
    error: Optional[str]
    wav_bytes: int          # 0 for tokens mode
    token_chars: int        # 0 for decoded/worker mode
    llm_s: Optional[float]
    decode_s: Optional[float]


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------
async def _redis() -> aioredis.Redis:
    return await aioredis.from_url("redis://localhost:6379/0", decode_responses=False)


async def _inject_tokens(rc: aioredis.Redis, call_id: str, text_id: str) -> None:
    payload = {
        "call_id": call_id,
        "text_id": text_id,
        "audio_tokens": SAMPLE_TOKENS,
        "is_final": True,
        "generated_at": time.time(),
        "llm_s": 0.5,
    }
    await rc.publish(f"flowtts:audio:{call_id}", json.dumps(payload))


async def _inject_decoded(rc: aioredis.Redis, call_id: str, text_id: str) -> None:
    sr = 48000
    n = sr  # 1 second silence
    data_bytes = b"\x00\x00" * n
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data_bytes)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data_bytes)))
    buf.write(data_bytes)
    audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    payload = {
        "call_id": call_id,
        "text_id": text_id,
        "audio_base64": audio_b64,
        "sample_rate": sr,
        "is_final": True,
        "llm_s": 0.5,
        "decode_s": 0.01,
    }
    await rc.publish(f"flowtts:decoded:{call_id}", json.dumps(payload))


# ---------------------------------------------------------------------------
# Single request runner
# ---------------------------------------------------------------------------
def _log(req_id: int, port: int, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] req{req_id:04d} port={port}  {msg}", flush=True)


async def _run_one(
    req_id: int,
    port: int,
    mode: str,
    out_dir: Path,
    worker,       # DecoderWorker instance or None
) -> RequestResult:
    call_id = str(uuid.uuid4())
    text_id = str(uuid.uuid4())
    url = f"ws://localhost:{port}/ws/{call_id}"

    _log(req_id, port, f"connecting → {url}")
    rc = await _redis()
    try:
        async with websockets.connect(url, open_timeout=5, max_size=100 * 1024 * 1024) as ws:
            _log(req_id, port, "connected")

            # Use Telugu bench text if available, else fallback
            if _BENCH_TEXTS:
                text = _BENCH_TEXTS[req_id % len(_BENCH_TEXTS)]
            else:
                text = f"test sentence {req_id}"
            req = {
                "type": "synthesize",
                "call_id": call_id,
                "text_id": text_id,
                "text": text,
            }
            await ws.send(json.dumps(req))
            _log(req_id, port, "sent synthesize request")

            # Small delay so gateway has subscribed before we publish
            await asyncio.sleep(0.2)

            if mode == "tokens":
                await _inject_tokens(rc, call_id, text_id)
                _log(req_id, port, f"injected audio_tokens → flowtts:audio:{call_id[:8]}…")
            elif mode == "decoded":
                await _inject_decoded(rc, call_id, text_id)
                _log(req_id, port, f"injected decoded WAV → flowtts:decoded:{call_id[:8]}…")
            else:  # worker — DecoderWorker picks it up automatically
                await _inject_tokens(rc, call_id, text_id)
                _log(req_id, port, f"injected audio_tokens → flowtts:audio:{call_id[:8]}… (awaiting DecoderWorker)")

            timeout = 60 if mode == "worker" else 10
            _log(req_id, port, f"waiting for WS response (timeout={timeout}s)…")
            t0 = time.time()
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            latency = round(time.time() - t0, 3)
            msg = json.loads(raw)

            _log(req_id, port, f"received type={msg.get('type')}  latency={latency}s")

            if msg.get("type") == "error":
                _log(req_id, port, f"FAIL gateway error: {msg.get('error')}")
                return RequestResult(req_id, port, False, latency, None,
                                     msg.get("error"), 0, 0, None, None)

            wav_path: Optional[Path] = None
            wav_bytes_len = 0
            token_chars = 0
            llm_s = msg.get("llm_s")
            decode_s = msg.get("decode_s")

            if mode == "tokens":
                tokens = msg.get("audio_tokens", "")
                if not tokens:
                    _log(req_id, port, f"FAIL audio_tokens missing — msg keys: {list(msg.keys())}")
                    return RequestResult(req_id, port, False, latency, None,
                                         "audio_tokens missing", 0, 0, llm_s, None)
                token_chars = len(tokens)
                # Save WAV if the server included decoded audio alongside tokens
                audio_b64 = msg.get("audio_base64", "")
                if audio_b64:
                    wav_data = base64.b64decode(audio_b64)
                    wav_bytes_len = len(wav_data)
                    wav_path = out_dir / f"req{req_id:04d}_port{port}.wav"
                    wav_path.write_bytes(wav_data)
                    _log(req_id, port, f"OK  {token_chars} token chars  {wav_bytes_len}B WAV → {wav_path.name}  llm_s={llm_s}  decode_s={decode_s}")
                else:
                    _log(req_id, port, f"OK  {token_chars} token chars  llm_s={llm_s}")

            else:
                audio_b64 = msg.get("audio_base64", "")
                if not audio_b64:
                    _log(req_id, port, f"FAIL audio_base64 missing — msg keys: {list(msg.keys())}")
                    return RequestResult(req_id, port, False, latency, None,
                                         "audio_base64 missing", 0, 0, llm_s, decode_s)
                wav_data = base64.b64decode(audio_b64)
                wav_bytes_len = len(wav_data)
                wav_path = out_dir / f"req{req_id:04d}_port{port}.wav"
                wav_path.write_bytes(wav_data)
                _log(req_id, port, f"OK  {wav_bytes_len}B WAV → {wav_path.name}  llm_s={llm_s}  decode_s={decode_s}")

            return RequestResult(req_id, port, True, latency, wav_path,
                                 None, wav_bytes_len, token_chars, llm_s, decode_s)

    except asyncio.TimeoutError:
        _log(req_id, port, "FAIL timeout waiting for response")
        return RequestResult(req_id, port, False, 0.0, None, "timeout waiting for response", 0, 0, None, None)
    except Exception as e:
        err = str(e) or type(e).__name__
        _log(req_id, port, f"FAIL {type(e).__name__}: {err}")
        return RequestResult(req_id, port, False, 0.0, None, err, 0, 0, None, None)
    finally:
        await rc.aclose()


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------
async def run_test(mode: str, ports: List[int], n_requests: int, out_dir: Path) -> List[RequestResult]:
    global _BENCH_TEXTS
    worker = None
    worker_task = None

    # Load Telugu sentences from bench_* once
    if not _BENCH_TEXTS:
        _BENCH_TEXTS = _load_bench_texts()
        if _BENCH_TEXTS:
            print(f"[bench] loaded {len(_BENCH_TEXTS)} Telugu sentences from bench_*/", flush=True)
        else:
            print("[bench] no bench texts found, using 'test sentence N' fallback", flush=True)

    # Verify ports are reachable before launching requests; drop dead ones
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] checking {len(ports)} port(s)…", flush=True)
    live_ports = [p for p in ports if _port_open(p)]
    dead_ports = [p for p in ports if p not in live_ports]
    if not live_ports:
        print(f"[{ts}] ERROR: no ports reachable: {dead_ports}", flush=True)
        print(f"[{ts}] Start the server first:  nohup bash run.sh --ports N > /tmp/run_sh.log 2>&1 &", flush=True)
        sys.exit(1)
    if dead_ports:
        print(f"[{ts}] WARNING: {len(dead_ports)} port(s) not reachable, dropped: {dead_ports}", flush=True)
    ports = live_ports
    print(f"[{ts}] using {len(ports)} live port(s): {ports}", flush=True)

    if mode == "worker":
        from flowtts.decoder.decoder import DecoderWorker
        print(f"[{ts}] loading DecoderWorker + ncodec…", flush=True)
        worker = DecoderWorker()
        worker_task = asyncio.create_task(worker.run())
        await asyncio.sleep(1.5)  # wait for ncodec load + Redis subscribe
        print(f"[{ts}] DecoderWorker ready", flush=True)

    print(f"\n{'='*60}")
    print(f"mode={mode}  requests={n_requests}  ports={ports}")
    print(f"output → {out_dir}")
    print(f"{'='*60}")

    tasks = [
        _run_one(i, ports[i % len(ports)], mode, out_dir, worker)
        for i in range(n_requests)
    ]
    results: List[RequestResult] = await asyncio.gather(*tasks)

    if worker_task:
        worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except (asyncio.CancelledError, Exception):
            pass

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _print_summary(results: List[RequestResult], mode: str, out_dir: Path) -> bool:
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"SUMMARY  mode={mode}  total={len(results)}  passed={len(passed)}  failed={len(failed)}")
    lines.append(f"{'='*70}")

    header = f"{'req':>4}  {'port':>5}  {'ok':>4}  {'lat(s)':>7}  {'llm_s':>6}  {'dec_s':>6}  {'bytes':>8}  {'tokens':>7}  detail"
    lines.append(header)
    lines.append("-" * 80)

    for r in sorted(results, key=lambda x: x.req_id):
        detail = str(r.wav_path.name) if r.wav_path else (r.error or "")
        lines.append(
            f"{r.req_id:>4}  {r.port:>5}  {'✓' if r.passed else '✗':>4}  "
            f"{r.latency_s:>7.3f}  "
            f"{r.llm_s if r.llm_s is not None else '-':>6}  "
            f"{r.decode_s if r.decode_s is not None else '-':>6}  "
            f"{r.wav_bytes:>8}  {r.token_chars:>7}  {detail}"
        )

    if passed:
        lats  = [r.latency_s for r in passed]
        llms  = [r.llm_s     for r in passed if r.llm_s    is not None]
        decs  = [r.decode_s  for r in passed if r.decode_s is not None]

        def _fmt(vals: list, unit: str = "s") -> str:
            if not vals:
                return "n/a"
            return f"min={min(vals):.3f}{unit}  avg={sum(vals)/len(vals):.3f}{unit}  max={max(vals):.3f}{unit}"

        lines.append(f"\n{'─'*60}")
        lines.append(f"  total latency : {_fmt(lats)}")
        lines.append(f"  llm           : {_fmt(llms)}")
        lines.append(f"  decoder       : {_fmt(decs)}")
        if llms and decs and len(llms) == len(decs):
            overhead = [l - d for l, d in zip(llms, decs)]
            lines.append(f"  llm - decode  : {_fmt(overhead)}  (net inference)")
        lines.append(f"{'─'*60}")
    if failed:
        lines.append(f"\nFailed requests:")
        for r in failed:
            lines.append(f"  req{r.req_id:04d} port={r.port}: {r.error}")

    lines.append(f"\n{'✓ ALL PASSED' if not failed else f'✗ {len(failed)} FAILED'}")
    lines.append(f"{'='*70}")

    text = "\n".join(lines)
    print(text)

    summary_file = out_dir / "summary.txt"
    summary_file.write_text(text)
    print(f"\n[output] {out_dir}/")

    return len(failed) == 0


# ---------------------------------------------------------------------------
# Port discovery
# ---------------------------------------------------------------------------
def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _is_flowtts_gateway(port: int, host: str = "localhost") -> bool:
    """Return True if the port serves a FlowTTS /health endpoint."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1) as resp:
            data = json.loads(resp.read())
            return data.get("service") == "FlowTTS"
    except Exception:
        return False


def _discover_ports(base: int = 8765, scan_range: int = 50) -> List[int]:
    """Return all live FlowTTS gateway ports in [base, base+scan_range)."""
    live = [p for p in range(base, base + scan_range)
            if _port_open(p) and _is_flowtts_gateway(p)]
    return live


def _resolve_ports(
    ports_arg: Optional[str],
    base_port: int,
    n_ports: Optional[int],
) -> List[int]:
    """Resolve ports from CLI args, mirroring run.sh logic.

    Priority:
      1. --ports list (explicit)
      2. --base-port + --n-ports  (sequential, same as run.sh)
      3. auto-discover live ports from base_port
    """
    if ports_arg:
        return [int(p.strip()) for p in ports_arg.split(",") if p.strip()]
    if n_ports is not None:
        # run.sh: BASE_PORT, BASE_PORT+1, …, BASE_PORT+N-1
        return [base_port + i for i in range(n_ports)]
    # Auto-discover
    live = _discover_ports(base=base_port)
    if not live:
        print(f"[ports] no live ports found starting from {base_port}, defaulting to [{base_port}]")
        return [base_port]
    print(f"[ports] auto-discovered: {live}")
    return live


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main(mode: str, ports: List[int], n_requests: int) -> None:
    out_dir = _make_out_dir()
    results = await run_test(mode, ports, n_requests, out_dir)
    ok = _print_summary(results, mode, out_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowTTS pipeline smoke test")
    parser.add_argument("--mode", choices=["tokens", "decoded", "worker"], default="tokens",
                        help="Test mode (default: tokens)")
    parser.add_argument("--requests", type=int, default=5,
                        help="Number of parallel requests (default: 5)")

    # Port selection — mirrors run.sh conventions
    # --port / --base-port : first (or only) port  (run.sh uses --port)
    # --n-ports            : number of sequential ports from base
    # --ports              : explicit comma-separated list
    pg = parser.add_mutually_exclusive_group()
    pg.add_argument("--ports", type=str, default=None,
                    help="Explicit comma-separated port list, e.g. 8765,8766,8780")
    pg.add_argument("--n-ports", type=int, default=None,
                    help="Number of sequential ports from --base-port (run.sh style)")
    parser.add_argument("--base-port", "--port", dest="base_port", type=int, default=8765,
                        help="Base/first port (default: 8765)")

    args = parser.parse_args()

    port_list = _resolve_ports(args.ports, args.base_port, args.n_ports)
    print(f"[ports] using: {port_list}")
    asyncio.run(main(args.mode, port_list, args.requests))
