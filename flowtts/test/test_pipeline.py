"""End-to-end pipeline smoke test.

Two execution modes:

  Managed (default, --launch):
    The test launches flowtts.server itself with --ctrl-port, waits for it to
    be ready, then opens WebSocket ports on demand via the control API
    (POST /ports/add?port=N) — one port per concurrent request.  The server
    subprocess is killed when the test finishes.

  External (--no-launch):
    Connect to an already-running server.  Ports are resolved from
    --ports / --n-ports / --base-port or auto-discovered.

Test modes (--mode):
  tokens   Send text, receive audio_tokens + audio_base64 from server.
  decoded  Inject pre-built WAV directly into flowtts:decoded:{call_id}.
  worker   Inject audio_tokens into flowtts:audio:{call_id}; DecoderWorker
           decodes and publishes to flowtts:decoded:{call_id}.

Output:
  WAV files saved to /root/FlowTTS/test/pipeline_test_YYYYMMDD_HHMMSS/
  Summary table printed and written as summary.txt.

Usage:
    # Managed — launch server, allocate 9 ports on demand, run 40 requests
    python -m flowtts.test.test_pipeline --requests 40 --concurrency 9

    # External — server already running on 8765-8773
    python -m flowtts.test.test_pipeline --no-launch --n-ports 9 --requests 40

    # command to kill all ports
    kill $(ss -tlnp | grep :8764 | grep -oP 'pid=\K[0-9]+')

    # command to check all open ports
    ss -tlnp | grep python3 2>/dev/null | awk '{print $4}' | sort -t: -k2 -n
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import List, NamedTuple, Optional

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

# Fallback Telugu sentences (short / medium / long mix) used when no bench_* data exists.
_TELUGU_FALLBACK: List[str] = [
    # short
    "నేను రేపు హైదరాబాద్ వెళ్తాను, మీరు వస్తారా?",
    "ఈ పుస్తకం చాలా ఆసక్తికరంగా ఉంది, మీరు తప్పకుండా చదవాలి.",
    "వాతావరణం బాగుంది కాబట్టి, సాయంత్రం పార్కుకు వెళ్దాం.",
    "మీ పేరు ఏమిటి? మీరు ఎక్కడ నుండి వచ్చారు?",
    "ఈ రోజు పని చాలా ఎక్కువగా ఉంది, అయినా పూర్తి చేశాను.",
    "తెలుగు భాష చాలా మధురంగా ఉంటుంది.",
    "మా ఊరిలో ప్రతి సంవత్సరం పెద్ద పండుగ జరుగుతుంది.",
    "కొత్త సినిమా చాలా బాగుంది, మీరు తప్పకుండా చూడండి.",
    "డాక్టర్ గారు చెప్పిన మందులు వేసుకుంటున్నావా?",
    "ఈ వంటకం తయారు చేయడం చాలా సులభం.",

    # medium
    "నేను రేపు హైదరాబాద్ వెళ్తాను, మీరు వస్తారా?",
    "ఈ పుస్తకం చాలా ఆసక్తికరంగా ఉంది, మీరు తప్పకుండా చదవాలి.",
    "వాతావరణం బాగుంది కాబట్టి, సాయంత్రం పార్కుకు వెళ్దాం.",
    "మీ పేరు ఏమిటి? మీరు ఎక్కడ నుండి వచ్చారు?",
    "ఈ రోజు పని చాలా ఎక్కువగా ఉంది, అయినా పూర్తి చేశాను.",
    "తెలుగు భాష చాలా మధురంగా ఉంటుంది.",
    "మా ఊరిలో ప్రతి సంవత్సరం పెద్ద పండుగ జరుగుతుంది.",
    "కొత్త సినిమా చాలా బాగుంది, మీరు తప్పకుండా చూడండి.",
    "డాక్టర్ గారు చెప్పిన మందులు వేసుకుంటున్నావా?",
    "ఈ వంటకం తయారు చేయడం చాలా సులభం."
]


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
    skip_decoder: bool = False,
) -> RequestResult:
    call_id = str(uuid.uuid4())
    text_id = str(uuid.uuid4())
    url = f"ws://localhost:{port}/ws/{call_id}"

    _log(req_id, port, f"connecting → {url}")
    try:
        async with websockets.connect(url, open_timeout=5, max_size=100 * 1024 * 1024) as ws:
            _log(req_id, port, "connected")

            # Use Telugu bench text if available, else built-in Telugu fallback list
            if _BENCH_TEXTS:
                text = _BENCH_TEXTS[req_id % len(_BENCH_TEXTS)]
            else:
                text = _TELUGU_FALLBACK[req_id % len(_TELUGU_FALLBACK)]
            req = {
                "type": "synthesize",
                "call_id": call_id,
                "text_id": text_id,
                "text": text,
                **({"skip_decoder": True} if skip_decoder else {}),
            }
            await ws.send(json.dumps(req))
            _log(req_id, port, "sent synthesize request")

            _log(req_id, port, "waiting for WS response…")
            t0 = time.time()
            raw = await ws.recv()
            latency = round(time.time() - t0, 3)
            msg = json.loads(raw)

            _log(req_id, port, f"received type={msg.get('type')}  latency={latency}s")

            if msg.get("type") == "error":
                _log(req_id, port, f"FAIL gateway error: {msg.get('error')}")
                return RequestResult(req_id, port, False, latency, None,
                                     msg.get("error"), 0, 0, None, None)

            wav_path: Optional[Path] = None
            wav_bytes_len = 0
            token_chars = len(msg.get("audio_tokens", ""))
            llm_s = msg.get("llm_s")
            decode_s = msg.get("decode_s")

            audio_b64 = msg.get("audio_base64", "")
            if not audio_b64 and not skip_decoder:
                _log(req_id, port, f"FAIL audio_base64 missing — msg keys: {list(msg.keys())}")
                return RequestResult(req_id, port, False, latency, None,
                                     "audio_base64 missing", 0, token_chars, llm_s, decode_s)
            if audio_b64:
                wav_data = base64.b64decode(audio_b64)
                wav_bytes_len = len(wav_data)
                wav_path = out_dir / f"req{req_id:04d}_port{port}.wav"
                wav_path.write_bytes(wav_data)
            _log(req_id, port, f"OK  {wav_bytes_len}B WAV → {wav_path.name}  llm_s={llm_s}  decode_s={decode_s}")

            return RequestResult(req_id, port, True, latency, wav_path,
                                 None, wav_bytes_len, token_chars, llm_s, decode_s)

    except Exception as e:
        err = str(e) or type(e).__name__
        _log(req_id, port, f"FAIL {type(e).__name__}: {err}")
        return RequestResult(req_id, port, False, 0.0, None, err, 0, 0, None, None)


# ---------------------------------------------------------------------------
# Server management (managed launch mode)
# ---------------------------------------------------------------------------
_VENV_PYTHON = "/root/CleanTTSData/.venv/bin/python3"
_FLOWTTS_DIR = Path("/root/FlowTTS")
_DEFAULT_CTRL_PORT = 8764


def _ctrl_url(ctrl_port: int, path: str) -> str:
    return f"http://127.0.0.1:{ctrl_port}{path}"


def _ctrl_get(ctrl_port: int, path: str, timeout: float = 2.0):
    with urllib.request.urlopen(_ctrl_url(ctrl_port, path), timeout=timeout) as r:
        return json.loads(r.read())


def _ctrl_post(ctrl_port: int, path: str, timeout: float = 2.0):
    req = urllib.request.Request(_ctrl_url(ctrl_port, path), method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _launch_server(ctrl_port: int, save_audio: Optional[str] = None) -> subprocess.Popen:
    """Start flowtts.server with --ports 0 (no WS ports) + control API."""
    cmd = [
        _VENV_PYTHON, "-m", "flowtts.server",
        "--ports", "0",
        "--ctrl-port", str(ctrl_port),
    ]
    if save_audio:
        cmd += ["--save-audio", save_audio]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_FLOWTTS_DIR)
    proc = subprocess.Popen(
        cmd, cwd=str(_FLOWTTS_DIR), env=env,
        stdout=sys.stdout, stderr=sys.stderr,
    )
    return proc


async def _wait_server_ready(ctrl_port: int, timeout: float = 300.0) -> None:
    """Poll /ready until the model is loaded."""
    deadline = time.time() + timeout
    interval = 2.0
    print(f"[server] waiting for model load (ctrl=:{ctrl_port})…", flush=True)
    while time.time() < deadline:
        try:
            data = _ctrl_get(ctrl_port, "/ready", timeout=1.0)
            if data.get("ready"):
                print(f"[server] ready  existing_ports={data.get('ports')}", flush=True)
                return
        except Exception:
            pass
        await asyncio.sleep(interval)
    raise TimeoutError(f"server not ready after {timeout}s")


async def _open_port(ctrl_port: int, ws_port: int) -> int:
    """Ask the running server to bind ws_port. Returns the port."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: _ctrl_post(ctrl_port, f"/ports/add?port={ws_port}"),
    )
    # Brief wait for the socket to be listening
    for _ in range(20):
        if _port_open(ws_port):
            return ws_port
        await asyncio.sleep(0.05)
    raise OSError(f"port {ws_port} did not open after /ports/add")


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------
async def run_test(
    mode: str,
    n_requests: int,
    out_dir: Path,
    *,
    # managed-launch args
    launch: bool = True,
    ctrl_port: int = _DEFAULT_CTRL_PORT,
    concurrency: int = 9,
    base_port: int = 8765,
    save_audio: Optional[str] = None,
    skip_decoder: bool = False,
    # external-server args
    ports: Optional[List[int]] = None,
) -> List[RequestResult]:
    global _BENCH_TEXTS

    if not _BENCH_TEXTS:
        _BENCH_TEXTS = _load_bench_texts()
        if _BENCH_TEXTS:
            print(f"[bench] loaded {len(_BENCH_TEXTS)} Telugu sentences", flush=True)
        else:
            print(f"[bench] no bench texts, using built-in {len(_TELUGU_FALLBACK)} Telugu sentences", flush=True)

    server_proc: Optional[subprocess.Popen] = None

    if launch:
        # ── Managed: start server, open ports on demand ──────────────────────
        server_proc = _launch_server(ctrl_port, save_audio)
        try:
            await _wait_server_ready(ctrl_port)
        except TimeoutError as e:
            server_proc.kill()
            print(f"[server] FATAL: {e}", flush=True)
            sys.exit(1)

        # Open exactly `concurrency` WS ports starting at base_port
        ws_ports: List[int] = []
        for i in range(concurrency):
            p = base_port + i
            await _open_port(ctrl_port, p)
            ws_ports.append(p)
        print(f"[server] opened {len(ws_ports)} port(s): {ws_ports}", flush=True)
        active_ports = ws_ports

    else:
        # ── External: open ports on demand via ctrl API, or use live ports ────
        if ctrl_port:
            if ports is not None:
                # Explicit port list — open any that aren't bound yet
                needed = ports
                opened, already = [], []
                for p in needed:
                    if not _port_open(p):
                        await _open_port(ctrl_port, p)
                        opened.append(p)
                    else:
                        already.append(p)
                if opened:
                    print(f"[server] opened new port(s): {opened}", flush=True)
                if already:
                    print(f"[server] reusing existing port(s): {already}", flush=True)
                active_ports = needed
            else:
                # No explicit ports — use all ports the server already has open
                data = _ctrl_get(ctrl_port, "/ports")
                active_ports = data.get("ports", [])
                print(f"[server] using {len(active_ports)} existing port(s): {active_ports}", flush=True)
        else:
            # No ctrl port — just use whatever is already live
            if ports is None:
                ports = _resolve_ports(None, base_port, None)
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] checking {len(ports)} port(s)…", flush=True)
            live = [p for p in ports if _port_open(p)]
            dead = [p for p in ports if p not in live]
            if not live:
                print(f"[{ts}] ERROR: no ports reachable: {dead}", flush=True)
                sys.exit(1)
            if dead:
                print(f"[{ts}] WARNING: dropped dead ports: {dead}", flush=True)
            active_ports = live
            print(f"[{ts}] using {len(active_ports)} live port(s): {active_ports}", flush=True)

    worker = None
    worker_task = None

    print(f"\n{'='*60}")
    print(f"mode={mode}  requests={n_requests}  ports={active_ports}")
    print(f"output → {out_dir}")
    print(f"{'='*60}\n")

    tasks = [
        _run_one(i, active_ports[i % len(active_ports)], mode, out_dir, worker,
                 skip_decoder=skip_decoder)
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

    if server_proc is not None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("[server] stopped", flush=True)

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
async def main(args: argparse.Namespace) -> None:
    out_dir = _make_out_dir()

    if args.launch:
        results = await run_test(
            args.mode, args.requests, out_dir,
            launch=True,
            ctrl_port=args.ctrl_port,
            concurrency=args.concurrency,
            base_port=args.base_port,
            save_audio=args.save_audio,
            skip_decoder=args.skip_decoder,
        )
    else:
        # Prefer ctrl API for port discovery if available and no explicit port list given
        if args.ctrl_port and args.ports is None and args.n_ports is None:
            port_list = None  # run_test will fetch from ctrl API
        else:
            port_list = _resolve_ports(args.ports, args.base_port, args.n_ports)
            if port_list:
                print(f"[ports] resolved: {port_list}")
        results = await run_test(
            args.mode, args.requests, out_dir,
            launch=False,
            ctrl_port=args.ctrl_port,
            concurrency=args.concurrency,
            base_port=args.base_port,
            ports=port_list,
        )

    ok = _print_summary(results, args.mode, out_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FlowTTS pipeline smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["synth", "tokens", "decoded", "worker"], default="synth",
                        help="Test mode (default: synth)")
    parser.add_argument("--requests", type=int, default=5,
                        help="Number of requests to send (default: 5)")

    # Managed-launch vs external
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--launch", dest="launch", action="store_true", default=None,
                     help="(default) Launch flowtts.server, open ports on demand")
    grp.add_argument("--no-launch", dest="launch", action="store_false",
                     help="Connect to an already-running server")

    # Managed-launch options
    parser.add_argument("--concurrency", type=int, default=9,
                        help="Number of WS ports to open (managed mode, default: 9)")
    parser.add_argument("--ctrl-port", type=int, default=_DEFAULT_CTRL_PORT,
                        help=f"Server control API port (default: {_DEFAULT_CTRL_PORT})")
    parser.add_argument("--save-audio", type=str, default=None, metavar="DIR",
                        help="Pass --save-audio DIR to the launched server")
    parser.add_argument("--skip-decoder", dest="skip_decoder", action="store_true", default=False,
                        help="Send skip_decoder=true in each WS request (LLM only, no WAV decode)")

    # External-server port selection
    pg = parser.add_mutually_exclusive_group()
    pg.add_argument("--ports", type=str, default=None,
                    help="Explicit comma-separated port list (--no-launch)")
    pg.add_argument("--n-ports", type=int, default=None,
                    help="Number of sequential ports from --base-port (--no-launch)")
    parser.add_argument("--base-port", "--port", dest="base_port", type=int, default=8765,
                        help="Base WS port (default: 8765)")

    args = parser.parse_args()
    # If neither --launch nor --no-launch was given, auto-detect from other flags.
    if args.launch is None:
        _external_hints = {"--ctrl-port", "--n-ports", "--ports", "--no-launch"}
        args.launch = not bool(_external_hints.intersection(sys.argv))
    asyncio.run(main(args))
