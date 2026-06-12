"""
Parallel port tester for FlowTTS.

Discovers which ports in a range are live, then fires random texts at
all of them concurrently. Each port gets its own WebSocket connection
and its own set of randomly-sampled texts.

Usage:
    # Auto-detect ports 8080-8774, use built-in texts
    python flowtts/test/test_ports.py

    # Custom range + more texts per port
    python flowtts/test/test_ports.py --base-port 9000 --n-ports 5 --texts-per-port 4

    # Point at a remote host
    python flowtts/test/test_ports.py --host 192.168.1.10

WAVs are saved to  FlowTTS/test/<run_YYYYMMDD_HHMMSS>/
A summary table is printed at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import random
import socket
import time
import uuid
from datetime import datetime
from pathlib import Path

import websockets
from websockets.exceptions import WebSocketException

# ── Sample texts (Telugu + English) ──────────────────────────────────────────
SAMPLE_TEXTS = [
    "నమస్కారం, మీరు ఎలా ఉన్నారు?",
    "తెలుగు భాషలో మాట్లాడటం చాలా అందంగా ఉంటుంది.",
    "ఈ రోజు వాతావరణం చాలా బాగుంది.",
    "మీకు శుభాకాంక్షలు తెలియజేస్తున్నాను.",
    "నేను మీతో మాట్లాడటం చాలా సంతోషంగా ఉంది.",
    "This is a FlowTTS parallel load test.",
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis is working correctly on this port.",
    "Hello, how are you doing today?",
    "Testing concurrent requests across multiple WebSocket ports.",
    "FlowTTS uses a single sglang engine shared across all connections.",
    "Audio quality should be consistent regardless of which port you connect to.",
]
# ─────────────────────────────────────────────────────────────────────────────


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP connection to host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def discover_ports(host: str, base_port: int, n_ports: int) -> list[int]:
    """Return subset of [base_port .. base_port+n_ports) that are open."""
    candidates = [base_port + i for i in range(n_ports)]
    live = [p for p in candidates if _is_port_open(host, p)]
    return live


# ── Per-port test ─────────────────────────────────────────────────────────────

async def test_port(
    host: str,
    port: int,
    texts: list[str],
    out_dir: Path,
) -> dict:
    """
    Open one WebSocket to ws://host:port/ws/<call_id>, send each text,
    save WAV, return a result dict.
    """
    # call_id = port address so Redis channels and logs are easy to trace
    call_id = f"{host}:{port}"
    url = f"ws://{host}:{port}/ws/{call_id}"
    results = []

    try:
        async with websockets.connect(url, open_timeout=5) as ws:
            for i, text in enumerate(texts):
                text_id = str(uuid.uuid4())
                payload = {
                    "type":    "synthesize",
                    "call_id": call_id,
                    "text_id": text_id,
                    "text":    text,
                }
                t0 = time.perf_counter()
                await ws.send(json.dumps(payload))

                raw = await asyncio.wait_for(ws.recv(), timeout=120)
                elapsed = time.perf_counter() - t0
                msg = json.loads(raw)

                if msg.get("type") == "error":
                    results.append({
                        "text_id": text_id, "text": text,
                        "ok": False, "error": msg.get("error"), "latency": elapsed,
                    })
                    print(f"  [:{port}][{i}] ERROR: {msg.get('error')}", flush=True)
                    continue

                audio_bytes = base64.b64decode(msg["audio_base64"])
                out_path = out_dir / f"port{port}_{i:02d}.wav"
                out_path.write_bytes(audio_bytes)

                results.append({
                    "text_id": text_id, "text": text,
                    "ok": True, "latency": elapsed,
                    "size_kb": len(audio_bytes) // 1024,
                    "file": out_path.name,
                })
                print(
                    f"  [:{port}][{i}] OK  {elapsed:.2f}s  "
                    f"{len(audio_bytes)//1024}KB  {text[:40]!r}",
                    flush=True,
                )

    except (WebSocketException, OSError, asyncio.TimeoutError) as e:
        print(f"  [:{port}] CONNECT FAILED: {e}", flush=True)
        return {"port": port, "connected": False, "error": str(e), "results": []}

    return {"port": port, "connected": True, "results": results}


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(host: str, base_port: int, n_ports: int, texts_per_port: int) -> None:
    # 1. Discover live ports
    print(f"Scanning {host}:{base_port}..{base_port+n_ports-1} for live ports...", flush=True)
    live_ports = discover_ports(host, base_port, n_ports)

    if not live_ports:
        print("No live ports found. Is the server running?")
        return

    print(f"Found {len(live_ports)} live port(s): {live_ports}\n", flush=True)

    # 2. Create timestamped output dir
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parents[2] / "test" / f"run_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving WAVs to {out_dir}/\n", flush=True)

    # 3. Assign random texts to each port (different sample per port)
    port_texts: dict[int, list[str]] = {}
    for port in live_ports:
        port_texts[port] = random.sample(SAMPLE_TEXTS, min(texts_per_port, len(SAMPLE_TEXTS)))

    # 4. Fire all ports in parallel
    t0 = time.perf_counter()
    all_results = await asyncio.gather(
        *[test_port(host, p, port_texts[p], out_dir) for p in live_ports]
    )
    total_elapsed = time.perf_counter() - t0

    # 5. Summary table
    print("\n" + "─" * 62)
    print(f"{'PORT':>6}  {'OK':>4}  {'FAIL':>4}  {'AVG LAT':>8}  {'FILES'}")
    print("─" * 62)
    for r in all_results:
        port = r["port"]
        if not r["connected"]:
            print(f"{port:>6}  {'—':>4}  {'—':>4}  {'—':>8}  connection failed")
            continue
        ok   = [x for x in r["results"] if x["ok"]]
        fail = [x for x in r["results"] if not x["ok"]]
        avg  = (sum(x["latency"] for x in ok) / len(ok)) if ok else 0.0
        files = ", ".join(x["file"] for x in ok)
        print(f"{port:>6}  {len(ok):>4}  {len(fail):>4}  {avg:>7.2f}s  {files}")
    print("─" * 62)
    print(f"Total wall time: {total_elapsed:.2f}s\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FlowTTS parallel port tester")
    parser.add_argument("--host",           default="localhost")
    parser.add_argument("--base-port",      type=int, default=8080)
    parser.add_argument("--n-ports",        type=int, default=10,
                        help="How many consecutive ports to scan (default: 10)")
    parser.add_argument("--texts-per-port", type=int, default=3,
                        help="Random texts to send per port (default: 3)")
    args = parser.parse_args()

    asyncio.run(run(args.host, args.base_port, args.n_ports, args.texts_per_port))


if __name__ == "__main__":
    main()
