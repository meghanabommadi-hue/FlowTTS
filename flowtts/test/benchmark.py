#!/usr/bin/env python3
"""
Pipeline position: GATEWAY LOAD TEST — end-to-end LLM throughput benchmark.

Role in pipeline:
  Drives the full gateway+worker pipeline from the client side:
    Text (random Telugu sentences)
      → WebSocket synthesize request → gateway port
        → Redis queue → worker → sglang inference
        → audio_tokens → gateway → WebSocket response
      → save JSON {audio_tokens, llm_s, total_s}

  Does NOT call the decoder — output is raw audio_tokens JSON files that
  can be fed into codec_client.py or decode_json.py for decoder benchmarking.

Call model (persistent connection per port):
  One WebSocket connection is opened per port and kept alive for the duration
  of the test — mirroring a real phone-call where a caller stays connected
  across multiple TTS utterances. Requests within a port are sequential;
  all ports run concurrently via asyncio.gather.

Outputs:
  test/bench_YYYYMMDD_HHMMSS/*.json   — one file per request, contains
    audio_tokens (input for decoder), llm_s, total_s, text, port, call_id.
  test/llm_streaming_benchmark.log    — token chunk timeline: for each
    response, audio_tokens is split into 100-token chunks with interpolated
    timestamps so you can see token generation rate over the run duration.

Latency fields:
  llm_s    — sglang inference time (set by worker, excludes queue wait)
  total_s  — wall time from WebSocket send to response received (includes
             queue wait + inference + network round-trip)

Usage:
    python flowtts/test/benchmark.py --requests 1 --n-ports 100
    python flowtts/test/benchmark.py --total 100 --no-save
    python flowtts/test/benchmark.py --requests 5 --n-ports 10 --length long
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import socket
import statistics
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import websockets
from websockets.exceptions import WebSocketException

# ── Streaming token log ───────────────────────────────────────────────────────
# The LLM returns all tokens at once (no true streaming). We split each
# response into chunks of 100 tokens and interpolate timestamps linearly
# across the known llm_s duration — giving a realistic per-100-token timeline.

LOG_PATH = Path(__file__).parents[2] / "test" / "llm_streaming_benchmark.log"
CHUNK_SIZE = 100  # tokens per logged checkpoint

import re as _re

def _parse_tokens(audio_tokens: str) -> list[int]:
    """Extract integer token IDs from a speech_token string."""
    return [int(x) for x in _re.findall(r"speech_token_(\d+)", audio_tokens)]


class _TokenLog:
    """
    Collects per-call token chunks with interpolated timestamps, then writes
    llm_streaming_benchmark.log (overwritten each run).

    Each entry = one 100-token chunk from one call, timestamped at the
    interpolated moment it would have arrived if tokens streamed linearly.
    """

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._t_start: Optional[float] = None
        # rows: (abs_elapsed_ms, port, chunk_no, chunk_tok, cum_tok, preview)
        self._rows: list[tuple] = []
        self._grand_total = 0

    def start(self) -> None:
        with self._lock:
            self._t_start = time.perf_counter()
            self._rows = []
            self._grand_total = 0

    def record_call(
        self,
        port: int,
        audio_tokens: str,
        llm_s: float,
        call_start_elapsed_ms: float,   # ms since run start when this call began
        text: str,
    ) -> None:
        """
        Split audio_tokens into CHUNK_SIZE chunks, interpolate a timestamp
        for each chunk linearly within [call_start_elapsed_ms,
        call_start_elapsed_ms + llm_s*1000], and record each chunk.
        """
        tokens = _parse_tokens(audio_tokens)
        total = len(tokens)
        if total == 0:
            return

        preview = (text[:35] + "..") if len(text) > 35 else text
        chunks = [tokens[i:i + CHUNK_SIZE] for i in range(0, total, CHUNK_SIZE)]
        n_chunks = len(chunks)
        llm_ms = (llm_s or 0.0) * 1000

        with self._lock:
            for ci, chunk in enumerate(chunks):
                # Interpolated offset within this call's llm window
                frac = (ci + 1) / n_chunks
                interp_ms = call_start_elapsed_ms + frac * llm_ms
                self._rows.append((
                    interp_ms,          # abs elapsed ms from run start
                    port,
                    ci + 1,             # chunk number (1-based)
                    n_chunks,           # total chunks for this call
                    len(chunk),         # tokens in this chunk
                    total,              # total tokens for this call
                    preview,
                ))
            self._grand_total += total

    def flush(self, wall_s: float) -> None:
        with self._lock:
            # Sort all chunks by interpolated timestamp
            sorted_rows = sorted(self._rows, key=lambda r: r[0])

            # Build cumulative across entire run
            cum = 0
            data_lines = []
            for elapsed_ms, port, ci, n_chunks, chunk_tok, call_total, preview in sorted_rows:
                cum += chunk_tok
                pct_call  = (ci * CHUNK_SIZE + chunk_tok) / call_total * 100
                pct_total = cum / self._grand_total * 100 if self._grand_total else 0.0
                data_lines.append(
                    f"{elapsed_ms:>10.0f}ms  :{port:<6}  "
                    f"chunk {ci:>3}/{n_chunks:<3}  "
                    f"{chunk_tok:>4} tok  "
                    f"call%={pct_call:>5.1f}%  "
                    f"run%={pct_total:>5.1f}%  "
                    f"cum={cum:>6}  "
                    f"{preview}"
                )

            header = [
                f"llm_streaming_benchmark  {datetime.now().isoformat(timespec='seconds')}",
                f"chunk_size={CHUNK_SIZE} tokens   wall={wall_s*1000:.0f}ms   "
                f"total_tokens={self._grand_total}   "
                f"tok/s={int(self._grand_total/wall_s) if wall_s else 0}",
                "",
                f"{'elapsed':>11}  {'port':<8}  {'chunk':>12}  {'size':>8}  "
                f"{'call%':>9}  {'run%':>8}  {'cum':>9}  text",
                "-" * 90,
            ]
            footer = [
                "-" * 90,
                f"grand_total={self._grand_total} tokens  "
                f"avg_tok_per_s={int(self._grand_total/wall_s) if wall_s else 0}",
            ]

            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            LOG_PATH.write_text(
                "\n".join(header + data_lines + footer) + "\n", encoding="utf-8"
            )
            print(f"\n  Token stream log → {LOG_PATH}", flush=True)


TOKEN_LOG = _TokenLog()

# ── Sentence pool ────────────────────────────────────────────────────────────
# Short sentences (~5 words) — fast to synthesize, good for throughput tests
SHORT_TEXTS = [
    "నమస్తే! ఎలా ఉన్నారు?",
    "కృపయా సహాయం చేయండి.",
    "ధన్యవాదాలు మీ సహకారానికి.",
    "శుభోదయం! మీరు బాగున్నారా?",
    "అర్థమైందా మీకు?",
    "సరే, కొనసాగించండి.",
    "మళ్ళీ చెప్పగలరా?",
    "సమస్య పరిష్కారమైంది.",
    "మీ పేరు ఏమిటి?",
    "ఒక నిమిషం ఉండండి.",
]

# Medium sentences (~10–15 words) — typical call-centre utterances
MEDIUM_TEXTS = [
    "నేను బజాజ్ ఫైనాన్స్ నుంచి వానీని మాట్లాడుతున్నాను.",
    "ఈ రోజు వాతావరణం చాలా మంచిగా ఉంది.",
    "శుభోదయం! మీరు ఎలా ప్రారంభించారు?",
    "ఈ ప్రాజెక్ట్ చాలా రోజుల నుంచి జరుగుతోంది.",
    "మా కంపెనీ కస్టమర్ సంతృప్తికి ప్రాధాన్యతిస్తుంది.",
    "విద్యార్థులు పరీక్షల కోసం చదువుతున్నారు.",
    "మీ ఖాతాలో నగదు జమ అయింది.",
    "దయచేసి మీ లోన్ వివరాలు చెప్పండి.",
    "మీ అప్లికేషన్ ప్రాసెస్ అవుతోంది.",
    "మేము మీకు త్వరలో తిరిగి కాల్ చేస్తాము.",
    "మీ EMI తేదీ మారిందా?",
    "పేమెంట్ విజయవంతంగా పూర్తైంది.",
    "మీ రిక్వెస్ట్ నమోదు చేయబడింది.",
    "కస్టమర్ కేర్ కు స్వాగతం.",
    "మీరు మా సేవలతో సంతృప్తిగా ఉన్నారా?",
]

# Long sentences (~20+ words) — stress-test for token generation
LONG_TEXTS = [
    "కంప్యూటర్ సైన్స్ లో కొత్త అభివృద్ధులు రోజు రోజుకు వస్తున్నాయి మరియు భవిష్యత్తులో మరిన్ని మార్పులు రానున్నాయి.",
    "మీ రుణ దరఖాస్తు మా బృందం సమీక్షిస్తున్నది మరియు మేము మీకు 24 గంటల్లో తెలియజేస్తాము.",
    "బజాజ్ ఫైనాన్స్ మీకు వ్యక్తిగత రుణాలు, గృహ రుణాలు మరియు వాహన రుణాలు అందిస్తోంది.",
    "మీరు ఎంచుకున్న ప్లాన్ ప్రకారం మీ నెలవారీ వాయిదా మొత్తం నిర్ణయించబడుతుంది.",
    "మా కస్టమర్ సేవా కేంద్రం ప్రతిరోజూ ఉదయం తొమ్మిది నుంచి రాత్రి తొమ్మిది వరకు అందుబాటులో ఉంటుంది.",
    "మీ ఖాతా సురక్షితత కోసం మీరు రెగ్యులర్ గా పాస్ వర్డ్ మార్చాలని మేము సూచిస్తున్నాము.",
    "ఈ నెల మీ లోన్ చెల్లింపు గడువు తేదీ ముగియబోతోంది కాబట్టి దయచేసి సకాలంలో చెల్లించండి.",
    "మీ ఫీడ్ బ్యాక్ మాకు చాలా విలువైనది మరియు మేము దాన్ని మా సేవలను మెరుగుపరచడానికి ఉపయోగిస్తాము.",
]

# All texts combined — used for randomised selection
SAMPLE_TEXTS = SHORT_TEXTS + MEDIUM_TEXTS + LONG_TEXTS


def random_texts(n: int, length: str = "all") -> list[str]:
    """
    Return n random texts sampled with replacement.

    length:
      "short"  — only short sentences  (~5 words)
      "medium" — only medium sentences (~10-15 words)
      "long"   — only long sentences   (~20+ words)
      "all"    — full mixed pool (default)
    """
    pool = {
        "short":  SHORT_TEXTS,
        "medium": MEDIUM_TEXTS,
        "long":   LONG_TEXTS,
        "all":    SAMPLE_TEXTS,
    }.get(length, SAMPLE_TEXTS)
    return [random.choice(pool) for _ in range(n)]
# ─────────────────────────────────────────────────────────────────────────────

WS_MAX_SIZE = 100 * 1024 * 1024  # 100 MB


# ── Helpers ───────────────────────────────────────────────────────────────────

def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _discover_ports(host: str, base_port: int, n_ports: int) -> list[int]:
    candidates = [base_port + i for i in range(n_ports)]
    return [p for p in candidates if _port_open(host, p)]


def _auto_ports_from_gateway(host: str, base_port: int) -> list[int]:
    """Ask /ports on any live gateway for the full list."""
    import urllib.request
    for port in range(base_port, base_port + 20):
        if not _port_open(host, port):
            continue
        try:
            url = f"http://{host}:{port}/ports"
            with urllib.request.urlopen(url, timeout=2) as r:
                data = json.loads(r.read())
                live = data.get("live", [])
                if live:
                    print(f"  Got port list from http://{host}:{port}/ports: {live}")
                    return live
        except Exception:
            continue
    return []


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


# ── Per-call worker — one persistent socket per port ─────────────────────────

async def _call_worker(
    host: str,
    port: int,
    texts: list[str],
    start_idx: int,
    out_dir: Optional[Path],
) -> list[dict]:
    """
    Phone-call model: one persistent WebSocket per port.
    All text requests are sent sequentially through the same socket.
    The connection stays open until all texts are processed.

    Returns a list of result dicts (one per text).
    """
    call_id = f"{host}:{port}"
    url = f"ws://{host}:{port}/ws/{call_id}"
    results = []
    run_t0 = TOKEN_LOG._t_start  # reference start of this benchmark run

    try:
        async with websockets.connect(url, max_size=WS_MAX_SIZE, open_timeout=5) as ws:
            print(f"  [:{port}] connected  ({len(texts)} requests)", flush=True)

            for i, text in enumerate(texts):
                idx = start_idx + i
                text_id = str(uuid.uuid4())

                t0 = time.perf_counter()
                await ws.send(json.dumps({
                    "type":    "synthesize",
                    "call_id": call_id,
                    "text_id": text_id,
                    "text":    text,
                }))

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=120)
                except asyncio.TimeoutError:
                    results.append({
                        "idx": idx, "port": port, "ok": False,
                        "error": "timeout", "total_s": time.perf_counter() - t0,
                        "text": text, "text_id": text_id,
                    })
                    print(f"  [:{port}] #{idx:4d} TIMEOUT", flush=True)
                    continue

                total_s = time.perf_counter() - t0

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError as e:
                    results.append({
                        "idx": idx, "port": port, "ok": False,
                        "error": f"JSON: {e}", "total_s": total_s,
                        "text": text, "text_id": text_id,
                    })
                    continue

                if msg.get("type") == "error":
                    results.append({
                        "idx": idx, "port": port, "ok": False,
                        "error": msg.get("error"), "total_s": total_s,
                        "text": text, "text_id": text_id,
                    })
                    print(f"  [:{port}] #{idx:4d} ERROR: {msg.get('error')}", flush=True)
                    continue

                llm_s        = msg.get("llm_s") or 0.0
                audio_tokens = msg.get("audio_tokens", "")
                token_count  = audio_tokens.count("<|speech_token_") if audio_tokens else 0

                # Log per-100-token chunks with interpolated timestamps
                call_start_elapsed_ms = (t0 - run_t0) * 1000
                if audio_tokens:
                    TOKEN_LOG.record_call(
                        port=port,
                        audio_tokens=audio_tokens,
                        llm_s=llm_s,
                        call_start_elapsed_ms=call_start_elapsed_ms,
                        text=text,
                    )

                # Save JSON output
                file_name = ""
                if out_dir is not None:
                    file_name = f"port{port}_{idx:04d}.json"
                    record = {
                        "idx": idx,
                        "port": port,
                        "call_id": call_id,
                        "text_id": text_id,
                        "text": text,
                        "llm_s": llm_s,
                        "total_s": total_s,
                        "token_count": token_count,
                        "audio_tokens": audio_tokens,
                    }
                    (out_dir / file_name).write_text(
                        json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                print(
                    f"  [:{port}] #{idx:4d}  total={total_s*1000:.0f}ms"
                    + (f"  llm={llm_s*1000:.0f}ms" if llm_s is not None else "")
                    + f"  tokens={token_count}"
                    + f"  {text[:35]!r}",
                    flush=True,
                )

                results.append({
                    "idx": idx, "port": port, "ok": True,
                    "total_s": total_s, "llm_s": llm_s,
                    "token_count": token_count,
                    "file": file_name, "text": text, "text_id": text_id,
                })

            # Socket stays open until this with-block exits naturally
            print(f"  [:{port}] call complete — closing connection", flush=True)

    except (WebSocketException, OSError) as e:
        print(f"  [:{port}] CONNECT FAILED: {e}", flush=True)
        for i, text in enumerate(texts):
            results.append({
                "idx": start_idx + i, "port": port, "ok": False,
                "error": str(e), "total_s": 0.0, "text": text,
            })

    return results


# ── Benchmark runner ──────────────────────────────────────────────────────────

async def run_benchmark(
    host: str,
    ports: list[int],
    n_requests: int,
    out_dir: Optional[Path],
    length: str = "all",
) -> list[dict]:
    """
    Distribute n_requests across ports round-robin.
    All ports run in parallel (each as a persistent call/socket).
    Within each port, requests are sent sequentially.
    """
    texts = random_texts(n_requests, length)

    # Split texts across ports
    port_texts: dict[int, list[str]] = defaultdict(list)
    port_start: dict[int, int] = {}
    for i, text in enumerate(texts):
        port = ports[i % len(ports)]
        if port not in port_start:
            port_start[port] = i
        port_texts[port].append(text)

    print(
        f"\n  Firing {n_requests} requests across {len(ports)} port(s) in parallel",
        flush=True,
    )
    for p in ports:
        n = len(port_texts.get(p, []))
        print(f"    :{p}  →  {n} requests", flush=True)
    print(flush=True)

    TOKEN_LOG.start()

    t0 = time.perf_counter()
    all_results_nested = await asyncio.gather(*[
        _call_worker(host, p, port_texts.get(p, []), port_start.get(p, 0), out_dir)
        for p in ports
    ])
    wall = time.perf_counter() - t0

    results = [r for group in all_results_nested for r in group]

    TOKEN_LOG.flush(wall_s=wall)

    print(f"\n  Wall time: {wall:.2f}s", flush=True)
    return results


# ── Stats printer ─────────────────────────────────────────────────────────────

def _print_stats(results: list[dict], n_requested: int) -> None:
    ok   = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]

    width = 72
    print("\n" + "═" * width)
    print(f"  RESULTS  {len(ok)}/{n_requested} ok   {len(fail)} failed")
    print("═" * width)

    if fail:
        print(f"\n  Failures ({min(len(fail), 5)} shown):")
        for r in fail[:5]:
            print(f"    [{r['idx']:4d}] port={r['port']}  {r.get('error','?')}")
        if len(fail) > 5:
            print(f"    ... and {len(fail)-5} more")

    if not ok:
        return

    total_vals = sorted(r["total_s"] for r in ok)
    llm_vals   = sorted(r["llm_s"] for r in ok if r.get("llm_s") is not None)

    def _stats_line(label: str, vals: list[float]) -> str:
        if not vals:
            return f"  {label}: (no data)"
        ms = [v * 1000 for v in vals]
        avg = statistics.mean(ms)
        std = statistics.stdev(ms) if len(ms) > 1 else 0.0
        return (
            f"  {label}:  "
            f"avg={avg:.0f}ms  min={min(ms):.0f}ms  max={max(ms):.0f}ms  std={std:.0f}ms  "
            f"p50={_percentile(ms,50):.0f}ms  p95={_percentile(ms,95):.0f}ms  p99={_percentile(ms,99):.0f}ms"
        )

    wall_max = max(r["total_s"] for r in ok)
    print(f"\n  Throughput: {len(ok)/wall_max:.2f} req/s  (parallel wall={wall_max*1000:.0f}ms)\n")
    print(_stats_line("total  (end-to-end)", total_vals))
    if llm_vals:
        print(_stats_line("llm    (sglang gen)", llm_vals))

    # Per-port breakdown
    by_port: dict[int, list[dict]] = defaultdict(list)
    for r in ok:
        by_port[r["port"]].append(r)

    print(f"\n  {'PORT':>6}  {'OK':>4}  {'total avg':>10}  {'llm avg':>9}  {'tokens avg':>11}")
    print("  " + "-" * 50)
    for port in sorted(by_port):
        reqs     = by_port[port]
        t_vals   = [r["total_s"] * 1000 for r in reqs]
        l_vals   = [r["llm_s"] * 1000 for r in reqs if r.get("llm_s") is not None]
        tk_vals  = [r.get("token_count", 0) for r in reqs]
        print(
            f"  {port:>6}  {len(reqs):>4}  "
            f"{statistics.mean(t_vals):>8.0f}ms  "
            f"{(statistics.mean(l_vals) if l_vals else 0):>7.0f}ms  "
            f"{(statistics.mean(tk_vals) if tk_vals else 0):>11.0f}"
        )

    # Per-request detail (cap at 50)
    print(f"\n  {'IDX':>5}  {'PORT':>6}  {'total':>8}  {'llm':>8}  {'tokens':>7}  TEXT")
    print("  " + "-" * width)
    shown = sorted(ok, key=lambda r: (r["port"], r["idx"]))[:50]
    for r in shown:
        preview = (r.get("text", "")[:36] + "..") if len(r.get("text", "")) > 36 else r.get("text", "")
        llm_str = f"{r['llm_s']*1000:>7.0f}ms" if r.get("llm_s") is not None else "     n/a"
        print(
            f"  {r['idx']:>5}  {r['port']:>6}  "
            f"{r['total_s']*1000:>7.0f}ms  "
            f"{llm_str}  "
            f"{r.get('token_count', 0):>7}  "
            f"{preview}"
        )
    if len(ok) > 50:
        print(f"  ... and {len(ok)-50} more")
    print("═" * width)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlowTTS benchmark: N ports × M requests (persistent connection per port)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Text length buckets (--length):
  short   ~5 words   — fast, good for throughput tests
  medium  ~10-15 w   — typical call-centre utterances  (default)
  long    ~20+ words — stress-test token generation
  all     mixed pool of all three
        """,
    )
    parser.add_argument("--host",        default="localhost",
                        help="Gateway host (default: localhost)")
    parser.add_argument("--base-port",   type=int, default=8765,
                        help="Base port to scan from (default: 8765)")
    parser.add_argument("--n-ports",     type=int, default=10,
                        help="How many consecutive ports to scan (default: 10)")
    parser.add_argument("--requests",    type=int, default=3,
                        help="Requests PER PORT — ignored if --total is set (default: 3)")
    parser.add_argument("--total",       type=int, default=None,
                        help="TOTAL requests to send, distributed round-robin across live ports")
    parser.add_argument("--length",      default="medium",
                        choices=["short", "medium", "long", "all"],
                        help="Sentence length bucket (default: medium)")
    parser.add_argument("--auto-ports",  action="store_true",
                        help="Ask /ports endpoint for live list instead of scanning")
    parser.add_argument("--no-save",     action="store_true",
                        help="Don't save JSON output files, just measure timing")
    args = parser.parse_args()

    # 1. Discover live ports
    print(f"\nFlowTTS Benchmark  host={args.host}  length={args.length}", flush=True)
    if args.auto_ports:
        print("  Fetching live ports from /ports endpoint...", flush=True)
        live_ports = _auto_ports_from_gateway(args.host, args.base_port)
        if not live_ports:
            print("  /ports gave nothing, falling back to TCP scan.")
            live_ports = _discover_ports(args.host, args.base_port, args.n_ports)
    else:
        print(f"  Scanning {args.base_port}..{args.base_port+args.n_ports-1}...", flush=True)
        live_ports = _discover_ports(args.host, args.base_port, args.n_ports)

    if not live_ports:
        print("No live ports found. Is the server running?\n")
        return

    print(f"  Live ports ({len(live_ports)}): {live_ports}", flush=True)

    # --total overrides --requests
    if args.total is not None:
        total_requests = args.total
        print(f"  Sending {total_requests} total requests across {len(live_ports)} port(s)\n", flush=True)
    else:
        total_requests = args.requests * len(live_ports)
        print(f"  Sending {args.requests} req/port × {len(live_ports)} ports = {total_requests} total\n", flush=True)

    # 2. Output directory
    out_dir: Optional[Path] = None
    if not args.no_save:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parents[2] / "test" / f"bench_{run_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  JSON outputs → {out_dir}/\n", flush=True)

    # 3. Run
    results = asyncio.run(run_benchmark(args.host, live_ports, total_requests, out_dir, args.length))

    # 4. Stats
    _print_stats(results, total_requests)

    if out_dir:
        jsons = sorted(out_dir.glob("*.json"))
        print(f"\n  {len(jsons)} JSON file(s) saved to {out_dir}/\n")


if __name__ == "__main__":
    main()
