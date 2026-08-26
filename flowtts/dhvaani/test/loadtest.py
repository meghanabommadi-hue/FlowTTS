#!/usr/bin/env python3
"""Load generator for a RUNNING DhVaani server (WebSocket or REST).

Unlike `bench.py`, which drives the engine in-process, this hits the real
network surface, so it measures what a client actually experiences including
framing, the event loop and the HTTP stack.

    # WebSocket, the production protocol, open-loop at a target arrival rate
    python -m flowtts.dhvaani.test.loadtest ws --url ws://localhost:8080 \
        --voice simran --rps 50 --duration 30

    # REST, OpenAI-compatible streaming
    python -m flowtts.dhvaani.test.loadtest rest --url http://localhost:8000 \
        --voice simran --rps 50 --duration 30 --stream

Arrivals are Poisson (open loop) rather than a fixed pool of workers. A closed
loop hides overload: when the server slows down, a closed loop simply sends
fewer requests and the latency looks fine. Open loop keeps offering the target
rate, so queueing shows up in the percentiles the way it will in production.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

CORPUS = [
    ("hi", "नमस्ते, आपकी ईएमआई दो हज़ार पांच सौ रुपये बकाया है।"),
    ("hi", "कृपया आज ही भुगतान करें, अन्यथा विलंब शुल्क लग सकता है।"),
    ("hi", "आपका खाता नंबर नौ आठ सात छह पांच चार तीन दो एक शून्य है।"),
    ("hi", "क्या मैं ग्राहक से बात कर रही हूं?"),
    ("en", "Your payment of twelve hundred rupees is due today."),
    ("en", "Please confirm your registered mobile number."),
    ("ta", "வணக்கம், உங்கள் கட்டணம் நிலுவையில் உள்ளது."),
    ("te", "నమస్కారం, మీ చెల్లింపు పెండింగ్‌లో ఉంది."),
    ("bn", "নমস্কার, আপনার পেমেন্ট বাকি আছে।"),
    ("mr", "नमस्कार, तुमचे पेमेंट प्रलंबित आहे."),
]


def split_frame(frame: bytes) -> tuple[dict, bytes]:
    """Split a FlowTTS audio frame into its JSON header and the raw PCM.

    The wire format concatenates `json.dumps(header).encode()` with the audio
    bytes in a single WebSocket frame, so the split point is the header's
    closing brace. Searching for the FIRST b"}" is wrong: `call_id` and
    `text_id` are client-supplied and may legitimately contain a brace, which
    would truncate the header and corrupt the audio offset.

    `json.dumps` defaults to ensure_ascii=True, so the header is pure ASCII and
    byte offsets equal character offsets. That makes a depth-tracking scan --
    skipping braces inside strings and honouring backslash escapes -- both
    correct and cheap.
    """
    depth = 0
    in_str = False
    escaped = False
    for i, b in enumerate(frame):
        c = chr(b)
        if in_str:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(frame[: i + 1].decode("utf-8")), frame[i + 1:]
    raise ValueError("malformed frame: no balanced JSON header found")


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))]


class Stats:
    def __init__(self):
        self.ttfb: list[float] = []
        self.total: list[float] = []
        self.audio_s = 0.0
        self.bytes = 0
        self.ok = 0
        self.errors = 0
        self.error_samples: list[str] = []

    def fail(self, msg: str) -> None:
        self.errors += 1
        if len(self.error_samples) < 5:
            self.error_samples.append(msg[:200])

    def report(self, elapsed: float, target_rps: float, label: str, sr: int) -> int:
        n = self.ok + self.errors
        print(f"\n=== {label} ===")
        print(f"  duration           : {elapsed:.1f}s")
        print(f"  requests sent      : {n}   (target {target_rps:.0f} RPS "
              f"-> {target_rps * elapsed:.0f} expected)")
        print(f"  succeeded / failed : {self.ok} / {self.errors}")
        print(f"  achieved RPS       : {self.ok / elapsed:.1f}")
        print(f"  audio generated    : {self.audio_s:.1f}s "
              f"({self.audio_s / elapsed:.1f}x realtime)")
        print(f"  bytes received     : {self.bytes / 2**20:.1f} MiB")
        if self.ttfb:
            print(f"  TTFB  p50/p90/p99  : {pct(self.ttfb, 50):.0f} / "
                  f"{pct(self.ttfb, 90):.0f} / {pct(self.ttfb, 99):.0f} ms   "
                  f"(mean {statistics.mean(self.ttfb):.0f})")
        if self.total:
            print(f"  total p50/p90/p99  : {pct(self.total, 50):.0f} / "
                  f"{pct(self.total, 90):.0f} / {pct(self.total, 99):.0f} ms")
        if self.audio_s:
            print(f"  aggregate RTF      : "
                  f"{sum(self.total) / 1000 / self.audio_s:.3f}")
        for e in self.error_samples:
            print(f"  error: {e}")
        ok = self.errors == 0 and self.ok > 0
        print(f"  VERDICT            : {'OK' if ok else 'ERRORS PRESENT'}")
        return 0 if ok else 1


async def ws_worker(url: str, voice: str, stats: Stats, lang: str, text: str) -> None:
    import websockets

    t0 = time.perf_counter()
    first = None
    total_bytes = 0
    try:
        async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
            await ws.send(json.dumps({
                "text": text, "call_id": f"lt-{random.random():.6f}",
                "text_id": f"t{random.random():.6f}", "voice_id": voice,
                "language": lang, "streaming": True,
            }))
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    header, audio = split_frame(msg)
                    if audio and first is None:
                        first = (time.perf_counter() - t0) * 1000
                    total_bytes += len(audio)
                    continue
                data = json.loads(msg)
                if data.get("type") == "audio_done":
                    stats.ok += 1
                    stats.ttfb.append(first if first is not None else 0.0)
                    stats.total.append((time.perf_counter() - t0) * 1000)
                    stats.bytes += total_bytes
                    stats.audio_s += (total_bytes / 2) / data.get("sample_rate", 24000)
                    return
                if data.get("type") == "error":
                    stats.fail(data.get("error", "unknown"))
                    return
    except Exception as e:
        stats.fail(f"{type(e).__name__}: {e}")


async def rest_worker(session, url: str, voice: str, stats: Stats,
                      lang: str, text: str, stream: bool, sr: int) -> None:
    t0 = time.perf_counter()
    first = None
    total = 0
    payload = {
        "model": "dhvaani-0.5", "input": text, "voice": voice,
        "language": lang, "response_format": "pcm", "stream": stream,
    }
    try:
        async with session.post(f"{url}/v1/audio/speech", json=payload) as r:
            if r.status != 200:
                stats.fail(f"HTTP {r.status}: {(await r.text())[:150]}")
                return
            async for block in r.content.iter_chunked(8192):
                if block and first is None:
                    first = (time.perf_counter() - t0) * 1000
                total += len(block)
        stats.ok += 1
        stats.ttfb.append(first if first is not None else 0.0)
        stats.total.append((time.perf_counter() - t0) * 1000)
        stats.bytes += total
        stats.audio_s += (total / 2) / sr
    except Exception as e:
        stats.fail(f"{type(e).__name__}: {e}")


async def run(args) -> int:
    stats = Stats()
    rng = random.Random(args.seed)
    inflight: set[asyncio.Task] = set()
    session = None
    if args.mode == "rest":
        import aiohttp

        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=args.timeout)
        )

    def spawn() -> asyncio.Task:
        lang, text = rng.choice(CORPUS)
        if args.mode == "ws":
            coro = ws_worker(args.url, args.voice, stats, lang, text)
        else:
            coro = rest_worker(session, args.url, args.voice, stats, lang, text,
                               args.stream, args.sample_rate)
        t = asyncio.create_task(coro)
        inflight.add(t)
        t.add_done_callback(inflight.discard)
        return t

    print(f"load test: mode={args.mode} target={args.rps} RPS for {args.duration}s "
          f"-> {args.url}")
    t0 = time.perf_counter()
    deadline = t0 + args.duration
    interval = 1.0 / args.rps
    while time.perf_counter() < deadline:
        spawn()
        if len(inflight) > args.max_inflight:
            # Backpressure only as a safety valve; hitting this means the server
            # is not keeping up and the numbers below understate the problem.
            await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
        # Poisson arrivals: exponential gaps at the target rate.
        await asyncio.sleep(rng.expovariate(1.0 / interval))

    if inflight:
        await asyncio.wait(inflight, timeout=args.timeout)
    elapsed = time.perf_counter() - t0
    if session is not None:
        await session.close()

    label = f"{args.mode.upper()} {args.url}"
    return stats.report(elapsed, args.rps, label, args.sample_rate)


def main() -> int:
    ap = argparse.ArgumentParser(description="DhVaani load generator")
    ap.add_argument("mode", choices=["ws", "rest"])
    ap.add_argument("--url", required=True, help="ws://host:8080 or http://host:8000")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--rps", type=float, default=20.0, help="Target arrival rate")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--max-inflight", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--stream", action="store_true", help="rest: chunked streaming")
    ap.add_argument("--seed", type=int, default=0)
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
