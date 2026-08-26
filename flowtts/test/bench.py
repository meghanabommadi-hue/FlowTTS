#!/usr/bin/env python3
"""Pipeline position: BENCHMARK CLIENT — measures what the service actually does.

Role in pipeline:
  Drives the running server over HTTP the way a real client does and reports the
  numbers that decide whether it is fit for purpose: time-to-first-byte, total
  latency, real-time factor and error rate, at a given concurrency.

  TTFB is measured at the socket — the wall-clock gap between sending the
  request and the first audio byte arriving — not from a server-side log. A
  server-side number excludes queueing, which is exactly where the latency goes
  when concurrency rises, so it flatters the result at precisely the load you
  care about.

Usage:
    python -m flowtts.test.bench --url http://127.0.0.1:9000 --concurrency 8
    python -m flowtts.test.bench --sweep 1,2,4,8,16,32,64,100 --requests 200
    python -m flowtts.test.bench --profile fast --voice anika --language hi
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

# Realistic Indian voice-bot traffic: short IVR turns, medium collections
# prompts, and a long paragraph, in the mix a live deployment actually sees.
DEFAULT_TEXTS = [
    "नमस्ते, मैं आपकी कैसे मदद कर सकती हूं?",
    "कृपया थोड़ा इंतज़ार करें।",
    "आपका भुगतान सफलतापूर्वक हो गया है।",
    "आपका बकाया ₹2,500 है, कृपया आज ही भुगतान करें।",
    "आपकी EMI ₹3,750 की due date निकल चुकी है, late charge से बचने के लिए आज ही payment करें।",
    "नमस्ते, मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से। "
    "क्या मैं customer name से बात कर रही हूं?",
    "आपकी loan application approve हो गई है और ₹50,000 सीधे आपके bank account में "
    "transfer कर दिए जाएंगे, जिसमें दो से तीन कार्य दिवस लग सकते हैं।",
]


@dataclass
class Result:
    ok: bool
    ttfb_ms: float = 0.0
    total_ms: float = 0.0
    audio_s: float = 0.0
    bytes_out: int = 0
    error: str = ""


@dataclass
class Summary:
    concurrency: int
    results: list[Result] = field(default_factory=list)

    @property
    def ok(self) -> list[Result]:
        return [r for r in self.results if r.ok]

    @staticmethod
    def _pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

    def report(self, wall_s: float) -> dict:
        good = self.ok
        ttfb = [r.ttfb_ms for r in good]
        total = [r.total_ms for r in good]
        audio = sum(r.audio_s for r in good)
        return {
            "concurrency": self.concurrency,
            "requests": len(self.results),
            "ok": len(good),
            "failed": len(self.results) - len(good),
            "wall_s": round(wall_s, 2),
            "rps": round(len(good) / wall_s, 2) if wall_s else 0.0,
            "audio_s": round(audio, 1),
            "realtime_x": round(audio / wall_s, 1) if wall_s else 0.0,
            "ttfb_p50": round(self._pct(ttfb, 0.50), 1),
            "ttfb_p90": round(self._pct(ttfb, 0.90), 1),
            "ttfb_p99": round(self._pct(ttfb, 0.99), 1),
            "ttfb_max": round(max(ttfb), 1) if ttfb else 0.0,
            "total_p50": round(self._pct(total, 0.50), 1),
            "total_p99": round(self._pct(total, 0.99), 1),
            "rtf_p50": round(statistics.median(
                [r.total_ms / 1000 / r.audio_s for r in good if r.audio_s > 0]) , 3)
            if good else 0.0,
        }


async def _one(session, url: str, payload: dict) -> Result:
    """One streaming request, timed at the socket."""
    started = time.perf_counter()
    first_byte: float | None = None
    total_bytes = 0
    try:
        async with session.post(f"{url}/v1/tts/stream", json=payload) as response:
            if response.status != 200:
                return Result(ok=False, error=f"HTTP {response.status}")
            async for chunk in response.content.iter_chunked(4096):
                if not chunk:
                    continue
                if first_byte is None:
                    first_byte = time.perf_counter() - started
                total_bytes += len(chunk)
    except Exception as exc:  # noqa: BLE001
        return Result(ok=False, error=f"{type(exc).__name__}: {exc}")

    if first_byte is None:
        return Result(ok=False, error="no audio received")

    elapsed = time.perf_counter() - started
    rate = payload.get("sample_rate") or 24000
    header = 44 if payload.get("format", "wav") != "pcm" else 0
    return Result(
        ok=True,
        ttfb_ms=first_byte * 1000,
        total_ms=elapsed * 1000,
        audio_s=max(0, total_bytes - header) / 2 / rate,
        bytes_out=total_bytes,
    )


async def run_at_rate(url: str, rps: float, seconds: float, payload_base: dict,
                      texts: list[str]) -> tuple[Summary, float]:
    """Open requests at a fixed arrival rate, regardless of how fast they finish.

    This is the measurement that sizes a deployment. A concurrency sweep answers
    "if N callers all speak at once, how long does the last one wait" — but a
    voice bot's 100 open sockets are mostly idle between turns, so what decides
    whether the service keeps up is arrival rate, not connection count. Here the
    load generator does not wait for a response before sending the next request,
    so a queue builds if and only if the server is genuinely behind.
    """
    import aiohttp

    summary = Summary(concurrency=0)
    timeout = aiohttp.ClientTimeout(total=300)
    interval = 1.0 / rps

    async with aiohttp.ClientSession(timeout=timeout) as session:
        started = time.perf_counter()
        tasks, index = [], 0
        while time.perf_counter() - started < seconds:
            payload = {**payload_base, "text": texts[index % len(texts)]}
            tasks.append(asyncio.create_task(_one(session, url, payload)))
            index += 1
            await asyncio.sleep(max(0.0, interval - (time.perf_counter() - started) % interval))
        summary.results = list(await asyncio.gather(*tasks))
        wall = time.perf_counter() - started

    summary.concurrency = round(rps, 1)
    return summary, wall


async def run(url: str, concurrency: int, requests: int, payload_base: dict,
              texts: list[str]) -> tuple[Summary, float]:
    import aiohttp

    summary = Summary(concurrency=concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def _task(index: int) -> Result:
            async with semaphore:
                payload = {**payload_base, "text": texts[index % len(texts)]}
                return await _one(session, url, payload)

        started = time.perf_counter()
        summary.results = list(await asyncio.gather(*[_task(i) for i in range(requests)]))
        wall = time.perf_counter() - started

    return summary, wall


async def warm(url: str, payload_base: dict, texts: list[str]) -> None:
    """One pass to prime kernels and caches before anything is measured."""
    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        await asyncio.gather(*[
            _one(session, url, {**payload_base, "text": text}) for text in texts[:4]
        ])


def _print_row(report: dict) -> None:
    print(
        f"  {report['concurrency']:>4}  {report['ok']:>4}/{report['requests']:<4}"
        f"  {report['ttfb_p50']:>8.1f}  {report['ttfb_p90']:>8.1f}  {report['ttfb_p99']:>9.1f}"
        f"  {report['total_p50']:>9.1f}  {report['rps']:>6.2f}  {report['realtime_x']:>8.1f}x"
        f"  {report['failed']:>6}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a running FlowTTS server")
    parser.add_argument("--url", default="http://127.0.0.1:9000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=None,
                        help="total requests (default: 6x concurrency)")
    parser.add_argument("--sweep", default=None,
                        help="comma-separated concurrency levels, e.g. 1,4,16,64,100")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--format", default="pcm", choices=["pcm", "wav"])
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--num-step", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--text", action="append", default=None,
                        help="override the built-in text mix (repeatable)")
    parser.add_argument("--rate", default=None,
                        help="comma-separated arrival rates in requests/second "
                             "(e.g. 1,2,4,6,8) — measures TTFB under offered load "
                             "rather than under fixed concurrency")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="seconds to hold each arrival rate")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    payload: dict = {"language": args.language, "format": args.format}
    if args.voice:
        payload["voice_id"] = args.voice
    if args.sample_rate:
        payload["sample_rate"] = args.sample_rate
    generation = {}
    if args.num_step is not None:
        generation["num_step"] = args.num_step
    if args.guidance_scale is not None:
        generation["guidance_scale"] = args.guidance_scale
    if generation:
        payload["generation"] = generation

    texts = args.text or DEFAULT_TEXTS
    levels = ([int(x) for x in args.sweep.split(",")] if args.sweep
              else [args.concurrency])

    asyncio.run(warm(args.url, payload, texts))

    if args.rate:
        rates = [float(x) for x in args.rate.split(",")]
        print(f"\n  {args.url}   offered-load test, {args.duration:.0f}s per rate"
              f"   voice={args.voice or 'default'}  gen={generation or 'server default'}")
        print(f"\n  {'rps in':>6}  {'ok/total':>9}  {'ttfb p50':>8}  {'ttfb p90':>8}"
              f"  {'ttfb p99':>9}  {'total p50':>9}  {'rps out':>7}  {'realtime':>9}  {'fail':>6}")
        print("  " + "-" * 86)
        for rate in rates:
            summary, wall = asyncio.run(run_at_rate(args.url, rate, args.duration,
                                                    payload, texts))
            report = summary.report(wall)
            print(f"  {rate:>6.1f}  {report['ok']:>4}/{report['requests']:<4}"
                  f"  {report['ttfb_p50']:>8.1f}  {report['ttfb_p90']:>8.1f}"
                  f"  {report['ttfb_p99']:>9.1f}  {report['total_p50']:>9.1f}"
                  f"  {report['rps']:>7.2f}  {report['realtime_x']:>8.1f}x"
                  f"  {report['failed']:>6}", flush=True)
        print()
        return

    reports = []
    if not args.json:
        print(f"\n  {args.url}   voice={args.voice or 'default'}  lang={args.language}"
              f"  gen={generation or 'server default'}")
        print(f"\n  {'conc':>4}  {'ok/total':>9}  {'ttfb p50':>8}  {'ttfb p90':>8}"
              f"  {'ttfb p99':>9}  {'total p50':>9}  {'rps':>6}  {'realtime':>9}  {'fail':>6}")
        print("  " + "-" * 84)

    for level in levels:
        requests = args.requests or level * 6
        summary, wall = asyncio.run(run(args.url, level, requests, payload, texts))
        report = summary.report(wall)
        reports.append(report)
        if args.json:
            print(json.dumps(report), flush=True)
        else:
            _print_row(report)
        errors = [r.error for r in summary.results if not r.ok]
        if errors and not args.json:
            print(f"        first error: {errors[0]}", flush=True)

    if not args.json and len(reports) > 1:
        under_200 = [r["concurrency"] for r in reports if r["ttfb_p50"] < 200 and r["failed"] == 0]
        print("\n  highest concurrency with p50 TTFB under 200 ms: "
              f"{max(under_200) if under_200 else 'none'}")
        best = max(reports, key=lambda r: r["rps"])
        print(f"  peak throughput: {best['rps']} rps at concurrency {best['concurrency']} "
              f"({best['realtime_x']}x realtime)")
    print()


if __name__ == "__main__":
    main()
