#!/usr/bin/env python3
"""
Async load balancer for the codec pool server.

Fires one request to EVERY live codec simultaneously using asyncio,
collects results as they arrive (not in submission order), and prints
a live per-codec progress bar via rich.

All codec requests are truly concurrent — asyncio opens N independent
Unix-socket connections at the same time, so all N codec subprocesses
start decoding at t=0 with no queueing.

Strategies (--strategy):
  all-at-once   — send to every codec simultaneously (default)
  least-busy    — query server inflight counts, pick least-busy N codecs
  round-robin   — distribute requests across codecs in order

Usage:
    # fire one request per codec, all at once
    python flowtts/test/codec_load_balancer.py

    # 3 requests per codec, 5 rounds
    python flowtts/test/codec_load_balancer.py --requests-per-codec 3 --rounds 5

    # custom socket, no WAV output
    python flowtts/test/codec_load_balancer.py --socket /tmp/codec5.sock --no-save

    # least-busy strategy, 20 total requests
    python flowtts/test/codec_load_balancer.py --strategy least-busy --total 20
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parents[2]))

# Same hardcoded context tokens used by decoder.py and TTSIntegration/decoder.py
from flowtts.decoder.decoder import CONTEXT_TOKENS

DEFAULT_SOCKET = "/tmp/codec_pool.sock"


# ── Async socket helpers ──────────────────────────────────────────────────────

async def _async_send_recv(payload: dict, sock_path: str) -> dict:
    """Open an async Unix-socket connection, send one JSON line, receive one."""
    reader, writer = await asyncio.open_unix_connection(sock_path)
    try:
        writer.write((json.dumps(payload) + "\n").encode())
        await writer.drain()
        line = await reader.readline()
        return json.loads(line)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def async_query_info(sock_path: str) -> dict:
    return await _async_send_recv({"cmd": "info"}, sock_path)


async def async_decode(
    audio_tokens: str,
    codec_name: Optional[str] = None,
    codec_idx: Optional[int] = None,
    context_tokens: Optional[str] = None,
    sock_path: str = DEFAULT_SOCKET,
) -> dict:
    """Send one async decode request, optionally pinned to a codec."""
    payload: dict = {"audio_tokens": audio_tokens}
    if context_tokens:
        payload["context_tokens"] = context_tokens
    if codec_name is not None:
        payload["codec_name"] = codec_name
    elif codec_idx is not None:
        payload["codec_idx"] = codec_idx
    return await _async_send_recv(payload, sock_path)


# ── Input collection ──────────────────────────────────────────────────────────

def _collect_inputs(bench_root: Path) -> list[dict]:
    """
    Load JSON benchmark records.  Each record must have audio_tokens.
    context_tokens is taken from the JSON if present, otherwise falls back
    to the same hardcoded constant used by TTSIntegration/decoder.py.
    """
    records: list[dict] = []
    dirs = sorted(bench_root.glob("bench_*")) if bench_root.is_dir() else [bench_root]
    if not dirs:
        dirs = [bench_root]
    for d in dirs:
        for jf in sorted(d.glob("*.json")):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            tok = data.get("audio_tokens", "")
            if tok and "<|speech_token_" in tok:
                # Use context_tokens from JSON if saved, else fall back to constant
                ctx = data.get("context_tokens") or CONTEXT_TOKENS
                records.append({
                    "source":         jf.name,
                    "audio_tokens":   tok,
                    "context_tokens": ctx,
                    "text":           data.get("text", ""),
                })
    return records


# ── Per-request async coroutine ───────────────────────────────────────────────

async def _one_request(
    req_id: int,
    codec_name: str,
    codec_idx: int,
    audio_tokens: str,
    context_tokens: str,
    source: str,
    sock_path: str,
    out_dir: Optional[Path],
    progress,            # rich Progress object
    task_id,             # rich task id for this codec's bar
) -> dict:
    t0 = time.perf_counter()
    try:
        resp = await async_decode(
            audio_tokens,
            codec_name=codec_name,
            context_tokens=context_tokens,
            sock_path=sock_path,
        )
        wall_s = time.perf_counter() - t0

        if resp.get("ok"):
            wav_bytes = base64.b64decode(resp["wav_b64"])
            wav_path = ""
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(source).stem
                wav_path = str(out_dir / f"req{req_id:04d}_{codec_name}_{stem}.wav")
                Path(wav_path).write_bytes(wav_bytes)

            decode_ms = resp.get("decode_s", 0) * 1000
            progress.update(
                task_id,
                completed=1,
                description=f"[green]{codec_name:>8}[/]  {decode_ms:.0f}ms  OK",
            )
            return {
                "req_id":     req_id,
                "codec_name": codec_name,
                "codec_idx":  resp.get("codec_idx", codec_idx),
                "ok":         True,
                "decode_s":   resp.get("decode_s"),
                "wall_s":     wall_s,
                "wav_bytes":  len(wav_bytes),
                "source":     source,
                "wav_path":   wav_path,
            }
        else:
            progress.update(
                task_id,
                completed=1,
                description=f"[red]{codec_name:>8}[/]  FAIL",
            )
            return {
                "req_id":     req_id,
                "codec_name": codec_name,
                "codec_idx":  codec_idx,
                "ok":         False,
                "error":      resp.get("error"),
                "wall_s":     wall_s,
                "source":     source,
            }
    except Exception as exc:
        progress.update(
            task_id,
            completed=1,
            description=f"[red]{codec_name:>8}[/]  ERR",
        )
        return {
            "req_id":     req_id,
            "codec_name": codec_name,
            "codec_idx":  codec_idx,
            "ok":         False,
            "error":      str(exc),
            "wall_s":     time.perf_counter() - t0,
            "source":     source,
        }


# ── Load balancing strategies ─────────────────────────────────────────────────

def _assign_all_at_once(
    names: list[tuple[str, int]],
    records: list[dict],
    total: int,
) -> list[tuple[str, int, dict]]:
    """One request per codec — all fire simultaneously."""
    n = min(len(names), total)
    chosen_names = names[:n]
    return [
        (name, idx, random.choice(records))
        for name, idx in chosen_names
    ]


def _assign_round_robin(
    names: list[tuple[str, int]],
    records: list[dict],
    total: int,
) -> list[tuple[str, int, dict]]:
    """Distribute `total` requests across codecs in round-robin order."""
    assignments = []
    for i in range(total):
        name, idx = names[i % len(names)]
        assignments.append((name, idx, random.choice(records)))
    return assignments


def _assign_least_busy(
    names: list[tuple[str, int]],
    records: list[dict],
    total: int,
    inflight: dict[str, int],
) -> list[tuple[str, int, dict]]:
    """
    Sort codecs by current inflight count (least-busy first),
    then distribute `total` requests.
    """
    sorted_names = sorted(names, key=lambda x: inflight.get(x[0], 0))
    assignments = []
    for i in range(total):
        name, idx = sorted_names[i % len(sorted_names)]
        assignments.append((name, idx, random.choice(records)))
    return assignments


# ── Main async runner ─────────────────────────────────────────────────────────

async def run_async(
    strategy: str,
    total: int,
    bench_root: Path,
    sock_path: str,
    out_dir: Optional[Path],
) -> list[dict]:
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TextColumn, TimeElapsedColumn,
    )
    from rich.console import Console

    # 1. Query server
    info = await async_query_info(sock_path)
    n = info["n_codecs"]
    names: list[tuple[str, int]] = sorted(
        info.get("names", {}).items(), key=lambda x: x[1]
    )
    if not names:
        names = [(str(i), i) for i in range(n)]

    print(f"Server: {n} codec(s) — {[nm for nm, _ in names]}")
    print(f"Strategy: {strategy}  |  total requests: {total}\n")

    # 2. Collect inputs
    records = _collect_inputs(bench_root)
    if not records:
        raise RuntimeError(f"No JSON benchmark files found under {bench_root}.")

    # 3. Build assignment list
    if strategy == "all-at-once":
        total = n   # override — one per codec
        assignments = _assign_all_at_once(names, records, total)
    elif strategy == "round-robin":
        assignments = _assign_round_robin(names, records, total)
    elif strategy == "least-busy":
        # Get a rough inflight snapshot (all zero before first round)
        inflight: dict[str, int] = {nm: 0 for nm, _ in names}
        assignments = _assign_least_busy(names, records, total, inflight)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    print(f"Dispatching {len(assignments)} request(s) concurrently …\n")

    # 4. Build rich progress — one bar per codec
    console = Console()
    results: list[dict] = []
    t_wall = time.perf_counter()

    # Group assignments by codec so we can show per-codec bars
    codec_req_counts: dict[str, int] = {}
    for name, _, _ in assignments:
        codec_req_counts[name] = codec_req_counts.get(name, 0) + 1

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=28),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        # One progress task per codec
        task_ids: dict[str, object] = {}
        for name, idx in names:
            cnt = codec_req_counts.get(name, 0)
            task_ids[name] = progress.add_task(
                f"[cyan]{name:>8}[/]  waiting …",
                total=max(cnt, 1),
                completed=0,
            )

        # Launch all coroutines simultaneously
        coros = [
            _one_request(
                req_id=i,
                codec_name=name,
                codec_idx=idx,
                audio_tokens=rec["audio_tokens"],
                context_tokens=rec.get("context_tokens") or CONTEXT_TOKENS,
                source=rec.get("source", "?"),
                sock_path=sock_path,
                out_dir=out_dir,
                progress=progress,
                task_id=task_ids[name],
            )
            for i, (name, idx, rec) in enumerate(assignments)
        ]

        # asyncio.gather fires all coroutines at the same time
        results = list(await asyncio.gather(*coros))

    wall = time.perf_counter() - t_wall
    print(f"\n  Wall time: {wall*1000:.0f}ms  ({len(results)/wall:.2f} req/s)\n")
    return results


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    ok   = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]
    width = 72

    print("═" * width)
    print(f"  LOAD BALANCER RESULTS  {len(ok)}/{len(results)} ok   {len(fail)} failed")
    print("═" * width)

    if fail:
        print("\n  Failures:")
        for r in fail:
            print(f"    req {r['req_id']:>4d}  [{r['codec_name']}]  {r.get('error','?')}")

    if not ok:
        print("═" * width + "\n")
        return

    wall_vals = sorted(r["wall_s"] * 1000 for r in ok)
    dec_vals  = sorted(r["decode_s"] * 1000 for r in ok if r.get("decode_s") is not None)

    def _pct(vals: list[float], p: float) -> float:
        k = (len(vals) - 1) * p / 100
        f = int(k)
        c = min(f + 1, len(vals) - 1)
        return vals[f] + (k - f) * (vals[c] - vals[f])

    print(f"\n  wall  (ms): avg={statistics.mean(wall_vals):.0f}"
          f"  min={min(wall_vals):.0f}  max={max(wall_vals):.0f}"
          f"  p50={_pct(wall_vals,50):.0f}  p95={_pct(wall_vals,95):.0f}")
    if dec_vals:
        print(f"  decode(ms): avg={statistics.mean(dec_vals):.0f}"
              f"  min={min(dec_vals):.0f}  max={max(dec_vals):.0f}"
              f"  p50={_pct(dec_vals,50):.0f}  p95={_pct(dec_vals,95):.0f}")

    # Per-codec breakdown
    by_codec: dict[str, list[dict]] = {}
    for r in ok:
        by_codec.setdefault(r["codec_name"], []).append(r)

    print(f"\n  {'NAME':>8}  {'IDX':>3}  {'REQS':>4}  {'avg wall':>9}  {'avg decode':>10}  {'avg bytes':>9}")
    print("  " + "-" * 56)
    for name in sorted(by_codec, key=lambda nm: by_codec[nm][0]["codec_idx"]):
        reqs  = by_codec[name]
        w_avg = statistics.mean(r["wall_s"] * 1000 for r in reqs)
        d_avg = statistics.mean(r["decode_s"] * 1000 for r in reqs if r.get("decode_s"))
        b_avg = statistics.mean(r.get("wav_bytes", 0) for r in reqs)
        cidx  = reqs[0]["codec_idx"]
        print(f"  {name:>8}  {cidx:>3}  {len(reqs):>4}  {w_avg:>7.0f}ms  {d_avg:>8.0f}ms  {b_avg:>9.0f}")

    # Per-request detail with source file
    print(f"\n  {'REQ':>4}  {'NAME':>8}  {'wall':>8}  {'decode':>8}  SOURCE")
    print("  " + "-" * (width - 2))
    for r in sorted(ok, key=lambda x: x["req_id"]):
        dec_str = f"{r['decode_s']*1000:>6.0f}ms" if r.get("decode_s") else "      -"
        print(f"  {r['req_id']:>4d}  {r['codec_name']:>8s}"
              f"  {r['wall_s']*1000:>6.0f}ms  {dec_str}  {r['source']}")

    print("═" * width + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Async load balancer — fires requests to all live codecs concurrently.\n"
            "Uses asyncio so all N connections open and decode at the same instant."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        choices=["all-at-once", "round-robin", "least-busy"],
        default="all-at-once",
        help=(
            "all-at-once : one request per codec, all fired simultaneously (default)\n"
            "round-robin : distribute --total requests across codecs in order\n"
            "least-busy  : route to least-loaded codecs first"
        ),
    )
    parser.add_argument(
        "--total", type=int, default=None,
        help="Total requests to send (default: N codecs for all-at-once, else required)",
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Repeat the full dispatch round this many times (default: 1)",
    )
    parser.add_argument(
        "--socket", default=DEFAULT_SOCKET,
        help=f"Unix socket path (default: {DEFAULT_SOCKET})",
    )
    parser.add_argument(
        "--bench-dir", type=Path,
        default=Path(__file__).parents[2] / "test",
        help="Root directory containing bench_*/ subdirs (default: test/)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write decoded WAV files to disk",
    )
    args = parser.parse_args()

    if args.strategy != "all-at-once" and args.total is None:
        parser.error("--total is required when strategy is not all-at-once")

    total = args.total or 0   # 0 means "use n_codecs" for all-at-once

    out_dir: Optional[Path] = None
    if not args.no_save:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parents[2] / "test" / f"lb_out_{tag}"
        print(f"WAV outputs → {out_dir}/")

    all_results: list[list[dict]] = []
    for rnd in range(args.rounds):
        if args.rounds > 1:
            print(f"\n── Round {rnd+1}/{args.rounds} {'─'*50}")
        results = asyncio.run(run_async(
            strategy=args.strategy,
            total=total,
            bench_root=args.bench_dir,
            sock_path=args.socket,
            out_dir=out_dir,
        ))
        all_results.append(results)
        print_summary(results)

    if args.rounds > 1:
        # Aggregate across all rounds
        flat = [r for rnd_r in all_results for r in rnd_r]
        print(f"\n{'═'*72}")
        print(f"  AGGREGATE  {args.rounds} rounds  {len([r for r in flat if r.get('ok')])}/{len(flat)} ok")
        print(f"{'═'*72}")
        ok_flat = [r for r in flat if r.get("ok")]
        if ok_flat:
            w = sorted(r["wall_s"]*1000 for r in ok_flat)
            print(f"  wall (ms): avg={statistics.mean(w):.0f}  min={min(w):.0f}  max={max(w):.0f}")
        print()


if __name__ == "__main__":
    main()
