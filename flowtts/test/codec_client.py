#!/usr/bin/env python3
"""
Codec pool client — sends decode requests to a running codec_server.py.

Default: fires --requests (default 10) requests in parallel, distributed
round-robin across all live codecs.

Import send_decode_request() into any test file for single requests, or
use send_requests() to fire a configurable number of parallel requests.

Usage:
    # 10 requests across all codecs (default)
    python flowtts/test/codec_client.py

    # custom number of requests
    python flowtts/test/codec_client.py --requests 20

    # custom socket
    python flowtts/test/codec_client.py --socket /tmp/codec5.sock

    # send a single audio_tokens string to a specific codec
    python flowtts/test/codec_client.py --tokens "<|speech_token_1|>..." --codec-idx 2

    # no WAV output
    python flowtts/test/codec_client.py --no-save
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import socket
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parents[2]))

DEFAULT_SOCKET = "/tmp/codec_pool.sock"


# ── Core send/receive ─────────────────────────────────────────────────────────

def _send_recv(payload: dict, sock_path: str) -> dict:
    """Open a connection, send one JSON line, receive one JSON line."""
    data = (json.dumps(payload) + "\n").encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(sock_path)
        s.sendall(data)
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.split(b"\n")[0])


def query_pool_info(sock_path: str = DEFAULT_SOCKET) -> dict:
    """Ask the server for pool size and name→index mapping."""
    resp = _send_recv({"cmd": "info"}, sock_path)
    if not resp.get("ok"):
        raise RuntimeError(f"Server info failed: {resp.get('error')}")
    return resp   # {"n_codecs": N, "names": {"alpha": 0, ...}}


def query_pool_size(sock_path: str = DEFAULT_SOCKET) -> int:
    return query_pool_info(sock_path)["n_codecs"]


def send_decode_request(
    audio_tokens: str,
    context_tokens: Optional[str] = None,
    codec_idx: Optional[int] = None,
    codec_name: Optional[str] = None,
    sock_path: str = DEFAULT_SOCKET,
) -> dict:
    """
    Send one decode request.
    codec_name → pin by name (e.g. "alpha")
    codec_idx  → pin by index
    both None  → server-side least-busy balancing
    """
    payload: dict = {"audio_tokens": audio_tokens, "context_tokens": context_tokens}
    if codec_name is not None:
        payload["codec_name"] = codec_name
    elif codec_idx is not None:
        payload["codec_idx"] = codec_idx
    return _send_recv(payload, sock_path)


# ── Input collection ──────────────────────────────────────────────────────────

def _collect_json_inputs(bench_root: Path) -> list[dict]:
    records: list[dict] = []
    search_dirs = sorted(bench_root.glob("bench_*")) if bench_root.is_dir() else [bench_root]
    if not search_dirs:
        search_dirs = [bench_root]
    for d in search_dirs:
        for jf in sorted(d.glob("*.json")):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            tokens = data.get("audio_tokens", "")
            if tokens and "<|speech_token_" in tokens:
                records.append({"source": jf.name, **data})
    return records


# ── Per-request worker ────────────────────────────────────────────────────────

def _one_request(
    req_id: int,
    codec_idx: int,          # pinned by index — unambiguous even when names cycle
    audio_tokens: str,
    source: str,
    sock_path: str,
    out_dir: Optional[Path],
) -> dict:
    t0 = time.perf_counter()
    try:
        # Pin by index — works correctly even when n_codecs > len(CODEC_NAMES)
        resp = send_decode_request(audio_tokens, codec_idx=codec_idx, sock_path=sock_path)
        wall_s = time.perf_counter() - t0

        if resp.get("ok"):
            wav_bytes = base64.b64decode(resp["wav_b64"])
            wav_path = ""
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(source).stem
                wav_path = str(out_dir / f"req{req_id:04d}_codec{codec_idx}_{stem}.wav")
                Path(wav_path).write_bytes(wav_bytes)
            return {
                "req_id":     req_id,
                "codec_idx":  resp.get("codec_idx", codec_idx),
                "codec_name": resp.get("codec_name", str(codec_idx)),
                "ok":         True,
                "decode_s":   resp.get("decode_s"),
                "wall_s":     wall_s,
                "wav_bytes":  len(wav_bytes),
                "source":     source,
                "wav_path":   wav_path,
            }
        else:
            return {
                "req_id":     req_id,
                "codec_idx":  codec_idx,
                "codec_name": resp.get("codec_name", str(codec_idx)),
                "ok":         False,
                "error":      resp.get("error"),
                "wall_s":     wall_s,
                "source":     source,
            }
    except Exception as exc:
        return {
            "req_id":     req_id,
            "codec_idx":  codec_idx,
            "codec_name": str(codec_idx),
            "ok":         False,
            "error":      str(exc),
            "wall_s":     time.perf_counter() - t0,
            "source":     source,
        }


# ── Multi-request parallel sender ────────────────────────────────────────────

def send_requests(
    n_requests: int,
    bench_root: Path,
    sock_path: str,
    out_dir: Optional[Path],
) -> list[dict]:
    """
    Fire n_requests in parallel, distributed round-robin across all live codecs.
    Each request is pinned to a codec by name.
    """
    info = query_pool_info(sock_path)
    n_codecs = info["n_codecs"]

    print(f"Server has {n_codecs} codec(s)")
    print(f"Sending {n_requests} parallel requests — round-robin across codecs.\n")

    all_records = _collect_json_inputs(bench_root)
    if not all_records:
        raise RuntimeError(f"No JSON benchmark files found under {bench_root}.")

    # Build jobs: assign codec round-robin by INDEX (avoids name-collision when
    # n_codecs > number of unique names). Server routes by codec_idx unambiguously.
    jobs = []
    for i in range(n_requests):
        idx = i % n_codecs
        rec = random.choice(all_records)
        jobs.append((i, idx, rec["audio_tokens"], rec.get("source", "?")))

    results: list[dict] = [{}] * n_requests
    completed = 0

    t_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_requests) as executor:
        future_to_id = {
            executor.submit(
                _one_request,
                req_id, codec_idx, audio_tokens, source,
                sock_path, out_dir,
            ): req_id
            for req_id, codec_idx, audio_tokens, source in jobs
        }
        for future in as_completed(future_to_id):
            req_id = future_to_id[future]
            try:
                res = future.result()
            except Exception as exc:
                _, codec_idx, _, source = jobs[req_id]
                res = {
                    "req_id":     req_id,
                    "codec_idx":  codec_idx,
                    "codec_name": str(codec_idx),
                    "ok":         False,
                    "error":      str(exc),
                    "wall_s":     0.0,
                    "source":     source,
                }
            results[req_id] = res
            completed += 1
            status = "OK" if res["ok"] else f"FAIL: {res.get('error')}"
            print(
                f"  req {res['req_id']:>4d} → [{res['codec_name']:>8s}]  {status}"
                f"  wall={res['wall_s']*1000:.0f}ms"
                + (f"  decode={res['decode_s']*1000:.0f}ms" if res.get("decode_s") else "")
                + f"  ({completed}/{n_requests})",
                flush=True,
            )

    wall = time.perf_counter() - t_wall
    print(f"\n  Wall time: {wall*1000:.0f}ms  ({n_requests/wall:.2f} req/s)\n")
    return results


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    ok   = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]
    width = 68

    print("═" * width)
    print(f"  CLIENT RESULTS  {len(ok)}/{len(results)} ok   {len(fail)} failed")
    print("═" * width)

    if fail:
        print("\n  Failures:")
        for r in fail:
            print(f"    req {r['req_id']:>4d} → codec {r['codec_idx']}  {r.get('error', '?')}")

    if ok:
        wall_vals = sorted(r["wall_s"] * 1000 for r in ok)
        dec_vals  = sorted(r["decode_s"] * 1000 for r in ok if r.get("decode_s") is not None)
        print(f"\n  wall  latency (ms):  avg={statistics.mean(wall_vals):.0f}"
              f"  min={min(wall_vals):.0f}  max={max(wall_vals):.0f}")
        if dec_vals:
            print(f"  decode latency (ms): avg={statistics.mean(dec_vals):.0f}"
                  f"  min={min(dec_vals):.0f}  max={max(dec_vals):.0f}")

        print(f"\n  {'REQ':>4}  {'NAME':>8}  {'IDX':>3}  {'wall':>8}  {'decode':>8}  SOURCE")
        print("  " + "-" * (width - 2))
        for r in sorted(ok, key=lambda x: x["req_id"]):
            dec_str = f"{r['decode_s']*1000:>6.0f}ms" if r.get("decode_s") else "      -"
            print(
                f"  {r['req_id']:>4d}  {r.get('codec_name','?'):>8s}  {r['codec_idx']:>3d}"
                f"  {r['wall_s']*1000:>6.0f}ms  {dec_str}  {r['source']}"
            )

    print("═" * width + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Send one request per live codec instance, all in parallel.\n"
            "Request i is pinned to codec i — strictly 1-to-1."
        ),
    )
    parser.add_argument(
        "--socket", default=DEFAULT_SOCKET,
        help=f"Unix socket path of the running server (default: {DEFAULT_SOCKET})",
    )
    parser.add_argument(
        "--tokens", default=None,
        help="Send a single audio_tokens string (use with --codec-name or --codec-idx)",
    )
    parser.add_argument(
        "--codec-name", default=None,
        help='Pin single request to a named codec, e.g. "alpha" (default: load-balanced)',
    )
    parser.add_argument(
        "--codec-idx", type=int, default=None,
        help="Pin single request to this codec index (default: load-balanced)",
    )
    parser.add_argument(
        "--bench-dir", type=Path,
        default=Path(__file__).parents[2] / "test",
        help="Root directory containing bench_*/ subdirs (default: test/)",
    )
    parser.add_argument(
        "--requests", type=int, default=10,
        help="Number of parallel requests to send (default: 10)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write decoded WAV files to disk",
    )
    args = parser.parse_args()

    out_dir: Optional[Path] = None
    if not args.no_save:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parents[2] / "test" / f"client_out_{tag}"
        print(f"WAV outputs → {out_dir}/")

    # Single-token mode
    if args.tokens:
        resp = send_decode_request(
            args.tokens,
            codec_idx=args.codec_idx,
            codec_name=args.codec_name,
            sock_path=args.socket,
        )
        if resp.get("ok"):
            wav_bytes = base64.b64decode(resp["wav_b64"])
            print(f"OK  codec=[{resp.get('codec_name','?')}]/{resp['codec_idx']}"
                  f"  decode={resp['decode_s']*1000:.0f}ms  wav={len(wav_bytes)} bytes")
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "single.wav"
                path.write_bytes(wav_bytes)
                print(f"Saved → {path}")
        else:
            print(f"FAIL: {resp.get('error')}")
        return

    results = send_requests(
        n_requests=args.requests,
        bench_root=args.bench_dir,
        sock_path=args.socket,
        out_dir=out_dir,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
