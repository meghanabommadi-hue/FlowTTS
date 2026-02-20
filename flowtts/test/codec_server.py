#!/usr/bin/env python3
"""
Codec pool server — keeps N AudioDecoder instances alive in separate processes.

Each codec runs in its own subprocess with its own Python interpreter and its
own CUDA context, so GPU kernels from different codecs truly overlap in
parallel (no GIL, no shared CUDA stream serialisation).

The main process accepts Unix-socket connections and routes each request to
the target codec's subprocess via a multiprocessing.Queue pair.

Protocol (newline-delimited JSON over Unix socket):
  Request  → {"audio_tokens": "...", "context_tokens": "..." | null,
               "codec_idx": int | null}
           | {"cmd": "info"}
  Response → {"ok": true, "wav_b64": "...", "sample_rate": 48000,
               "codec_idx": 0, "decode_s": 0.12}
           | {"ok": false, "error": "..."}
           | {"ok": true, "n_codecs": N}   (info response)

Routing:
  codec_idx set  → pinned to that codec (strict 1-to-1)
  codec_idx None → least-busy (fewest pending jobs)

Usage:
    python flowtts/test/codec_server.py                 # 3 codecs
    python flowtts/test/codec_server.py --n-codecs 5
    python flowtts/test/codec_server.py --n-codecs 5 --socket /tmp/codec5.sock
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import multiprocessing as mp
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parents[2]))

from flowtts.decoder.decoder import CONTEXT_TOKENS

DEFAULT_SOCKET = "/tmp/codec_pool.sock"

# Codec names — one per instance, cycling if more instances than names
CODEC_NAMES = [
    "alpha", "beta", "gamma", "delta", "epsilon",
    "zeta",  "eta",  "theta", "iota",  "kappa",
    "lambda","mu",   "nu",    "xi",    "omicron",
    "pi",    "rho",  "sigma", "tau",   "upsilon",
    "phi",   "chi",  "psi",   "omega",
]

def _name(idx: int) -> str:
    return CODEC_NAMES[idx % len(CODEC_NAMES)]


# ── Codec worker process ──────────────────────────────────────────────────────

def _codec_process(idx: int, req_q: mp.Queue, res_q: mp.Queue) -> None:
    """
    Runs in a dedicated subprocess. Owns one AudioDecoder (one CUDA context).
    Pulls jobs from req_q, decodes, puts results in res_q.
    """
    # Import inside the subprocess so each process gets its own CUDA init
    from flowtts.decoder.decoder import AudioDecoder

    try:
        codec = AudioDecoder()
        res_q.put({"type": "ready", "idx": idx})
    except Exception as exc:
        res_q.put({"type": "error", "idx": idx, "error": str(exc)})
        return

    while True:
        job = req_q.get()
        if job is None:          # sentinel → shut down
            break

        job_id         = job["job_id"]
        audio_tokens   = job["audio_tokens"]
        context_tokens = job.get("context_tokens") or CONTEXT_TOKENS

        t0 = time.perf_counter()
        try:
            result   = codec.decode_to_wav(audio_tokens, context_tokens)
            decode_s = time.perf_counter() - t0

            wav_io = io.BytesIO(result.wav_bytes)
            wav_b64 = base64.b64encode(result.wav_bytes).decode()

            res_q.put({
                "type":        "result",
                "job_id":      job_id,
                "idx":         idx,
                "ok":          True,
                "wav_b64":     wav_b64,
                "sample_rate": result.sample_rate,
                "decode_s":    decode_s,
            })
        except Exception as exc:
            decode_s = time.perf_counter() - t0
            res_q.put({
                "type":     "result",
                "job_id":   job_id,
                "idx":      idx,
                "ok":       False,
                "error":    str(exc),
                "decode_s": decode_s,
            })


# ── Codec pool (main process) ─────────────────────────────────────────────────

class CodecPool:
    """
    Manages N codec subprocesses.
    Each subprocess has its own req_q (work in) and shared res_q (results out).
    A result-collector thread routes results back to waiting connection threads.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self._req_qs: list[mp.Queue] = []
        self._res_q:  mp.Queue = mp.Queue()
        self._procs:  list[mp.Process] = []

        # pending[job_id] = threading.Event, response dict
        self._pending: dict[int, tuple[threading.Event, dict]] = {}
        self._pending_lock = threading.Lock()
        self._job_counter  = 0
        self._inflight     = [0] * n   # per-codec queue depth
        self._inflight_lock = threading.Lock()

        # Stats
        self.jobs_done   = [0] * n
        self.last_ms     = [0.0] * n
        self.avg_ms      = [0.0] * n
        self.state       = ["init"] * n

        print(f"Starting {n} codec subprocess(es) …", flush=True)
        for i in range(n):
            req_q = mp.Queue()
            self._req_qs.append(req_q)
            p = mp.Process(
                target=_codec_process,
                args=(i, req_q, self._res_q),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

        # Wait for all subprocesses to signal ready
        ready = 0
        while ready < n:
            msg = self._res_q.get(timeout=120)
            i = msg["idx"]
            if msg["type"] == "ready":
                self.state[i] = "idle"
                print(f"  [{_name(i)}] codec {i} ready", flush=True)
                ready += 1
            elif msg["type"] == "error":
                raise RuntimeError(f"Codec {i} ({_name(i)}) failed to init: {msg['error']}")

        print(f"\nPool of {n} live codec process(es) ready.\n", flush=True)
        # Build name→index lookup
        self._name_to_idx: dict[str, int] = {_name(i): i for i in range(n)}

        # Start result collector thread
        self._collector = threading.Thread(target=self._collect_results, daemon=True)
        self._collector.start()

    def _collect_results(self) -> None:
        while True:
            msg = self._res_q.get()
            if msg.get("type") != "result":
                continue
            job_id = msg["job_id"]
            idx    = msg["idx"]

            with self._inflight_lock:
                self._inflight[idx] = max(0, self._inflight[idx] - 1)

            done_ms = msg.get("decode_s", 0) * 1000
            self.jobs_done[idx] += 1
            self.last_ms[idx]    = done_ms
            n = self.jobs_done[idx]
            self.avg_ms[idx]     = self.avg_ms[idx] * (n-1)/n + done_ms/n
            self.state[idx]      = "idle"

            with self._pending_lock:
                entry = self._pending.pop(job_id, None)
            if entry:
                event, resp = entry
                resp.update(msg)
                event.set()

    def _next_job_id(self) -> int:
        with self._pending_lock:
            jid = self._job_counter
            self._job_counter += 1
        return jid

    def resolve(self, pin_idx: Optional[int], pin_name: Optional[str]) -> int:
        """Resolve a codec index from an index or a name."""
        if pin_name is not None:
            if pin_name not in self._name_to_idx:
                raise ValueError(f"Unknown codec name {pin_name!r}. "
                                 f"Known: {list(self._name_to_idx)}")
            return self._name_to_idx[pin_name]
        if pin_idx is not None:
            return pin_idx % self.n
        return None  # least-busy

    def dispatch(
        self,
        audio_tokens: str,
        context_tokens: Optional[str],
        pin_idx: Optional[int],
        pin_name: Optional[str] = None,
    ) -> dict:
        """
        Submit a decode job and block until the result arrives.
        pin_name → route to named codec (e.g. "alpha")
        pin_idx  → route to indexed codec
        both None → least-busy load balancing
        Returns the result dict from the subprocess.
        """
        resolved = self.resolve(pin_idx, pin_name)
        with self._inflight_lock:
            if resolved is not None:
                chosen = resolved
            else:
                chosen = self._inflight.index(min(self._inflight))
            self._inflight[chosen] += 1
            self.state[chosen] = "busy"

        job_id = self._next_job_id()
        event  = threading.Event()
        resp: dict = {}

        with self._pending_lock:
            self._pending[job_id] = (event, resp)

        self._req_qs[chosen].put({
            "job_id":        job_id,
            "audio_tokens":  audio_tokens,
            "context_tokens": context_tokens,
        })

        event.wait()
        return resp

    def shutdown(self) -> None:
        for q in self._req_qs:
            q.put(None)


# ── Live progress display ─────────────────────────────────────────────────────

def _display_loop(pool: CodecPool, stop_event: threading.Event, start_time: float) -> None:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text

    console = Console()

    def build_table() -> Table:
        total = sum(pool.jobs_done)
        uptime = time.perf_counter() - start_time
        table = Table(
            title=f"Codec Pool  |  uptime {uptime:.0f}s  |  total done: {total}",
            show_lines=False,
        )
        table.add_column("Name",    justify="left",   style="cyan")
        table.add_column("Idx",     justify="right")
        table.add_column("State",   justify="center")
        table.add_column("Queued",  justify="right")
        table.add_column("Done",    justify="right")
        table.add_column("Last ms", justify="right")
        table.add_column("Avg ms",  justify="right")
        table.add_column("Bar",     min_width=24)

        max_done = max(pool.jobs_done) or 1
        for i in range(pool.n):
            st   = pool.state[i]
            done = pool.jobs_done[i]
            with pool._inflight_lock:
                q = pool._inflight[i]

            state_text = Text(st, style="green bold" if st == "busy" else "dim white")
            bar_fill   = int(24 * done / max_done)
            bar        = Text(
                "█" * bar_fill + "░" * (24 - bar_fill),
                style="green" if st == "busy" else "blue",
            )
            table.add_row(
                _name(i),
                str(i),
                state_text,
                str(q),
                str(done),
                f"{pool.last_ms[i]:.0f}" if done else "-",
                f"{pool.avg_ms[i]:.0f}"  if done else "-",
                bar,
            )
        return table

    with Live(console=console, refresh_per_second=8, screen=False) as live:
        while not stop_event.is_set():
            live.update(build_table())
            time.sleep(0.125)


# ── Per-connection handler ────────────────────────────────────────────────────

def _handle(conn: socket.socket, pool: CodecPool) -> None:
    buf = b""
    try:
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError as e:
                    _send(conn, {"ok": False, "error": f"bad JSON: {e}"})
                    continue

                if req.get("cmd") == "info":
                    _send(conn, {
                        "ok":       True,
                        "n_codecs": pool.n,
                        "names":    {_name(i): i for i in range(pool.n)},
                    })
                    continue

                audio_tokens   = req.get("audio_tokens", "")
                context_tokens = req.get("context_tokens") or None
                pin_idx        = req.get("codec_idx")    # int or null
                pin_name       = req.get("codec_name")   # str or null  e.g. "alpha"

                if not audio_tokens:
                    _send(conn, {"ok": False, "error": "missing audio_tokens"})
                    continue

                try:
                    result = pool.dispatch(audio_tokens, context_tokens, pin_idx, pin_name)
                except ValueError as e:
                    _send(conn, {"ok": False, "error": str(e)})
                    continue

                chosen_idx = result.get("idx", -1)
                _send(conn, {
                    "ok":          result.get("ok", False),
                    "wav_b64":     result.get("wav_b64", ""),
                    "sample_rate": result.get("sample_rate", 0),
                    "codec_idx":   chosen_idx,
                    "codec_name":  _name(chosen_idx) if chosen_idx >= 0 else "?",
                    "decode_s":    result.get("decode_s", 0),
                    "error":       result.get("error"),
                })
    finally:
        conn.close()


def _send(conn: socket.socket, obj: dict) -> None:
    conn.sendall((json.dumps(obj) + "\n").encode())


# ── Server loop ───────────────────────────────────────────────────────────────

def serve(n_codecs: int, sock_path: str) -> None:
    pool = CodecPool(n_codecs)

    if os.path.exists(sock_path):
        os.unlink(sock_path)

    start_time = time.perf_counter()
    stop_event = threading.Event()
    display_thr = threading.Thread(
        target=_display_loop,
        args=(pool, stop_event, start_time),
        daemon=True,
    )
    display_thr.start()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(sock_path)
        server.listen(256)
        print(f"Listening on {sock_path}  (Ctrl-C to stop)\n", flush=True)

        try:
            while True:
                conn, _ = server.accept()
                t = threading.Thread(target=_handle, args=(conn, pool), daemon=True)
                t.start()
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            pool.shutdown()
            if os.path.exists(sock_path):
                os.unlink(sock_path)
            print("\nShutting down.", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description=(
            "Codec pool server: N codecs in separate processes (separate CUDA contexts).\n"
            "Routing: pin with codec_idx or least-busy load balancing."
        ),
    )
    parser.add_argument("--n-codecs", type=int, default=3,
                        help="Number of codec subprocesses (default: 3)")
    parser.add_argument("--socket", default=DEFAULT_SOCKET,
                        help=f"Unix socket path (default: {DEFAULT_SOCKET})")
    args = parser.parse_args()
    serve(args.n_codecs, args.socket)


if __name__ == "__main__":
    main()
