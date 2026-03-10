#!/usr/bin/env python3
"""
testing_call_pipeline.py — FlowTTS call pipeline test client.

Simulates N concurrent calls, each with its own persistent WebSocket connection.
Within each call, utterances are sent sequentially (one at a time, wait for both
response frames before sending the next). The WebSocket stays open for the entire
call and is closed only when all utterances are done.

This validates the core invariant: one call = one persistent WebSocket.

Usage:
    # 3 concurrent calls, 4 utterances each, against localhost:8765..8767
    python testing_call_pipeline.py

    # Custom
    python testing_call_pipeline.py --calls 5 --utterances 6 --port 9000

    # Save WAV files
    python testing_call_pipeline.py --calls 3 --save-audio ./test_out
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import websockets
from websockets.exceptions import WebSocketException

# ---------------------------------------------------------------------------
# Test sentences (Hindi — same domain as warmup in server.py)
# ---------------------------------------------------------------------------
CALL_TEXTS = [
    "नमस्ते, मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से. क्या मैं customer से बात कर रही हूं?",
    "आपके loan की किस्त अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "हमारे रिकॉर्ड के अनुसार आपका बकाया amount ₹५,००० है, कृपया जल्द से जल्द इसे जमा करें.",
    "आपकी EMI की due date निकल चुकी है, late charge से बचने के लिए आज ही payment करें.",
    "आपकी loan application approve हो गई है और ₹५०,००० सीधे आपके bank account में transfer कर दिए जाएंगे.",
    "क्या आप अपनी personal details verify कर सकते हैं ताकि हम आपके account की जानकारी दे सकें?",
    "आप हमारे mobile app के माध्यम से अपनी EMI pay कर सकते हैं, इसके अलावा NEFT, IMPS, या UPI का भी उपयोग किया जा सकता है.",
    "हमारी company की policy के अनुसार अगर payment ३० दिनों के अंदर नहीं होती तो आपके credit score पर असर पड़ सकता है.",
    "आपका भुगतान सफलतापूर्वक प्राप्त हो गया है और आपका account अब up to date है, धन्यवाद.",
    "कृपया अपना registered mobile number confirm करें ताकि हम आपको OTP भेज सकें.",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class UtteranceResult:
    text_id: str
    text: str
    ok: bool
    llm_s: Optional[float] = None
    decode_s: Optional[float] = None
    total_s: Optional[float] = None
    wav_bytes: Optional[int] = None
    error: Optional[str] = None


@dataclass
class CallResult:
    call_id: str
    port: int
    utterances: list[UtteranceResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(u.ok for u in self.utterances)

    @property
    def ok_count(self) -> int:
        return sum(1 for u in self.utterances if u.ok)


# ---------------------------------------------------------------------------
# Core: one persistent connection per call
# ---------------------------------------------------------------------------
async def run_call(
    call_id: str,
    host: str,
    port: int,
    n_utterances: int,
    save_dir: Optional[Path],
) -> CallResult:
    result = CallResult(call_id=call_id, port=port)
    url = f"ws://{host}:{port}"
    texts = [CALL_TEXTS[i % len(CALL_TEXTS)] for i in range(n_utterances)]

    # Short socket ID shown in every log line — makes it easy to verify
    # the same WebSocket was reused for all utterances in this call.
    ws_id = uuid.uuid4().hex[:8]

    def _ts() -> str:
        return time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"

    print(f"[{_ts()}][{call_id}][ws={ws_id}] connecting → {url}  ({n_utterances} utterances)", flush=True)

    try:
        async with websockets.connect(
            url,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
            open_timeout=15,
        ) as ws:
            print(f"[{_ts()}][{call_id}][ws={ws_id}] connected", flush=True)

            for i, text in enumerate(texts):
                text_id = f"{call_id}-utt{i}"
                print(f"[{_ts()}][{call_id}][ws={ws_id}] → sending utt {i}: {text[:50]!r}", flush=True)
                utt = await _send_utterance(ws, call_id, text_id, text, save_dir, ws_id)
                result.utterances.append(utt)

                if utt.ok:
                    print(
                        f"[{_ts()}][{call_id}][ws={ws_id}] ← utt {i} ok"
                        f"  llm={utt.llm_s:.2f}s  decode={utt.decode_s:.2f}s"
                        f"  total={utt.total_s:.2f}s  wav={utt.wav_bytes}B",
                        flush=True,
                    )
                else:
                    print(f"[{_ts()}][{call_id}][ws={ws_id}] ← utt {i} ERROR: {utt.error}", flush=True)

            print(f"[{_ts()}][{call_id}][ws={ws_id}] all utterances done — closing socket", flush=True)
            # WebSocket closes cleanly here when the `async with` block exits

    except WebSocketException as e:
        print(f"[{_ts()}][{call_id}][ws={ws_id}] WebSocket error: {e}", flush=True)
        result.utterances.append(UtteranceResult(
            text_id=f"{call_id}-connect",
            text="",
            ok=False,
            error=f"Connection failed: {e}",
        ))
    except Exception as e:
        print(f"[{_ts()}][{call_id}][ws={ws_id}] unexpected error: {e}", flush=True)
        result.utterances.append(UtteranceResult(
            text_id=f"{call_id}-connect",
            text="",
            ok=False,
            error=str(e),
        ))

    return result


async def _send_utterance(
    ws,
    call_id: str,
    text_id: str,
    text: str,
    save_dir: Optional[Path],
    ws_id: str = "",
) -> UtteranceResult:
    """Send one synthesize request and wait for both response frames."""
    req = json.dumps({
        "type": "synthesize",
        "call_id": call_id,
        "text_id": text_id,
        "text": text,
    })

    t0 = time.perf_counter()
    try:
        await ws.send(req)

        # Frame 1: JSON metadata
        raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        msg = json.loads(raw)

        if msg.get("type") == "error":
            return UtteranceResult(
                text_id=text_id, text=text, ok=False,
                error=msg.get("error", "unknown error"),
            )

        # Validate call_id and text_id echo-back
        if msg.get("call_id") != call_id:
            return UtteranceResult(
                text_id=text_id, text=text, ok=False,
                error=f"call_id mismatch: sent={call_id!r} got={msg.get('call_id')!r}",
            )
        if msg.get("text_id") != text_id:
            return UtteranceResult(
                text_id=text_id, text=text, ok=False,
                error=f"text_id mismatch: sent={text_id!r} got={msg.get('text_id')!r}",
            )

        # Frame 2: raw WAV bytes
        wav_frame = await asyncio.wait_for(ws.recv(), timeout=30.0)
        if not isinstance(wav_frame, bytes):
            return UtteranceResult(
                text_id=text_id, text=text, ok=False,
                error=f"Expected binary WAV frame, got {type(wav_frame).__name__}",
            )

        total_s = time.perf_counter() - t0

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            wav_path = save_dir / f"{text_id}.wav"
            wav_path.write_bytes(wav_frame)

        return UtteranceResult(
            text_id=text_id,
            text=text,
            ok=True,
            llm_s=msg.get("llm_s"),
            decode_s=msg.get("decode_s"),
            total_s=total_s,
            wav_bytes=len(wav_frame),
        )

    except asyncio.TimeoutError:
        return UtteranceResult(
            text_id=text_id, text=text, ok=False, error="Timeout waiting for response",
        )
    except Exception as e:
        return UtteranceResult(
            text_id=text_id, text=text, ok=False, error=str(e),
        )


# ---------------------------------------------------------------------------
# Runner: N concurrent calls
# ---------------------------------------------------------------------------
async def run_all(
    host: str,
    base_port: int,
    n_calls: int,
    n_utterances: int,
    save_dir: Optional[Path],
    n_ports: int = 1,
) -> tuple[list[CallResult], float]:
    tasks = []
    for i in range(n_calls):
        call_id = f"call-{i:02d}"
        # Round-robin across available ports
        port = base_port + (i % n_ports)
        # Each call gets a random number of utterances between 4 and 5
        n = random.randint(4, 5) if n_utterances == 0 else n_utterances
        tasks.append(run_call(call_id, host, port, n, save_dir))

    ports_desc = f"{base_port}" if n_ports == 1 else f"{base_port}..{base_port+n_ports-1}"
    print(f"\nStarting {n_calls} concurrent call(s) → {host}:{ports_desc}  (round-robin across {n_ports} port(s))\n", flush=True)
    t0 = time.perf_counter()
    results: list[CallResult] = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0

    return results, elapsed


def print_summary(results: list[CallResult], elapsed: float) -> None:
    print("\n" + "=" * 60, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 60, flush=True)

    total_utts = 0
    ok_utts = 0
    ok_calls = 0

    for r in results:
        call_status = "PASS" if r.ok else "FAIL"
        print(f"\n  {call_status}  {r.call_id}  port={r.port}  {r.ok_count}/{len(r.utterances)} utterances ok", flush=True)
        for u in r.utterances:
            if u.ok:
                print(
                    f"    ok   {u.text_id}"
                    f"  llm={u.llm_s:.2f}s  decode={u.decode_s:.2f}s"
                    f"  total={u.total_s:.2f}s  wav={u.wav_bytes}B",
                    flush=True,
                )
            else:
                print(f"    FAIL {u.text_id}  error={u.error}", flush=True)
        total_utts += len(r.utterances)
        ok_utts += r.ok_count
        if r.ok:
            ok_calls += 1

    print(f"\n  Calls:      {ok_calls}/{len(results)} ok", flush=True)
    print(f"  Utterances: {ok_utts}/{total_utts} ok", flush=True)
    print(f"  Wall time:  {elapsed:.2f}s", flush=True)
    print("=" * 60 + "\n", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlowTTS call pipeline test — one persistent WebSocket per call"
    )
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Base WS port (default: 8765)")
    parser.add_argument("--ports", type=int, default=1,
                        help="Number of open server ports to round-robin across (default: 1)")
    parser.add_argument("--calls", type=int, default=10, help="Number of concurrent calls (default: 10)")
    parser.add_argument("--utterances", type=int, default=0,
                        help="Utterances per call (default: 0 = random 4-5 per call)")
    parser.add_argument("--save-audio", metavar="DIR", default=None,
                        help="Save WAV files to this directory")
    args = parser.parse_args()

    save_dir = Path(args.save_audio) if args.save_audio else None

    results, elapsed = asyncio.run(run_all(
        host=args.host,
        base_port=args.port,
        n_calls=args.calls,
        n_utterances=args.utterances,
        save_dir=save_dir,
        n_ports=args.ports,
    ))
    print_summary(results, elapsed)

    # Exit with non-zero if any call failed
    if not all(r.ok for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
