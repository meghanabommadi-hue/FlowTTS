"""
FlowTTS Streaming Stress Test
==============================
Fires N requests spread randomly over S seconds, all in streaming mode.
Measures TTFF and RTF per request. Results saved to a single file.

Usage:
    python stress_test.py --requests 100 --seconds 60 --port 8765 --cache 10
    python stress_test.py --requests 50  --seconds 120 --port 8766 
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import random
import socket
import struct
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import websockets

# ---------------------------------------------------------------------------
# Code-mixed Hindi-English corpus
# ---------------------------------------------------------------------------
TEXTS_GEN: List[str] = [
    # short
    "आपका account number 9876543210 है, कृपया confirm करें।",
    "आपका बकाया Rs. 2500 है, कृपया आज ही जमा करें।",
    "आपकी EMI Rs. 3750 हर महीने देय है।",
    "आपका payment successfully receive हो गया है।",
    "कृपया अपना OTP 4 5 6 7 share करें।",
    "आपका loan approved हो गया है।",
    "आपकी next due date 15 April है।",
    "आपका balance Rs. 1200 है।",
    "आपका case escalate कर दिया गया है।",
    # medium
    "नमस्ते, मैं Bajaj Finance से बात कर रही हूं, क्या मैं customer name से बात कर सकती हूं?",
    "आपके loan की किस्त Rs. 3750 अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "आपके account number 4567890123 पर Rs. 15000 का loan approve हुआ है, क्या आप details verify करेंगे?",
    "हमारे records के अनुसार आपकी last EMI bounce हो गई है, कृपया आज ही payment करें।",
    "आपकी EMI की due date 30 April निकल चुकी है, late charge से बचने के लिए आज ही payment करें।",
    "आप हमारे app के through NEFT, IMPS, या UPI से payment कर सकते हैं।",
    "आपकी KYC verification pending है, कृपया nearest branch में जाएं।",
    "आपका credit score improve करने के लिए time पर payment करना जरूरी है।",
    "इस महीने की 5 तारीख तक payment नहीं हुई तो penalty charges लगेंगे।",
    "आपका loan account number ending in 3456 पर outstanding balance है, कृपया contact करें।",
    # long
    "आपकी loan application approve हो गई है और Rs. 50000 सीधे आपके bank account 7890123456 में transfer कर दिए जाएंगे, जिसमें 2 से 3 कार्य दिवस लग सकते हैं।",
    "हमारी company की policy के अनुसार अगर payment 30 दिनों के अंदर नहीं होती तो आपके credit score पर असर पड़ सकता है, इसलिए कृपया समय पर Rs. 8250 का भुगतान करें।",
    "आप हमारे mobile app के माध्यम से अपनी EMI pay कर सकते हैं, इसके अलावा NEFT, IMPS, या UPI का भी उपयोग किया जा सकता है, और किसी भी समस्या के लिए हमारी customer care team हमेशा available है।",
    "हमारे records के अनुसार आपका loan account number 1234567890 है और आपकी monthly EMI Rs. 4500 है जो हर महीने की 5 तारीख को deduct होती है।",
    "आपके loan की next installment की due date 15 तारीख है और amount Rs. 6200 है, कृपया समय पर भुगतान करें ताकि कोई late fee न लगे।",
    "नमस्ते, मैं Bajaj Finance की तरफ से बात कर रही हूं, आपके loan account number ending in 3456 पर total due amount Rs. 22500 है जिसमें principal Rs. 15000 और late charges Rs. 7500 शामिल हैं, कृपया आज ही payment करें।",
    "आपके loan की EMI जो हर महीने की 5 तारीख को आती है वो इस बार bounce हो गई है, और अगर आप अगले 3 working days में payment नहीं करते तो आपके CIBIL score पर इसका असर पड़ेगा।",
    "आपकी payment successfully receive हो गई है और आपका account अब up to date है, अगर आपको कोई और जानकारी चाहिए तो हमें call करें।",
    # spoken-number / call-centre style
    "ये कॉल आपके loan number जो three six nine पर end होता है उसी के बारे में है, आपका EMI bounce हो गया है और total overdue amount sixteen zero one two rupees है।",
    "आपकी next EMI की due date five April है और amount two thousand three hundred rupees है, कृपया समय पर payment करें नहीं तो आपके credit score पर negative impact पड़ेगा।",
    "आपका account number जो seven eight nine zero पर end होता है उस पर last month की EMI receive नहीं हुई है, कृपया जल्द से जल्द payment करें।",
    "हम आपको inform करना चाहते हैं कि आपका loan number four five six seven के against outstanding balance forty five hundred rupees है और अगर आप आज payment करते हैं तो कोई extra charge नहीं लगेगा।",
    "इस call के दौरान आप हमें बता सकते हैं कि आप payment कब करेंगे, हम आपके लिए एक convenient date और time arrange कर सकते हैं।",
]

TEXTS_CACHE: List[str] = [
    #cache
    "क्या आप इस कॉल पर तुरंत भुगतान कर सकते हैं?",
    "मैं बजाज Finance से वाणी बोल रही हूं recorded line के ज़रिए.",
    "ठीक है, मैं आपको बाद में call करूँगी.",
    "हमने देखा है ये first time bounce हुआ है.",
    "आपकी पहली EMI नियत तारीख पर आपके bank से clear नहीं हुई.",
    "धन्यवाद कॉल लेने के लिए.",
    "Can you tell which mode you will use?",
    "नमस्कार, क्या आपको मेरी आवाज़ आ रही है?",
    "Thanks for confirming.",
    "I'm sorry, I didn't understand that.",
    "I did not understand you clearly.",
    "बहुत बढ़िया, आप किस mode से payment करेंगी?",
    "Thank you for letting me know.",
    "थैंक्यू कॉल उठाने के लिए.",
    "क्या आप अभी, इसी time payment कर सकते हैं?",
    "क्या आप line पर हैं?",
    "धन्यवाद, मैं इंतज़ार कर रही हूँ।",
    "धन्यवाद जी, कन्फर्म करने के लिए।",
    "क्या आप अभी इसी time payment कर सकते हैं?",
    "हम आपको और समय नहीं दे सकते.",
    "नमस्ते, मैं बजाज फाइनेंस से वाणी बोल रही हूँ। एक recorded line के माध्यम से कॉल है।",
    "Alright, I would request you to rethink your decision, as delaying the payment will invite further penalty charges.",
    "मुझे समझ नहीं आया, क्या आप कृपया स्पष्ट कर सकते हैं?",
    "मुझे समझ नहीं आया, क्या आप इसे दोहरा सकते हैं?",
    "थैंक यू for answering.",
    "Great. You should make the payment today to avoid any further reminders.",
]


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------
def _wav_duration_s(wav: bytes) -> float:
    try:
        pos = 12
        sr, ch, bits = 16000, 1, 16
        while pos + 8 <= len(wav):
            cid = wav[pos:pos + 4]
            csz = struct.unpack_from("<I", wav, pos + 4)[0]
            if cid == b"fmt ":
                ch   = struct.unpack_from("<H", wav, pos + 10)[0]
                sr   = struct.unpack_from("<I", wav, pos + 12)[0]
                bits = struct.unpack_from("<H", wav, pos + 22)[0]
            elif cid == b"data":
                return csz / (sr * ch * bits // 8)
            pos += 8 + csz
    except Exception:
        pass
    return max(0.0, len(wav) - 44) / (16000 * 2)


def _concat_wavs(chunks: list[bytes]) -> bytes:
    if not chunks:
        return b""
    if len(chunks) == 1:
        return chunks[0]
    pcm, sr, ch = b"", 16000, 1
    for wav in chunks:
        try:
            pos = 12
            while pos + 8 <= len(wav):
                cid = wav[pos:pos + 4]
                csz = struct.unpack_from("<I", wav, pos + 4)[0]
                if cid == b"fmt ":
                    ch = struct.unpack_from("<H", wav, pos + 10)[0]
                    sr = struct.unpack_from("<I", wav, pos + 12)[0]
                elif cid == b"data":
                    pcm += wav[pos + 8: pos + 8 + csz]
                    break
                pos += 8 + csz
        except Exception:
            pcm += wav[44:]
    dsz = len(pcm)
    br  = sr * ch * 2
    return (b"RIFF" + struct.pack("<I", 36 + dsz) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, br, ch * 2, 16)
            + b"data" + struct.pack("<I", dsz) + pcm)


# ---------------------------------------------------------------------------
# Per-request result
# ---------------------------------------------------------------------------
@dataclass
class Result:
    req_id:    int
    fire_at_s: float
    text:      str
    passed:    bool  = False
    error:     str   = ""
    ttff_s:    float = 0.0   # time from fire → first audio chunk
    total_s:   float = 0.0   # time from fire → audio_done
    audio_s:   float = 0.0   # duration of synthesised audio
    rtf:       float = 0.0   # total_s / audio_s  (lower = faster than real-time)
    llm_s:     float = 0.0
    decode_s:  float = 0.0
    chunks:    int   = 0
    wav_bytes: int   = 0


# ---------------------------------------------------------------------------
# Single streaming request
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT_S = 60.0  # max seconds to wait for a single request to complete

async def _run_one(r: Result, port: int, out_dir: Path) -> None:
    call_id = str(uuid.uuid4())
    url     = f"ws://localhost:{port}/ws/{call_id}"
    t0      = time.perf_counter()

    try:
        async with websockets.connect(url, open_timeout=10, max_size=200 * 1024 * 1024) as ws:
            await ws.send(json.dumps({
                "type":      "synthesize",
                "call_id":   call_id,
                "text_id":   str(uuid.uuid4()),
                "text":      r.text,
                "streaming": True,
            }))

            chunk_wavs: list[bytes] = []
            got_first = False

            while True:
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    continue
                msg   = json.loads(raw)
                mtype = msg.get("type")

                if mtype == "error":
                    r.error = msg.get("error", "server error")
                    return

                if mtype == "audio_chunk":
                    wav_chunk = await ws.recv()
                    if isinstance(wav_chunk, str):
                        wav_chunk = wav_chunk.encode()
                    if not got_first:
                        r.ttff_s  = round(time.perf_counter() - t0, 4)
                        got_first = True
                    chunk_wavs.append(wav_chunk)

                elif mtype == "audio_done":
                    r.total_s  = round(time.perf_counter() - t0, 4)
                    r.llm_s    = msg.get("llm_s") or 0.0
                    r.decode_s = msg.get("decode_s") or 0.0
                    r.chunks   = msg.get("chunks", len(chunk_wavs))

                    if chunk_wavs:
                        combined    = _concat_wavs(chunk_wavs)
                        r.wav_bytes = len(combined)
                        r.audio_s   = round(_wav_duration_s(combined), 4)
                        if r.audio_s > 0:
                            r.rtf = round(r.total_s / r.audio_s, 4)
                        (out_dir / f"req{r.req_id:04d}.wav").write_bytes(combined)

                    r.passed = True
                    return

    except Exception as e:
        r.error = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Scheduler + runner
# ---------------------------------------------------------------------------
async def run(port: int, n_requests: int, duration_s: float,
              out_dir: Path, seed: Optional[int], max_concurrent: int,
              n_cache: int = 0) -> List[Result]:

    dur_s = duration_s
    rng   = random.Random(seed)

    n_gen  = n_requests - n_cache
    texts  = ([TEXTS_GEN[i % len(TEXTS_GEN)] for i in range(n_gen)]
            + [TEXTS_CACHE[i % len(TEXTS_CACHE)] for i in range(n_cache)])
    rng.shuffle(texts)

    fire_ats = sorted(rng.uniform(0, dur_s) for _ in range(n_requests))
    results  = [Result(i, round(t, 4), texts[i]) for i, t in enumerate(fire_ats)]

    # Build and display per-second schedule
    sec_to_ids: dict[int, list[int]] = {}
    for r in results:
        sec_to_ids.setdefault(int(r.fire_at_s), []).append(r.req_id)

    print(f"\n[schedule] {n_requests} requests over {dur_s:.0f}s  (gen={n_requests - n_cache}  cache={n_cache})  port={port}  seed={seed}")
    print("[schedule] requests per second:")
    for s in sorted(sec_to_ids):
        print(f"  {s:>4}s │{'█' * len(sec_to_ids[s])} {len(sec_to_ids[s])}")
    print()

    # Save schedule file
    lines = ["second  count  req_ids"]
    for s in sorted(sec_to_ids):
        lines.append(f"{s:<7} {len(sec_to_ids[s]):<6} {' '.join(map(str, sec_to_ids[s]))}")
    (out_dir / "schedule.txt").write_text("\n".join(lines))

    # Fire requests on schedule
    sem    = asyncio.Semaphore(max_concurrent)
    t_loop = asyncio.get_event_loop().time()
    tasks  = []

    async def _fire(r: Result) -> None:
        async with sem:
            try:
                await asyncio.wait_for(_run_one(r, port, out_dir), timeout=REQUEST_TIMEOUT_S)
            except asyncio.TimeoutError:
                r.error = f"timed out after {REQUEST_TIMEOUT_S:.0f}s"
            ok = "✓" if r.passed else "✗"
            print(
                f"  [{ok}] req{r.req_id:04d}"
                f"  ttff={r.ttff_s:.3f}s"
                f"  total={r.total_s:.3f}s"
                f"  rtf={r.rtf:.3f}"
                f"  audio={r.audio_s:.3f}s"
                + (f"  ← {r.error}" if r.error else ""),
                flush=True,
            )

    for r in results:
        delay = r.fire_at_s - (asyncio.get_event_loop().time() - t_loop)
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(_fire(r)))

    await asyncio.gather(*tasks)
    return results


# ---------------------------------------------------------------------------
# Save results to a single file
# ---------------------------------------------------------------------------
def _save(results: List[Result], out_dir: Path, port: int, duration_s: float) -> None:
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    def _stats(vals: list) -> str:
        if not vals:
            return "n/a"
        s = sorted(vals)
        n = len(s)
        return (f"min={min(s):.3f}  mean={sum(s)/n:.3f}"
                f"  p50={s[n//2]:.3f}  p90={s[int(n*.90)]:.3f}"
                f"  p95={s[int(n*.95)]:.3f}  max={max(s):.3f}")

    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "─" * 74

    agg = "\n".join([
        "AGGREGATE  (passed requests only)",
        sep,
        f"  {'Time-to-first-audio':<22}  {_stats([r.ttff_s   for r in passed if r.ttff_s])}  s",
        f"  {'Total latency':<22}  {_stats([r.total_s  for r in passed if r.total_s])}  s",
        f"  {'Audio duration':<22}  {_stats([r.audio_s  for r in passed if r.audio_s])}  s",
        f"  {'RTF (lower=faster)':<22}  {_stats([r.rtf      for r in passed if r.rtf])}",
        f"  {'LLM':<22}  {_stats([r.llm_s    for r in passed if r.llm_s])}  s",
        f"  {'Decoder':<22}  {_stats([r.decode_s for r in passed if r.decode_s])}  s",
        sep,
    ])

    col_hdr = (f"{'req':>4}  {'fire_at':>7}  {'ok':>3}  "
               f"{'ttff':>7}  {'total':>7}  {'rtf':>6}  {'audio':>7}  "
               f"{'llm':>6}  {'dec':>6}  text")
    rows = [col_hdr, "─" * 110]
    for r in sorted(results, key=lambda x: x.req_id):
        rows.append(
            f"{r.req_id:>4}  {r.fire_at_s:>6.2f}s  {'✓' if r.passed else '✗':>3}  "
            f"{r.ttff_s:>6.3f}s  {r.total_s:>6.3f}s  {r.rtf:>6.3f}  {r.audio_s:>6.3f}s  "
            f"{r.llm_s:>6.3f}  {r.decode_s:>6.3f}  "
            + (r.text if r.passed else f"FAIL: {r.error}")
        )

    def _bucket_table(label: str, metric: str, bucket: list) -> str:
        if not bucket:
            return f"{label}  —  no requests\n"
        col = (f"{'req':>4}  {'fire_at':>7}  {'ttff':>7}  {'rtf':>6}  {'total':>7}  {'audio':>7}  text")
        lines = [label, "─" * 110, col, "─" * 110]
        for r in sorted(bucket, key=lambda x: x.req_id):
            lines.append(
                f"{r.req_id:>4}  {r.fire_at_s:>6.2f}s  "
                f"{r.ttff_s:>6.3f}s  {r.rtf:>6.3f}  {r.total_s:>6.3f}s  {r.audio_s:>6.3f}s  "
                + r.text
            )
        lines.append(f"  count={len(bucket)}")
        return "\n".join(lines)

    def _buckets_for(metric: str) -> tuple:
        lo, mid, hi = [], [], []
        for r in passed:
            v = r.ttff_s if metric == "ttff" else r.rtf
            if v < 0.5:
                lo.append(r)
            elif v < 1.0:
                mid.append(r)
            else:
                hi.append(r)
        return lo, mid, hi

    ttff_lo, ttff_mid, ttff_hi = _buckets_for("ttff")
    rtf_lo,  rtf_mid,  rtf_hi  = _buckets_for("rtf")

    bucket_block = "\n\n".join([
        "TTFF BUCKETS",
        _bucket_table("TTFF  0.0 – 0.5s", "ttff", ttff_lo),
        _bucket_table("TTFF  0.5 – 1.0s", "ttff", ttff_mid),
        _bucket_table("TTFF  > 1.0s",     "ttff", ttff_hi),
        "RTF BUCKETS",
        _bucket_table("RTF   0.0 – 0.5",  "rtf",  rtf_lo),
        _bucket_table("RTF   0.5 – 1.0",  "rtf",  rtf_mid),
        _bucket_table("RTF   > 1.0",      "rtf",  rtf_hi),
    ])

    fail_block = ""
    if failed:
        fail_block = "\nFAILED REQUESTS\n" + sep + "\n"
        for r in failed:
            fail_block += f"  req{r.req_id:04d}  @{r.fire_at_s:.2f}s  {r.error}\n"

    body = "\n".join([
        "=" * 74,
        f"FlowTTS Stress Test  —  {ts}",
        f"port={port}  requests={len(results)}  duration={duration_s:.0f}s  "
        f"passed={len(passed)}  failed={len(failed)}",
        "=" * 74,
        "",
        agg,
        "",
        "PER-REQUEST BREAKDOWN",
        "\n".join(rows),
        "",
        "=" * 74,
        bucket_block,
        fail_block,
        "=" * 74,
        "✓ ALL PASSED" if not failed else f"✗  {len(failed)} FAILED",
        "=" * 74,
    ])

    print("\n" + body)
    (out_dir / "results.txt").write_text(body)
    print(f"\n[output] {out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def _main(args: argparse.Namespace) -> None:
    try:
        with socket.create_connection(("127.0.0.1", args.port), timeout=1):
            pass
    except OSError:
        print(f"ERROR: port {args.port} unreachable. Is the server running?")
        sys.exit(1)

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path.home() / "FlowTTS" / "test" / f"stress_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = await run(
        port          = args.port,
        n_requests    = args.requests,
        duration_s    = args.seconds,
        out_dir       = out_dir,
        seed          = args.seed,
        max_concurrent= args.max_concurrent,
        n_cache       = args.cache,
    )
    _save(results, out_dir, args.port, args.seconds)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FlowTTS streaming stress test — code-mixed Hindi-English")
    p.add_argument("--requests",       type=int,   default=100,  help="Number of requests  (default: 100)")
    p.add_argument("--seconds",        type=float, default=60.0, help="Spread duration in seconds  (default: 60)")
    p.add_argument("--port",           type=int,   default=8765, help="WebSocket port  (default: 8765)")
    p.add_argument("--cache",          type=int,   default=0,    help="Number of requests to serve from cache texts  (default: 0)")
    p.add_argument("--seed",           type=int,   default=None, help="Random seed for reproducible schedule")
    p.add_argument("--max-concurrent", type=int,   default=64,   help="Max simultaneous WS connections  (default: 64)")
    asyncio.run(_main(p.parse_args()))