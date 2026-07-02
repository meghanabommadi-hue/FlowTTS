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
  WAV files saved to ~/FlowTTS/test/pipeline_test_YYYYMMDD_HHMMSS/
  Summary table printed and written as summary.txt.

Usage:
    # Managed — launch server, allocate 9 ports on demand, run 40 requests
    python -m flowtts.test.test_pipeline --requests 40 --concurrency 9

    # External — server already running on 8080-8773
    python -m flowtts.test.test_pipeline --no-launch --n-ports 9 --requests 40

    # command to kill all ports
    kill $(ss -tlnp | grep :8764 | grep -oP 'pid=\K[0-9]+')

    # command to check all open ports
    ss -tlnp | grep python3 2>/dev/null | awk '{print $4}' | sort -t: -k2 -n
"""

from __future__ import annotations

import argparse
import asyncio
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

from flowtts.core.config import settings as _settings

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_TEST_ROOT = Path.home() / "FlowTTS/test"
_LLM_LOG   = Path.home() / "FlowTTS/llm.log"


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
    """Load text sentences from bench_* JSON files."""
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
_BENCH_TEXTS: List[str] = []   # loaded by _build_cache_mix or lazily on first use
_VOICE_ID: str = ""            # set via --voice arg
_FIXED_SENTENCE: str = ""      # set via --sentence arg; repeats same text every request

_BAJAJ_SENTENCES_FILE = Path.home() / "FlowTTS/sample_files/bajaj_sentences_unique.txt"


def _build_cache_mix(n: int, cache_mix: str, voice: str) -> List[str]:
    """Return n sentences with requested cache ratio from cached_texts.txt / bajaj_sentences_unique.txt."""
    import hashlib as _hashlib
    import random as _random

    voice     = voice or "simran"
    cache_dir = Path.home() / f"FlowTTS/cached_data_{voice}"
    cached_txt = cache_dir / "cached_texts.txt"

    if cached_txt.exists():
        cached = [l.strip() for l in cached_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"[cache_mix] {len(cached)} cached sentences from {cached_txt}", flush=True)
    elif cache_dir.exists():
        cached_hashes = {f.stem for f in cache_dir.glob("*.wav")}
        try:
            all_sents = [l.strip() for l in _BAJAJ_SENTENCES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
            cached = [s for s in all_sents if _hashlib.sha256(s.encode()).hexdigest() in cached_hashes]
        except Exception:
            cached = []
    else:
        cached = []

    try:
        all_sents = [l.strip() for l in _BAJAJ_SENTENCES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception:
        all_sents = cached[:]

    cached_set = set(cached)
    uncached   = [s for s in all_sents if s not in cached_set]

    mix = cache_mix.lower().strip()
    pct = (100 if mix in ("full", "100") else
           0   if mix in ("none", "0")   else
           50  if mix in ("half", "partial", "50") else
           max(0, min(100, int(mix))) if mix.isdigit() else 50)

    n_cached   = round(n * pct / 100)
    n_uncached = n - n_cached

    def _sample(pool: list, k: int) -> list:
        if not pool or k == 0:
            return []
        return [_random.choice(pool) for _ in range(k)]

    chosen = _sample(cached, n_cached) + _sample(uncached, n_uncached)
    _random.shuffle(chosen)
    print(f"[cache_mix] {pct}% cached → {n_cached} cached + {n_uncached} uncached"
          f"  (pool: {len(cached)} cached, {len(uncached)} uncached)", flush=True)
    return chosen

# English sentences for testing American accent.
_ENGLISH_AMERICAN: List[str] = [
    "Hey there! I just wanted to check in and see how everything's going on your end.",
    "We're all set for the meeting at three o'clock — I'll send over the agenda in just a bit.",
    "I can't believe how fast the semester went by; finals are already right around the corner.",
    "Could you go ahead and pull up that report from last quarter so we can walk through the numbers?",
    "Honestly, the weather out here in California has been absolutely perfect this time of year.",
]

# Per-language fallback sentences (short / medium / long mix).
_HINDI_FALLBACK: List[str] = [
    # short — Hindi numerals'
    # "थैंक यू hold करने के लिए, अब पूरे steps फिर से clear बता دیتी हूँ:",
    "नमस्ते. मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से. क्या मैं customer name से बात कर रही हूं?",
    # "😭☺️ I am vaani ✅👀",
    # "😭 steps फिर से clear बता",
    # " دیتी हूँ: steps फिर से clear बता دیتी हूँ:",
    "Hello",
    "Hi",
    "नमस्ते मैं आपकी कैसे मदद कर सकती हूं?",
    "नमस्ते, मैं आपकी कैसे मदद कर सकती हूं?",
    "नमस्ते, आपकी कैसे मदद कर सकती हूं?",
    "Hello, मैं कर सकती हूं?",
    "Hi, मैं कर सकती हूं?",
    "नमस्ते. मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से. क्या मैं customer name से बात कर रही हूं?",
    "क्या आप अपना नाम बता सकते हैं?",
    "आपका खाता नंबर नौ आठ सात छह पांच चार तीन दो एक शून्य है, कृपया confirm करें।",
    "कृपया थोड़ा इंतज़ार करें।",
    "आपकी समस्या हल हो गई है।",
    "आपका बकाया दो हज़ार पांच सौ रुपये है, कृपया आज ही जमा करें।",
    "हम जल्द ही आपसे संपर्क करेंगे।",
    "आपका भुगतान दस हज़ार रुपये सफलतापूर्वक हो गया है।",
    # short — English numerals in Hindi sentences
    "आपका account number 9876543210 है, कृपया confirm करें।",
    "आपका बकाया Rs. 2500 है, कृपया आज ही जमा करें।",
    "आपकी EMI Rs. 3750 हर महीने देय है।",
    # medium — Hindi numerals
    "नमस्ते. मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से. क्या मैं customer name से बात कर रही हूं?",
    "आपके loan की किस्त तीन हज़ार सात सौ पचास रुपये अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "हमारे रिकॉर्ड के अनुसार आपका बकाया amount पांच हज़ार रुपये है, कृपया जल्द से जल्द इसे जमा करें।",
    "आपकी EMI की due date तीस अप्रैल निकल चुकी है, late charge से बचने के लिए आज ही payment करें।",
    "आपके account नंबर चार पांच छह सात आठ नौ शून्य एक दो तीन पर पंद्रह हज़ार रुपये का loan approve हुआ है, क्या आप details verify करेंगे?",
    # medium — English numerals in Hindi sentences
    "आपके loan की किस्त Rs. 3750 अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "आपके account number 4567890123 पर Rs. 15000 का loan approve हुआ है, क्या आप details verify करेंगे?",
    # long — Hindi numerals
    "आपकी loan application approve हो गई है और पचास हज़ार रुपये सीधे आपके bank account सात आठ नौ शून्य एक दो तीन चार पांच छह में transfer कर दिए जाएंगे, जिसमें दो से तीन कार्य दिवस लग सकते हैं।",
    "हमारी company की policy के अनुसार अगर payment तीस दिनों के अंदर नहीं होती तो आपके credit score पर असर पड़ सकता है, इसलिए कृपया समय पर आठ हज़ार दो सौ पचास रुपये का भुगतान करें।",
    "आप हमारे mobile app के माध्यम से अपनी चार हज़ार पांच सौ रुपये की EMI pay कर सकते हैं, इसके अलावा NEFT, IMPS, या UPI का भी उपयोग किया जा सकता है।",
    # long — English numerals in Hindi sentences
    "आपकी loan application approve हो गई है और Rs. 50000 सीधे आपके bank account 7890123456 में transfer कर दिए जाएंगे, जिसमें 2 से 3 कार्य दिवस लग सकते हैं।",
    "हमारी company की policy के अनुसार अगर payment 30 दिनों के अंदर नहीं होती तो आपके credit score पर असर पड़ सकता है, इसलिए कृपया समय पर Rs. 8250 का भुगतान करें।",
    # long — extra variety
    "आपकी loan application approve हो गई है और loan amount सीधे आपके bank account में transfer कर दी जाएगी, जिसमें दो से तीन कार्य दिवस लग सकते हैं।",
    "आप हमारे mobile app के माध्यम से अपनी EMI pay कर सकते हैं, इसके अलावा NEFT, IMPS, या UPI का भी उपयोग किया जा सकता है, और किसी भी समस्या के लिए हमारी customer care team हमेशा available है।",
    "हमारे records के अनुसार आपका loan account number 1234567890 है और आपकी monthly EMI Rs. 4500 है जो हर महीने की 5 तारीख को deduct होती है।",
    "आपकी payment successfully receive हो गई है और आपका account अब up to date है, अगर आपको कोई और जानकारी चाहिए तो हमें call करें।",
    "आपके loan की next installment की due date 15 तारीख है और amount Rs. 6200 है, कृपया समय पर भुगतान करें ताकि कोई late fee न लगे।",
    "हम आपको सूचित करना चाहते हैं कि आपकी KYC verification pending है, कृपया अपने नजदीकी branch में जाकर या हमारे app के माध्यम से इसे complete करें।",
    # spoken-number style (numbers as Hindi words / romanized English)
    "ये कॉल आपके loan number जो three six nine पर end होता है, उसी के बारे में है। आपका EMI bounce हो गया है और total overdue amount sixteen zero one two rupees है।",
    "आपकी next EMI की due date five April है और amount two thousand three hundred rupees है, कृपया समय पर payment करें नहीं तो आपके credit score पर negative impact पड़ेगा।",
    "आपका account number जो seven eight nine zero पर end होता है, उस पर last month की EMI receive नहीं हुई है, कृपया जल्द से जल्द payment करें और किसी भी assistance के लिए हमें call back करें।",
    "हम आपको inform करना चाहते हैं कि आपका loan number four five six seven के against outstanding balance forty five hundred rupees है और अगर आप आज payment करते हैं तो आपको कोई extra charge नहीं लगेगा।",
    "नमस्ते, मैं Bajaj Finance की तरफ से बात कर रही हूं। आपके loan account number ending in three four five six पर total due amount is twenty two thousand five hundred rupees जिसमें principal amount fifteen thousand और late payment charges seven thousand five hundred rupees शामिल हैं, कृपया आज ही payment करें।",
    "आपके loan की EMI जो हर महीने की five तारीख को आती है वो इस बार bounce हो गई है, और अगर आप अगले three working days में payment नहीं करते तो आपके CIBIL score पर इसका असर पड़ेगा जिससे future में loan लेने में problem हो सकती है।",
]

# Mixed very-long + very-short sentences — specifically for testing trimming under batch load.
# Short sentences get batched with long ones → ONNX padding → trim logic exercised.
_HINDI_MIXED_STRESS: List[str] = [
    # very short
    "नमस्ते।",
    "ठीक है।",
    "धन्यवाद।",
    "क्या आप सुन सकते हैं?",
    "कृपया रुकिए।",
    "हाँ, बिल्कुल।",
    # very long
    "आपकी loan application approve हो गई है और Rs. 50000 सीधे आपके bank account 7890123456 में transfer कर दिए जाएंगे, जिसमें 2 से 3 कार्य दिवस लग सकते हैं।",
    "आप हमारे mobile app के माध्यम से अपनी EMI pay कर सकते हैं, इसके अलावा NEFT, IMPS, या UPI का भी उपयोग किया जा सकता है, और किसी भी समस्या के लिए हमारी customer care team हमेशा available है।",
    "हमारी company की policy के अनुसार अगर payment 30 दिनों के अंदर नहीं होती तो आपके credit score पर असर पड़ सकता है, इसलिए कृपया समय पर Rs. 8250 का भुगतान करें।",
    "हम आपको सूचित करना चाहते हैं कि आपकी KYC verification pending है, कृपया अपने नजदीकी branch में जाकर या हमारे app के माध्यम से इसे complete करें।",
    # very short again (interleaved)
    "समझ गया।",
    "एक मिनट।",
    # very long
    "हमारे records के अनुसार आपका loan account number 1234567890 है और आपकी monthly EMI Rs. 4500 है जो हर महीने की 5 तारीख को deduct होती है।",
    "आपके loan की next installment की due date 15 तारीख है और amount Rs. 6200 है, कृपया समय पर भुगतान करें ताकि कोई late fee न लगे।",
]

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
    # medium
    "డాక్టర్ గారు చెప్పిన మందులు వేసుకుంటున్నావా?",
    "ఈ వంటకం తయారు చేయడం చాలా సులభం.",
    "వర్షం పడుతున్న సాయంత్రంలో చిన్న గ్రామం మొత్తం మట్టి వాసనతో నిండిపోయి అందరినీ ఆనందంగా ముంచెత్తింది.",
]

# OmniVoice is multilingual; default to the Hindi/English test sentences.
_FALLBACK_TEXTS: List[str] = _HINDI_FALLBACK + _HINDI_MIXED_STRESS


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
    ttff_s: Optional[float] = None          # time-to-first-chunk (streaming only, client-measured)
    rtf: Optional[float] = None             # real-time factor for this request
    cache_hit: bool = False                 # served from WAV cache, no LLM
    llm_ttft_ms: Optional[int] = None       # ms to first LLM speech token (server-measured)
    decoder_ttft_ms: Optional[int] = None   # ms to first decode_async completion (server-measured)




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
    streaming: bool = True,
    save_chunks: bool = False,
) -> RequestResult:
    call_id = str(uuid.uuid4())
    text_id = str(uuid.uuid4())
    url = f"ws://localhost:{port}/ws/{call_id}"

    _log(req_id, port, f"connecting → {url}")
    try:
        async with websockets.connect(url, open_timeout=5, max_size=100 * 1024 * 1024) as ws:
            _log(req_id, port, "connected")

            # Pick text: fixed > bench texts > english (non-hindi voices) > fallback
            if _FIXED_SENTENCE:
                text = _FIXED_SENTENCE
            elif _BENCH_TEXTS:
                text = _BENCH_TEXTS[req_id % len(_BENCH_TEXTS)]
            elif _VOICE_ID in ("simran", "british_rose"):
                text = _ENGLISH_AMERICAN[req_id % len(_ENGLISH_AMERICAN)]
            else:
                text = _FALLBACK_TEXTS[req_id % len(_FALLBACK_TEXTS)]
            req = {
                "type": "synthesize",
                "call_id": call_id,
                "text_id": text_id,
                "text": text,
                **({"voice_id": _VOICE_ID} if _VOICE_ID else {}),
                **({"skip_decoder": True} if skip_decoder else {}),
                **({"streaming": True} if streaming else {}),
            }
            await ws.send(json.dumps(req))
            _log(req_id, port, f"sent {'streaming' if streaming else 'synthesize'} request")

            t0 = time.time()

            if streaming:
                return await _recv_streaming(req_id, port, out_dir, ws, call_id, text_id, t0, save_chunks)

            _log(req_id, port, "waiting for WS response…")
            raw = await ws.recv()
            latency = round(time.time() - t0, 3)
            # Combined frame: json_bytes + wav_bytes, or plain JSON text frame
            if isinstance(raw, bytes):
                end = raw.index(b'}') + 1
                msg = json.loads(raw[:end])
                wav_data = raw[end:]
            else:
                msg = json.loads(raw)
                wav_data = await ws.recv()
                if isinstance(wav_data, str):
                    wav_data = wav_data.encode()

            _log(req_id, port, f"received type={msg.get('type')}  latency={latency}s")

            if msg.get("type") == "error":
                _log(req_id, port, f"FAIL gateway error: {msg.get('error')}")
                return RequestResult(req_id, port, False, latency, None,
                                     msg.get("error"), 0, 0, None, None)

            wav_path: Optional[Path] = None
            token_chars = len(msg.get("audio_tokens", ""))
            llm_s = msg.get("llm_s")
            decode_s = msg.get("decode_s")
            wav_bytes_len = len(wav_data)

            if not skip_decoder:
                if not wav_data:
                    _log(req_id, port, f"FAIL empty WAV bytes")
                    return RequestResult(req_id, port, False, latency, None,
                                         "empty WAV bytes", 0, token_chars, llm_s, decode_s)
                wav_path = out_dir / f"req{req_id:04d}_port{port}.wav"
                wav_path.write_bytes(wav_data)
            _log(req_id, port, f"OK  {wav_bytes_len}B WAV → {wav_path.name if wav_path else '-'}  llm_s={llm_s}  decode_s={decode_s}")

            return RequestResult(req_id, port, True, latency, wav_path,
                                 None, wav_bytes_len, token_chars, llm_s, decode_s)

    except Exception as e:
        err = str(e) or type(e).__name__
        _log(req_id, port, f"FAIL {type(e).__name__}: {err}")
        return RequestResult(req_id, port, False, 0.0, None, err, 0, 0, None, None)


def _wav_chunks_to_combined(chunk_wavs: list[bytes]) -> bytes:
    """Concatenate WAV chunks into one valid WAV file by decoding each and re-encoding."""
    import io
    import struct

    def _pcm_from_wav(data: bytes) -> tuple[bytes, int, int]:
        """Extract raw PCM from a WAV, return (pcm, sample_rate, num_channels)."""
        # Minimal WAV parser: find 'data' chunk
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return data, 16000, 1  # fallback: treat as raw PCM
        pos = 12
        sr, ch = 16000, 1
        while pos + 8 <= len(data):
            chunk_id = data[pos:pos+4]
            chunk_sz = struct.unpack_from("<I", data, pos+4)[0]
            if chunk_id == b"fmt ":
                ch = struct.unpack_from("<H", data, pos+10)[0]
                sr = struct.unpack_from("<I", data, pos+12)[0]
            elif chunk_id == b"data":
                return data[pos+8 : pos+8+chunk_sz], sr, ch
            pos += 8 + chunk_sz
        return b"", sr, ch

    if not chunk_wavs:
        return b""
    if len(chunk_wavs) == 1:
        return chunk_wavs[0]

    all_pcm = b""
    sr, ch = 16000, 1
    for wav in chunk_wavs:
        pcm, sr, ch = _pcm_from_wav(wav)
        all_pcm += pcm

    # Build a new WAV header around the concatenated PCM
    bits = 16
    byte_rate = sr * ch * bits // 8
    block_align = ch * bits // 8
    data_sz = len(all_pcm)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_sz))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, ch, sr, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_sz))
    buf.write(all_pcm)
    return buf.getvalue()


async def _recv_streaming(
    req_id: int,
    port: int,
    out_dir: Path,
    ws,
    call_id: str,
    text_id: str,
    t0: float,
    save_chunks: bool,
) -> RequestResult:
    """Receive streamed audio_chunk frames; save concatenated WAV at audio_done."""
    chunk_wavs: list[bytes] = []
    llm_s = None
    decode_s = None
    llm_ttft_ms = None
    decoder_ttft_ms = None
    total_tokens = 0
    first_chunk_latency: Optional[float] = None
    wav_path: Optional[Path] = None
    is_cache_hit = False

    try:
        while True:
            raw = await ws.recv()
            # Combined frame: json_bytes + wav_bytes in one binary message.
            # Find the end of the JSON object by tracking brace depth.
            if isinstance(raw, bytes):
                depth = 0
                end = 0
                for i, b in enumerate(raw):
                    if b == ord('{'):
                        depth += 1
                    elif b == ord('}'):
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                msg = json.loads(raw[:end])
                wav_inline = raw[end:]
            else:
                msg = json.loads(raw)
                wav_inline = None
            mtype = msg.get("type")

            if mtype == "error":
                _log(req_id, port, f"FAIL stream error: {msg.get('error')}")
                return RequestResult(req_id, port, False, round(time.time() - t0, 3), None,
                                     msg.get("error"), 0, 0, None, None)

            if mtype == "audio_chunk":
                chunk_idx = msg.get("chunk_index", 0)
                n_tok = msg.get("tokens", 0)
                total_tokens += n_tok
                if msg.get("cache_hit"):
                    is_cache_hit = True

                # Inline wav from combined frame, or separate binary frame (fallback)
                if wav_inline is not None:
                    wav_chunk = wav_inline
                else:
                    wav_chunk = await ws.recv()
                    if isinstance(wav_chunk, str):
                        wav_chunk = wav_chunk.encode()

                if first_chunk_latency is None:
                    first_chunk_latency = round(time.time() - t0, 3)
                    _log(req_id, port, f"first_chunk  latency={first_chunk_latency}s  tokens={n_tok}")

                chunk_wavs.append(wav_chunk)

                if save_chunks:
                    chunk_path = out_dir / f"req{req_id:04d}_port{port}_chunk{chunk_idx:03d}.wav"
                    chunk_path.write_bytes(wav_chunk)

            elif mtype == "audio_done":
                latency         = round(time.time() - t0, 3)
                llm_s           = msg.get("llm_s")
                decode_s        = msg.get("decode_s")
                rtf             = msg.get("rtf")
                llm_ttft_ms     = msg.get("llm_ttft_ms")
                decoder_ttft_ms = msg.get("decoder_ttft_ms")
                chunks          = msg.get("chunks", len(chunk_wavs))
                total_wav_b     = sum(len(w) for w in chunk_wavs)

                if chunk_wavs:
                    wav_path = out_dir / f"req{req_id:04d}_port{port}.wav"
                    wav_path.write_bytes(_wav_chunks_to_combined(chunk_wavs))

                _log(req_id, port,
                     f"OK  stream_done  chunks={chunks}  tokens={total_tokens}"
                     f"  {total_wav_b}B → {wav_path.name if wav_path else '-'}"
                     f"  ttff={first_chunk_latency}s  llm_ttft={llm_ttft_ms}ms"
                     f"  decoder_ttft={decoder_ttft_ms}ms  total={latency}s"
                     f"  llm_s={llm_s}  decode_s={decode_s}  rtf={rtf}")

                return RequestResult(req_id, port, True, latency, wav_path,
                                     None, total_wav_b, total_tokens * 20, llm_s, decode_s,
                                     ttff_s=first_chunk_latency, rtf=rtf,
                                     llm_ttft_ms=llm_ttft_ms, decoder_ttft_ms=decoder_ttft_ms)

    except Exception as e:
        err = str(e) or type(e).__name__
        _log(req_id, port, f"FAIL stream {type(e).__name__}: {err}")
        return RequestResult(req_id, port, False, round(time.time() - t0, 3), None, err, 0, 0, None, None)


# ---------------------------------------------------------------------------
# Server management (managed launch mode)
# ---------------------------------------------------------------------------
_FLOWTTS_DIR = Path.home() / "FlowTTS"
_DEFAULT_CTRL_PORT = 8764

# Prefer the project venv's Python for the server subprocess so it gets
# onnxruntime-gpu, torch+CUDA, and all other heavy deps — even when the test
# is invoked via the system Python.
_VENV_PYTHON = _FLOWTTS_DIR / ".venv" / "bin" / "python3"
_SERVER_PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


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
        _SERVER_PYTHON, "-m", "flowtts.server",
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
    base_port: int = 8080,
    save_audio: Optional[str] = None,
    skip_decoder: bool = False,
    streaming: bool = True,
    save_chunks: bool = False,
    # external-server args
    ports: Optional[List[int]] = None,
) -> List[RequestResult]:
    global _BENCH_TEXTS

    if not _BENCH_TEXTS:
        _BENCH_TEXTS = _load_bench_texts()
        if _BENCH_TEXTS:
            print(f"[bench] loaded {len(_BENCH_TEXTS)} sentences", flush=True)
        else:
            print(f"[bench] no bench texts, using built-in {len(_FALLBACK_TEXTS)} fallback sentences", flush=True)

    server_proc: Optional[subprocess.Popen] = None
    active_ports: List[int] = [base_port]
    _already_running = False

    if launch:
        # ── Managed: reuse existing server if already ready, else launch ──────
        try:
            data = _ctrl_get(ctrl_port, "/ready", timeout=1.0)
            _already_running = bool(data.get("ready"))
        except Exception:
            pass

        if _already_running:
            print(f"[server] reusing running server on ctrl=:{ctrl_port} (ref_audio stays loaded)", flush=True)
        else:
            server_proc = _launch_server(ctrl_port, save_audio)
            try:
                await _wait_server_ready(ctrl_port)
            except TimeoutError as e:
                server_proc.kill()
                print(f"[server] FATAL: {e}", flush=True)
                sys.exit(1)

        # Open exactly `concurrency` WS ports starting at base_port (skip already-open ones)
        ws_ports: List[int] = []
        for i in range(concurrency):
            p = base_port + i
            if not _port_open(p):
                await _open_port(ctrl_port, p)
            ws_ports.append(p)
        print(f"[server] using {len(ws_ports)} port(s): {ws_ports}", flush=True)
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
                # No explicit ports — use only the base port
                active_ports = [base_port]
                print(f"[server] using single port: {active_ports}", flush=True)
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

    # Build the final port list to round-robin across.
    # active_ports is set in managed/external branches above; fall back to base_port.
    routing_ports: List[int] = active_ports if active_ports else [base_port]

    print(f"\n{'='*60}")
    if len(routing_ports) == 1:
        print(f"mode={mode}  requests={n_requests}  port={routing_ports[0]}  (each request = unique WS connection)")
    else:
        print(f"mode={mode}  requests={n_requests}  ports={routing_ports}  (round-robin, unique WS connection per request)")
    print(f"output → {out_dir}")
    print(f"{'='*60}\n")

    tasks = [
        _run_one(i, routing_ports[i % len(routing_ports)], mode, out_dir, worker,
                 skip_decoder=skip_decoder, streaming=streaming, save_chunks=save_chunks)
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
    elif launch and _already_running:
        print("[server] left running (was already up before test)", flush=True)

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

    cache_hits = [r for r in passed if r.cache_hit]
    llm_hits   = [r for r in passed if not r.cache_hit]

    has_ttff     = any(r.ttff_s          is not None for r in results)
    has_llm_ttft = any(r.llm_ttft_ms     is not None for r in results)
    has_dec_ttft = any(r.decoder_ttft_ms is not None for r in results)

    header = (
        f"{'req':>4}  {'port':>5}  {'ok':>4}  {'lat(s)':>7}  "
        + (f"{'ttff(s)':>7}  " if has_ttff else "")
        + (f"{'llm_ttft':>8}  " if has_llm_ttft else "")
        + (f"{'dec_ttft':>8}  " if has_dec_ttft else "")
        + f"{'llm_s':>6}  {'dec_s':>6}  {'bytes':>8}  {'tokens':>7}  detail"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in sorted(results, key=lambda x: x.req_id):
        detail       = str(r.wav_path.name) if r.wav_path else (r.error or "")
        ttff_col     = (f"{r.ttff_s:>7.3f}  "              if r.ttff_s          is not None else f"{'─':>7}  ")  if has_ttff     else ""
        llm_ttft_col = (f"{r.llm_ttft_ms/1000:>8.3f}  "    if r.llm_ttft_ms    is not None else f"{'─':>8}  ")  if has_llm_ttft else ""
        dec_ttft_col = (f"{r.decoder_ttft_ms/1000:>8.3f}  " if r.decoder_ttft_ms is not None else f"{'─':>8}  ") if has_dec_ttft else ""
        lines.append(
            f"{r.req_id:>4}  {r.port:>5}  {'✓' if r.passed else '✗':>4}  "
            f"{r.latency_s:>7.3f}  "
            + ttff_col + llm_ttft_col + dec_ttft_col
            + f"{r.llm_s if r.llm_s is not None else '-':>6}  "
            f"{r.decode_s if r.decode_s is not None else '-':>6}  "
            f"{r.wav_bytes:>8}  {r.token_chars:>7}  {detail}"
        )

    if passed:
        lats            = [r.latency_s      for r in passed]
        llms            = [r.llm_s          for r in passed if r.llm_s          is not None]
        decs            = [r.decode_s       for r in passed if r.decode_s       is not None]
        ttffs           = [r.ttff_s         for r in passed if r.ttff_s         is not None]
        rtfs            = [r.rtf            for r in passed if r.rtf            is not None]
        llm_ttfts       = [r.llm_ttft_ms    for r in passed if r.llm_ttft_ms    is not None]
        decoder_ttfts   = [r.decoder_ttft_ms for r in passed if r.decoder_ttft_ms is not None]

        def _fmt(vals: list, unit: str = "s") -> str:
            if not vals:
                return "n/a"
            sv = sorted(vals)
            p95 = sv[int(len(sv) * 0.95)]
            return f"min={min(vals):.3f}{unit}  avg={sum(vals)/len(vals):.3f}{unit}  p95={p95:.3f}{unit}  max={max(vals):.3f}{unit}"

        def _fmt_ms(vals: list) -> str:
            if not vals:
                return "n/a"
            return f"min={min(vals)}ms  avg={sum(vals)//len(vals)}ms  max={max(vals)}ms"

        lines.append(f"\n{'─'*60}")
        lines.append(f"  total latency    : {_fmt(lats)}")
        if ttffs:
            lines.append(f"  time-to-first : {_fmt(ttffs)}  (first audio chunk, client-measured)")
        if llm_ttfts:
            lines.append(f"  llm ttft      : {_fmt_ms(llm_ttfts)}  (first speech token from LLM)")
        if decoder_ttfts:
            lines.append(f"  decoder ttft  : {_fmt_ms(decoder_ttfts)}  (first decode_async done)")
        if llm_ttfts and decoder_ttfts and len(llm_ttfts) == len(decoder_ttfts):
            decode_lag = [d - l for d, l in zip(decoder_ttfts, llm_ttfts)]
            lines.append(f"  decode lag    : {_fmt_ms(decode_lag)}  (decoder_ttft - llm_ttft)")
        lines.append(f"  llm           : {_fmt(llms)}")
        lines.append(f"  decoder       : {_fmt(decs)}")
        if llms and decs and len(llms) == len(decs):
            overhead = [l - d for l, d in zip(llms, decs)]
            lines.append(f"  llm - decode  : {_fmt(overhead)}  (net inference)")
        if rtfs:
            over_rt = sum(1 for v in rtfs if v > 1.0)
            lines.append(f"  rtf           : {_fmt(rtfs, '')}  (realtime factor, <1 = faster than realtime)")
            lines.append(f"  rtf > 1.0     : {over_rt}/{len(rtfs)} requests  ({100*over_rt/len(rtfs):.1f}% slower than realtime)")
        lines.append(f"{'─'*60}")
    if failed:
        lines.append(f"\nFailed requests:")
        for r in failed:
            lines.append(f"  req{r.req_id:04d} port={r.port}: {r.error}")

    pct = round(len(cache_hits) / len(passed) * 100) if passed else 0
    lines.append(f"\n  cache_hits={len(cache_hits)}  llm_requests={len(llm_hits)}  ({pct}% cached)")
    lines.append(f"\n{'✓ ALL PASSED' if not failed else f'✗ {len(failed)} FAILED'}")
    lines.append(f"{'='*70}")

    text = "\n".join(lines)
    print(text)

    summary_file = out_dir / "summary.txt"
    summary_file.write_text(text)
    print(f"\n[output] {out_dir}/")

    # Append summary to llm.log so each test run is recorded alongside inference logs.
    try:
        with _LLM_LOG.open("a") as f:
            f.write("\n" + text + "\n")
    except OSError:
        pass  # server not running / log not writable — non-fatal

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


def _discover_ports(base: int = 8080, scan_range: int = 50) -> List[int]:
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
    global _VOICE_ID, _FIXED_SENTENCE, _BENCH_TEXTS
    out_dir = _make_out_dir()

    streaming        = getattr(args, "streaming", None)
    if streaming is None:
        streaming = _settings.streaming.enabled
    save_chunks      = getattr(args, "save_chunks", False)
    _VOICE_ID        = getattr(args, "voice", "") or ""
    _FIXED_SENTENCE  = getattr(args, "sentence", "") or ""
    cache_mix        = getattr(args, "cache_mix", None)

    if _VOICE_ID:
        print(f"[voice] using voice_id={_VOICE_ID!r}", flush=True)
    if _FIXED_SENTENCE:
        print(f"[sentence] repeating: {_FIXED_SENTENCE!r}", flush=True)
    if cache_mix and not _FIXED_SENTENCE:
        _BENCH_TEXTS = _build_cache_mix(args.requests, cache_mix, _VOICE_ID)
        if not _BENCH_TEXTS:
            print("[cache_mix] fallback to default texts", flush=True)

    if args.launch:
        results = await run_test(
            args.mode, args.requests, out_dir,
            launch=True,
            ctrl_port=args.ctrl_port,
            concurrency=args.concurrency,
            base_port=args.base_port,
            save_audio=args.save_audio,
            skip_decoder=args.skip_decoder,
            streaming=streaming,
            save_chunks=save_chunks,
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
            streaming=streaming,
            save_chunks=save_chunks,
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
    parser.add_argument("--streaming", action="store_true", default=None,
                        help="Use streaming mode (default: settings.streaming.enabled)")
    parser.add_argument("--save-chunks", dest="save_chunks", action="store_true", default=False,
                        help="In streaming mode, also save each individual chunk WAV (in addition to the concatenated file)")
    parser.add_argument("--voice", default="", choices=["", "simran", "tara", "vikram", "daya", "british_rose", "rani", "sana", "anita", "vanita", "sunita", "anika", "anika2", "monika", "saavi", "zara", "gargi"],
                        help="Voice ID to use for synthesis (default: server default)")
    parser.add_argument("--sentence", default="", metavar="TEXT",
                        help="Repeat this single sentence for all requests")
    parser.add_argument("--cache-mix", default=None, metavar="MIX",
                        help="Sentence mix: full/100=all cached, none/0=all uncached, half/50, or 0-100")

    # External-server port selection
    pg = parser.add_mutually_exclusive_group()
    pg.add_argument("--ports", type=str, default=None,
                    help="Explicit comma-separated port list (--no-launch)")
    pg.add_argument("--n-ports", type=int, default=None,
                    help="Number of sequential ports from --base-port (--no-launch)")
    parser.add_argument("--base-port", "--port", dest="base_port", type=int, default=8080,
                        help="Base WS port (default: 8080)")

    args = parser.parse_args()
    # If neither --launch nor --no-launch was given, auto-detect from other flags.
    if args.launch is None:
        _external_hints = {"--n-ports", "--ports", "--no-launch"}
        args.launch = not bool(_external_hints.intersection(sys.argv))
    asyncio.run(main(args))
