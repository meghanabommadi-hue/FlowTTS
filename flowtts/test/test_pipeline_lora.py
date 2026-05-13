"""End-to-end LoRA pipeline smoke test.

Like test_pipeline.py but every request carries a ``language`` tag so the
server routes it to the matching LoRA adapter.  Each language has its own
text list (named by language tag).  Requests are interleaved across all
active languages; summary shows per-language latency breakdowns.

Two execution modes:

  Managed (default, --launch):
    Launches flowtts.server with --ctrl-port, waits for ready, then opens
    WebSocket ports on demand via POST /ports/add.  Server is killed on exit.

  External (--no-launch):
    Connects to an already-running server.  Ports resolved from
    --ports / --n-ports / --base-port or auto-discovered.

Usage:
    # Run 40 requests (interleaved hi + ta) on a managed server, 9 ports
    python -m flowtts.test.test_pipeline_lora --requests 40 --concurrency 9

    # Hindi only, external server
    python -m flowtts.test.test_pipeline_lora --no-launch --languages hi --requests 20

    # Tamil only with streaming
    python -m flowtts.test.test_pipeline_lora --no-launch --languages ta --streaming

    # Both languages, explicit port list
    python -m flowtts.test.test_pipeline_lora --no-launch --ports 8765,8766,8767 --requests 30
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import websockets

from flowtts.core.config import settings as _settings

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_TEST_ROOT = Path.home() / "FlowTTS/test"
_LLM_LOG   = Path.home() / "FlowTTS/llm.log"


def _make_out_dir() -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = _TEST_ROOT / f"lora_test_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Per-language text lists  (keys = language tags matching config's lora_map)
# ---------------------------------------------------------------------------

hi: List[str] = [
    # short — numerals
    # "नमस्ते, मैं आपकी कैसे मदद कर सकती हूं?",
    # "क्या आप अपना नाम बता सकते हैं?",
    # "आपका खाता नंबर ९८७६५४३२१० है, कृपया confirm करें।",
    # "कृपया थोड़ा इंतज़ार करें।",
    # "आपकी समस्या हल हो गई है।",
    # "आपका बकाया ₹२,५०० है, कृपया आज ही जमा करें।",
    # "हम जल्द ही आपसे संपर्क करेंगे।",
    # "आपका भुगतान ₹१०,००० सफलतापूर्वक हो गया है।",
    # # short — English numerals in Hindi sentences
    # "आपका account number 9876543210 है, कृपया confirm करें।",
    # "आपका बकाया Rs. 2500 है, कृपया आज ही जमा करें।",
    # "आपकी EMI Rs. 3750 हर महीने देय है।",
    # medium
    "नमस्ते. मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से. क्या मैं customer name से बात कर रही हूं?",
    "आपके loan की किस्त ₹३,७५० अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "हमारे रिकॉर्ड के अनुसार आपका बकाया amount ₹५,००० है, कृपया जल्द से जल्द इसे जमा करें।",
    "आपकी EMI की due date ३० अप्रैल निकल चुकी है, late charge से बचने के लिए आज ही payment करें।",
    "आपके account नंबर ४५६७८९०१२३ पर ₹१५,००० का loan approve हुआ है, क्या आप details verify करेंगे?",
    "आपके loan की किस्त Rs. 3750 अभी तक नहीं आई है, क्या आप बता सकते हैं कि भुगतान कब होगा?",
    "आपके account number 4567890123 पर Rs. 15000 का loan approve हुआ है, क्या आप details verify करेंगे?",
    # long
    "आपकी loan application approve हो गई है और ₹५०,००० सीधे आपके bank account ७८९०१२३४५६ में transfer कर दिए जाएंगे, जिसमें २ से ३ कार्य दिवस लग सकते हैं।",
    "हमारी company की policy के अनुसार अगर payment ३० दिनों के अंदर नहीं होती तो आपके credit score पर असर पड़ सकता है, इसलिए कृपया समय पर ₹८,२५० का भुगतान करें।",
    "आप हमारे mobile app के माध्यम से अपनी ₹४,५०० की EMI pay कर सकते हैं, इसके अलावा NEFT, IMPS, या UPI का भी उपयोग किया जा सकता है।",
    "आपकी loan application approve हो गई है और Rs. 50000 सीधे आपके bank account 7890123456 में transfer कर दिए जाएंगे, जिसमें 2 से 3 कार्य दिवस लग सकते हैं।",
    "हमारी company की policy के अनुसार अगर payment 30 दिनों के अंदर नहीं होती तो आपके credit score पर असर पड़ सकता है, इसलिए कृपया समय पर Rs. 8250 का भुगतान करें।",
    "आपकी loan application approve हो गई है और loan amount सीधे आपके bank account में transfer कर दी जाएगी, जिसमें दो से तीन कार्य दिवस लग सकते हैं।",
    "आप हमारे mobile app के माध्यम से अपनी EMI pay कर सकते हैं, इसके अलावा NEFT, IMPS, या UPI का भी उपयोग किया जा सकता है, और किसी भी समस्या के लिए हमारी customer care team हमेशा available है।",
    "हमारे records के अनुसार आपका loan account number 1234567890 है और आपकी monthly EMI Rs. 4500 है जो हर महीने की 5 तारीख को deduct होती है।",
    "आपकी payment successfully receive हो गई है और आपका account अब up to date है, अगर आपको कोई और जानकारी चाहिए तो हमें call करें।",
    "आपके loan की next installment की due date 15 तारीख है और amount Rs. 6200 है, कृपया समय पर भुगतान करें ताकि कोई late fee न लगे।",
    "हम आपको सूचित करना चाहते हैं कि आपकी KYC verification pending है, कृपया अपने नजदीकी branch में जाकर या हमारे app के माध्यम से इसे complete करें।",
    # spoken-number style
    "ये कॉल आपके loan number जो three six nine पर end होता है, उसी के बारे में है। आपका EMI bounce हो गया है और total overdue amount sixteen zero one two rupees है।",
    "आपकी next EMI की due date five April है और amount two thousand three hundred rupees है, कृपया समय पर payment करें।",
    "आपका account number जो seven eight nine zero पर end होता है, उस पर last month की EMI receive नहीं हुई है, कृपया जल्द से जल्द payment करें।",
    "नमस्ते, मैं Bajaj Finance की तरफ से बात कर रही हूं। आपके loan account number ending in three four five six पर total due amount is twenty two thousand five hundred rupees, कृपया आज ही payment करें।",
]

ta: List[str] = [
    # short
    # "நமஸ்தே, நான் உங்களுக்கு எப்படி உதவ முடியும்?",
    # "உங்கள் பெயரை சொல்ல முடியுமா?",
    # "தயவுசெய்து கொஞ்சம் காத்திருங்கள்.",
    # "உங்கள் சிக்கல் தீர்க்கப்பட்டது.",
    # "நாங்கள் விரைவில் உங்களை தொடர்பு கொள்வோம்.",
    # "உங்கள் கணக்கு எண் 9876543210, தயவுசெய்து உறுதிப்படுத்தவும்.",
    # "உங்கள் நிலுவை தொகை ₹2,500, இன்றே செலுத்தவும்.",
    # "உங்கள் கட்டணம் ₹10,000 வெற்றிகரமாக பெறப்பட்டது.",
    # # short — call center
    # "Aarav உங்கள் payment successfully receive ஆகிவிட்டது.",
    # "உங்கள் OTP four five six seven share செய்யவும்.",
    # "உங்கள் EMI Rs. 3750 ஒவ்வொரு மாதமும் செலுத்த வேண்டும்.",
    # medium
    "நமஸ்தே. நான் Kapture Finance சார்பாக பேசுகிறேன். இது recorded line. நான் customer name பேசுகிறேனா?",
    "உங்கள் கடன் தவணை ₹3,750 இன்னும் வரவில்லை, எப்போது செலுத்துவீர்கள் என்று சொல்ல முடியுமா?",
    "எங்கள் பதிவுகளின்படி, உங்கள் நிலுவை தொகை ₹5,000, தயவுசெய்து விரைவில் செலுத்தவும்.",
    "உங்கள் EMI due date ஏப்ரல் 30 கடந்துவிட்டது, late charge தவிர்க்க இன்றே payment செய்யவும்.",
    "உங்கள் account number 4567890123 இல் ₹15,000 loan approve ஆகியுள்ளது, details verify செய்வீர்களா?",
    "உங்கள் கடன் தவணை Rs. 3750 இன்னும் வரவில்லை, எப்போது செலுத்துவீர்கள்?",
    "உங்கள் account 4567890123 இல் Rs. 15000 loan approve ஆகியுள்ளது, details verify செய்யவும்.",
    # long
    "உங்கள் loan application approve ஆகியுள்ளது, ₹50,000 நேரடியாக உங்கள் bank account 7890123456 இல் transfer செய்யப்படும், இதற்கு 2 முதல் 3 working days ஆகும்.",
    "எங்கள் company policy படி 30 நாட்களுக்குள் payment இல்லாவிட்டால் உங்கள் credit score பாதிக்கப்படும், எனவே ₹8,250 சரியான நேரத்தில் செலுத்தவும்.",
    "நீங்கள் எங்கள் mobile app மூலம் ₹4,500 EMI செலுத்தலாம், மேலும் NEFT, IMPS, அல்லது UPI மூலமும் செலுத்தலாம்.",
    "உங்கள் loan application approve ஆகியுள்ளது, Rs. 50000 நேரடியாக உங்கள் bank account 7890123456 இல் transfer செய்யப்படும், 2 to 3 working days ஆகும்.",
    "உங்கள் loan account number 1234567890, மாதாந்திர EMI Rs. 4500, ஒவ்வொரு மாதமும் 5ம் தேதி deduct ஆகும்.",
    "உங்கள் payment வெற்றிகரமாக பெறப்பட்டது, உங்கள் account தற்போது up to date ஆகியுள்ளது, மேலும் தகவல்களுக்கு எங்களை call செய்யவும்.",
    "உங்கள் அடுத்த தவணை due date 15ம் தேதி, தொகை Rs. 6200, late fee தவிர்க்க சரியான நேரத்தில் செலுத்தவும்.",
    "உங்கள் KYC verification pending ஆகியுள்ளது, தயவுசெய்து அருகிலுள்ள branch ல் அல்லது app மூலம் complete செய்யவும்.",
    # spoken-number style
    "இது உங்கள் loan number three six nine இல் முடியும் கடனைப் பற்றியது. உங்கள் EMI bounce ஆகியுள்ளது, மொத்த நிலுவை sixteen zero one two rupees.",
    "உங்கள் அடுத்த EMI due date five April, தொகை two thousand three hundred rupees, சரியான நேரத்தில் payment செய்யவும்.",
    "நான் Bajaj Finance சார்பாக பேசுகிறேன். உங்கள் account ending in three four five six இல் total due twenty two thousand five hundred rupees, இன்றே payment செய்யவும்.",
]

# Map tag → list  (mirrors config's language_lora_map keys)
LANGUAGE_TEXTS: Dict[str, List[str]] = {
    "hi": hi,
    "ta": ta,
}


def _build_request_pairs(languages: List[str]) -> List[Tuple[str, str]]:
    """Build an interleaved list of (text, language_tag) pairs.

    Languages are round-robined so each consecutive block of len(languages)
    requests covers all active languages once.
    """
    lists = {lang: LANGUAGE_TEXTS[lang] for lang in languages if lang in LANGUAGE_TEXTS}
    if not lists:
        raise ValueError(f"No text lists found for languages: {languages}")

    # Find the max length across selected lists
    max_len = max(len(v) for v in lists.values())

    pairs: List[Tuple[str, str]] = []
    for i in range(max_len):
        for lang in languages:
            if lang not in lists:
                continue
            texts = lists[lang]
            pairs.append((texts[i % len(texts)], lang))
    return pairs


# ---------------------------------------------------------------------------
# Result type — extends base with language field
# ---------------------------------------------------------------------------
class RequestResult(NamedTuple):
    req_id: int
    port: int
    language: str
    passed: bool
    latency_s: float
    wav_path: Optional[Path]
    error: Optional[str]
    wav_bytes: int
    token_chars: int
    llm_s: Optional[float]
    decode_s: Optional[float]
    ttff_s: Optional[float] = None
    rtf: Optional[float] = None
    llm_ttft_ms: Optional[int] = None
    decoder_ttft_ms: Optional[int] = None


# ---------------------------------------------------------------------------
# Single request runner
# ---------------------------------------------------------------------------
def _log(req_id: int, port: int, language: str, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] req{req_id:04d} port={port} [{language}]  {msg}", flush=True)


async def _run_one(
    req_id: int,
    port: int,
    language: str,
    text: str,
    out_dir: Path,
    streaming: bool = False,
    save_chunks: bool = False,
) -> RequestResult:
    call_id = str(uuid.uuid4())
    text_id = str(uuid.uuid4())
    url = f"ws://localhost:{port}/ws/{call_id}"

    _log(req_id, port, language, f"connecting → {url}")
    try:
        async with websockets.connect(url, open_timeout=5, max_size=100 * 1024 * 1024) as ws:
            _log(req_id, port, language, "connected")

            req = {
                "type": "synthesize",
                "call_id": call_id,
                "text_id": text_id,
                "text": text,
                "language": language,
                **({"streaming": True} if streaming else {}),
            }
            await ws.send(json.dumps(req))
            _log(req_id, port, language, f"sent {'streaming' if streaming else 'synthesize'}  text={text[:50]!r}")

            t0 = time.time()

            if streaming:
                return await _recv_streaming(req_id, port, language, out_dir, ws, call_id, text_id, t0, save_chunks)

            _log(req_id, port, language, "waiting for response…")
            raw = await ws.recv()
            latency = round(time.time() - t0, 3)
            msg = json.loads(raw)

            _log(req_id, port, language, f"received type={msg.get('type')}  latency={latency}s")

            if msg.get("type") == "error":
                _log(req_id, port, language, f"FAIL gateway error: {msg.get('error')}")
                return RequestResult(req_id, port, language, False, latency,
                                     None, msg.get("error"), 0, 0, None, None)

            # Frame 2: raw WAV bytes
            wav_data = await ws.recv()
            if isinstance(wav_data, str):
                wav_data = wav_data.encode()

            token_chars = len(msg.get("audio_tokens", ""))
            llm_s = msg.get("llm_s")
            decode_s = msg.get("decode_s")
            rtf = msg.get("rtf")
            wav_bytes_len = len(wav_data)

            wav_path: Optional[Path] = None
            if wav_data:
                wav_path = out_dir / f"req{req_id:04d}_port{port}_{language}.wav"
                wav_path.write_bytes(wav_data)
            else:
                _log(req_id, port, language, "FAIL empty WAV bytes")
                return RequestResult(req_id, port, language, False, latency,
                                     None, "empty WAV bytes", 0, token_chars, llm_s, decode_s)

            _log(req_id, port, language,
                 f"OK  {wav_bytes_len}B → {wav_path.name}  llm_s={llm_s}  decode_s={decode_s}  rtf={rtf}")
            return RequestResult(req_id, port, language, True, latency, wav_path,
                                 None, wav_bytes_len, token_chars, llm_s, decode_s, rtf=rtf)

    except Exception as e:
        err = str(e) or type(e).__name__
        _log(req_id, port, language, f"FAIL {type(e).__name__}: {err}")
        return RequestResult(req_id, port, language, False, 0.0, None, err, 0, 0, None, None)


def _wav_chunks_to_combined(chunk_wavs: list) -> bytes:
    """Concatenate WAV chunks into one valid WAV file."""
    import io
    import struct

    def _pcm_from_wav(data: bytes):
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return data, 16000, 1
        pos = 12
        sr, ch = 16000, 1
        while pos + 8 <= len(data):
            chunk_id = data[pos:pos+4]
            chunk_sz = struct.unpack_from("<I", data, pos+4)[0]
            if chunk_id == b"fmt ":
                ch = struct.unpack_from("<H", data, pos+10)[0]
                sr = struct.unpack_from("<I", data, pos+12)[0]
            elif chunk_id == b"data":
                return data[pos+8:pos+8+chunk_sz], sr, ch
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
    language: str,
    out_dir: Path,
    ws,
    call_id: str,
    text_id: str,
    t0: float,
    save_chunks: bool,
) -> RequestResult:
    chunk_wavs: list = []
    llm_s = None
    decode_s = None
    total_tokens = 0
    first_chunk_latency: Optional[float] = None
    wav_path: Optional[Path] = None

    try:
        while True:
            raw = await ws.recv()

            # Server sends audio_chunk as one binary frame: JSON header + PCM bytes.
            # Split by scanning for the closing brace of the JSON object.
            wav_chunk: Optional[bytes] = None
            if isinstance(raw, bytes):
                depth = 0
                end = 0
                for end, b in enumerate(raw):
                    if b == ord('{'):
                        depth += 1
                    elif b == ord('}'):
                        depth -= 1
                        if depth == 0:
                            break
                try:
                    msg = json.loads(raw[:end + 1])
                except Exception:
                    continue
                wav_chunk = raw[end + 1:]
            else:
                msg = json.loads(raw)

            mtype = msg.get("type")

            if mtype == "error":
                _log(req_id, port, language, f"FAIL stream error: {msg.get('error')}")
                return RequestResult(req_id, port, language, False, round(time.time() - t0, 3),
                                     None, msg.get("error"), 0, 0, None, None)

            if mtype == "audio_chunk":
                chunk_idx = msg.get("chunk_index", 0)
                n_tok = msg.get("tokens", 0)
                total_tokens += n_tok

                if wav_chunk is None:
                    wav_chunk = await ws.recv()
                    if isinstance(wav_chunk, str):
                        wav_chunk = wav_chunk.encode()

                if first_chunk_latency is None:
                    first_chunk_latency = round(time.time() - t0, 3)
                    _log(req_id, port, language,
                         f"first_chunk  latency={first_chunk_latency}s  tokens={n_tok}")

                chunk_wavs.append(wav_chunk)

                if save_chunks:
                    chunk_path = out_dir / f"req{req_id:04d}_port{port}_{language}_chunk{chunk_idx:03d}.wav"
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
                    wav_path = out_dir / f"req{req_id:04d}_port{port}_{language}.wav"
                    wav_path.write_bytes(_wav_chunks_to_combined(chunk_wavs))

                _log(req_id, port, language,
                     f"OK  stream_done  chunks={chunks}  tokens={total_tokens}"
                     f"  {total_wav_b}B → {wav_path.name if wav_path else '-'}"
                     f"  ttff={first_chunk_latency}s  llm_ttft={llm_ttft_ms}ms"
                     f"  decoder_ttft={decoder_ttft_ms}ms  total={latency}s  rtf={rtf}")

                return RequestResult(req_id, port, language, True, latency, wav_path,
                                     None, total_wav_b, total_tokens * 20, llm_s, decode_s,
                                     ttff_s=first_chunk_latency, rtf=rtf,
                                     llm_ttft_ms=llm_ttft_ms, decoder_ttft_ms=decoder_ttft_ms)

    except Exception as e:
        err = str(e) or type(e).__name__
        _log(req_id, port, language, f"FAIL stream {type(e).__name__}: {err}")
        return RequestResult(req_id, port, language, False, round(time.time() - t0, 3),
                             None, err, 0, 0, None, None)


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------
_FLOWTTS_DIR    = Path.home() / "FlowTTS"
_DEFAULT_CTRL_PORT = 8764
_VENV_PYTHON    = _FLOWTTS_DIR / ".venv" / "bin" / "python3"
_SERVER_PYTHON  = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


def _ctrl_url(ctrl_port: int, path: str) -> str:
    return f"http://127.0.0.1:{ctrl_port}{path}"


def _ctrl_get(ctrl_port: int, path: str, timeout: float = 2.0):
    with urllib.request.urlopen(_ctrl_url(ctrl_port, path), timeout=timeout) as r:
        return json.loads(r.read())


def _ctrl_post(ctrl_port: int, path: str, timeout: float = 2.0):
    req = urllib.request.Request(_ctrl_url(ctrl_port, path), method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _launch_server(ctrl_port: int, save_audio: Optional[str] = None) -> subprocess.Popen:
    cmd = [
        _SERVER_PYTHON, "-m", "flowtts.server",
        "--ports", "0",
        "--ctrl-port", str(ctrl_port),
    ]
    if save_audio:
        cmd += ["--save-audio", save_audio]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_FLOWTTS_DIR)
    return subprocess.Popen(cmd, cwd=str(_FLOWTTS_DIR), env=env,
                            stdout=sys.stdout, stderr=sys.stderr)


async def _wait_server_ready(ctrl_port: int, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    print(f"[server] waiting for model load (ctrl=:{ctrl_port})…", flush=True)
    while time.time() < deadline:
        try:
            data = _ctrl_get(ctrl_port, "/ready", timeout=1.0)
            if data.get("ready"):
                print(f"[server] ready  existing_ports={data.get('ports')}", flush=True)
                return
        except Exception:
            pass
        await asyncio.sleep(2.0)
    raise TimeoutError(f"server not ready after {timeout}s")


async def _open_port(ctrl_port: int, ws_port: int) -> int:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: _ctrl_post(ctrl_port, f"/ports/add?port={ws_port}"),
    )
    for _ in range(20):
        if _port_open(ws_port):
            return ws_port
        await asyncio.sleep(0.05)
    raise OSError(f"port {ws_port} did not open after /ports/add")


def _resolve_ports(ports_arg: Optional[str], base_port: int, n_ports: Optional[int]) -> List[int]:
    if ports_arg:
        return [int(p.strip()) for p in ports_arg.split(",") if p.strip()]
    if n_ports is not None:
        return [base_port + i for i in range(n_ports)]
    live = [p for p in range(base_port, base_port + 50) if _port_open(p)]
    if not live:
        print(f"[ports] no live ports found from {base_port}, defaulting to [{base_port}]")
        return [base_port]
    print(f"[ports] auto-discovered: {live}")
    return live


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------
async def run_test(
    n_requests: int,
    out_dir: Path,
    languages: List[str],
    *,
    launch: bool = True,
    ctrl_port: int = _DEFAULT_CTRL_PORT,
    concurrency: int = 9,
    base_port: int = 8765,
    save_audio: Optional[str] = None,
    streaming: bool = False,
    save_chunks: bool = False,
    ports: Optional[List[int]] = None,
) -> List[RequestResult]:

    # Build the full interleaved (text, language) list
    pairs = _build_request_pairs(languages)
    print(f"[lora] languages={languages}  text pool={len(pairs)} pairs (cycling for {n_requests} requests)", flush=True)

    server_proc: Optional[subprocess.Popen] = None
    active_ports: List[int] = [base_port]
    _already_running = False

    if launch:
        try:
            data = _ctrl_get(ctrl_port, "/ready", timeout=1.0)
            _already_running = bool(data.get("ready"))
        except Exception:
            pass

        if _already_running:
            print(f"[server] reusing running server on ctrl=:{ctrl_port}", flush=True)
        else:
            server_proc = _launch_server(ctrl_port, save_audio)
            try:
                await _wait_server_ready(ctrl_port)
            except TimeoutError as e:
                server_proc.kill()
                print(f"[server] FATAL: {e}", flush=True)
                sys.exit(1)

        ws_ports: List[int] = []
        for i in range(concurrency):
            p = base_port + i
            if not _port_open(p):
                await _open_port(ctrl_port, p)
            ws_ports.append(p)
        print(f"[server] using {len(ws_ports)} port(s): {ws_ports}", flush=True)
        active_ports = ws_ports

    else:
        if ctrl_port:
            if ports is not None:
                opened, already = [], []
                for p in ports:
                    if not _port_open(p):
                        await _open_port(ctrl_port, p)
                        opened.append(p)
                    else:
                        already.append(p)
                if opened:
                    print(f"[server] opened new port(s): {opened}", flush=True)
                if already:
                    print(f"[server] reusing existing port(s): {already}", flush=True)
                active_ports = ports
            else:
                data = _ctrl_get(ctrl_port, "/ports")
                active_ports = data.get("ports", [])
                print(f"[server] using {len(active_ports)} existing port(s): {active_ports}", flush=True)
        else:
            if ports is None:
                ports = _resolve_ports(None, base_port, None)
            live = [p for p in ports if _port_open(p)]
            dead = [p for p in ports if p not in live]
            if not live:
                print(f"[ports] ERROR: no ports reachable: {dead}", flush=True)
                sys.exit(1)
            if dead:
                print(f"[ports] WARNING: dropped dead ports: {dead}", flush=True)
            active_ports = live
            print(f"[ports] using {len(active_ports)} live port(s): {active_ports}", flush=True)

    routing_ports = active_ports if active_ports else [base_port]

    print(f"\n{'='*60}", flush=True)
    print(f"languages={languages}  requests={n_requests}  ports={routing_ports}", flush=True)
    print(f"streaming={streaming}  output → {out_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    tasks = [
        _run_one(
            i,
            routing_ports[i % len(routing_ports)],
            pairs[i % len(pairs)][1],   # language tag
            pairs[i % len(pairs)][0],   # text
            out_dir,
            streaming=streaming,
            save_chunks=save_chunks,
        )
        for i in range(n_requests)
    ]
    results: List[RequestResult] = await asyncio.gather(*tasks)

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
def _print_summary(results: List[RequestResult], out_dir: Path) -> bool:
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    all_langs = sorted({r.language for r in results})

    has_ttff     = any(r.ttff_s          is not None for r in results)
    has_llm_ttft = any(r.llm_ttft_ms    is not None for r in results)
    has_dec_ttft = any(r.decoder_ttft_ms is not None for r in results)
    has_rtf      = any(r.rtf             is not None for r in results)

    lines: List[str] = []
    lines.append(f"\n{'='*70}")
    lines.append(f"LORA SUMMARY  total={len(results)}  passed={len(passed)}  failed={len(failed)}")
    lines.append(f"{'='*70}")

    header = (
        f"{'req':>4}  {'port':>5}  {'lang':>4}  {'ok':>4}  {'lat(s)':>7}  "
        + (f"{'ttff(s)':>7}  " if has_ttff else "")
        + (f"{'llm_ttft':>8}  " if has_llm_ttft else "")
        + (f"{'dec_ttft':>8}  " if has_dec_ttft else "")
        + f"{'llm_s':>6}  {'dec_s':>6}  "
        + (f"{'rtf':>5}  " if has_rtf else "")
        + "detail"
    )
    lines.append(header)
    lines.append("-" * (len(header) + 4))

    for r in sorted(results, key=lambda x: x.req_id):
        detail        = str(r.wav_path.name) if r.wav_path else (r.error or "")
        ttff_col      = (f"{r.ttff_s:>7.3f}  "              if r.ttff_s          is not None else f"{'─':>7}  ")  if has_ttff     else ""
        llm_ttft_col  = (f"{r.llm_ttft_ms/1000:>8.3f}  "    if r.llm_ttft_ms    is not None else f"{'─':>8}  ")  if has_llm_ttft else ""
        dec_ttft_col  = (f"{r.decoder_ttft_ms/1000:>8.3f}  " if r.decoder_ttft_ms is not None else f"{'─':>8}  ") if has_dec_ttft else ""
        rtf_col       = (f"{r.rtf:>5.3f}  " if r.rtf is not None else f"{'─':>5}  ") if has_rtf else ""
        lines.append(
            f"{r.req_id:>4}  {r.port:>5}  {r.language:>4}  {'✓' if r.passed else '✗':>4}  "
            f"{r.latency_s:>7.3f}  "
            + ttff_col + llm_ttft_col + dec_ttft_col
            + f"{r.llm_s if r.llm_s is not None else '-':>6}  "
            f"{r.decode_s if r.decode_s is not None else '-':>6}  "
            + rtf_col
            + detail
        )

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

    # Per-language stats
    for lang in all_langs:
        lang_results  = [r for r in passed if r.language == lang]
        if not lang_results:
            continue
        lats          = [r.latency_s      for r in lang_results]
        llms          = [r.llm_s          for r in lang_results if r.llm_s          is not None]
        decs          = [r.decode_s       for r in lang_results if r.decode_s       is not None]
        ttffs         = [r.ttff_s         for r in lang_results if r.ttff_s         is not None]
        rtfs          = [r.rtf            for r in lang_results if r.rtf            is not None]
        llm_ttfts     = [r.llm_ttft_ms    for r in lang_results if r.llm_ttft_ms    is not None]
        decoder_ttfts = [r.decoder_ttft_ms for r in lang_results if r.decoder_ttft_ms is not None]

        lines.append(f"\n  [{lang}]  {len(lang_results)} passed / {len([r for r in results if r.language == lang])} total")
        lines.append(f"  {'─'*60}")
        lines.append(f"    total latency : {_fmt(lats)}")
        if ttffs:
            lines.append(f"    time-to-first : {_fmt(ttffs)}  (first audio chunk, client-measured)")
        if llm_ttfts:
            lines.append(f"    llm ttft      : {_fmt_ms(llm_ttfts)}  (first speech token from LLM)")
        if decoder_ttfts:
            lines.append(f"    decoder ttft  : {_fmt_ms(decoder_ttfts)}  (first decode_async done)")
        if llm_ttfts and decoder_ttfts and len(llm_ttfts) == len(decoder_ttfts):
            decode_lag = [d - l for d, l in zip(decoder_ttfts, llm_ttfts)]
            lines.append(f"    decode lag    : {_fmt_ms(decode_lag)}  (decoder_ttft - llm_ttft)")
        lines.append(f"    llm           : {_fmt(llms)}")
        lines.append(f"    decoder       : {_fmt(decs)}")
        if llms and decs and len(llms) == len(decs):
            overhead = [l - d for l, d in zip(llms, decs)]
            lines.append(f"    llm - decode  : {_fmt(overhead)}  (net inference)")
        if rtfs:
            over_rt = sum(1 for v in rtfs if v > 1.0)
            lines.append(f"    rtf           : {_fmt(rtfs, '')}  (realtime factor, <1 = faster than realtime)")
            lines.append(f"    rtf > 1.0     : {over_rt}/{len(rtfs)} requests  ({100*over_rt/len(rtfs):.1f}% slower than realtime)")

    # Overall stats across all languages
    if passed:
        all_lats      = [r.latency_s      for r in passed]
        all_llms      = [r.llm_s          for r in passed if r.llm_s          is not None]
        all_decs      = [r.decode_s       for r in passed if r.decode_s       is not None]
        all_ttffs     = [r.ttff_s         for r in passed if r.ttff_s         is not None]
        all_rtfs      = [r.rtf            for r in passed if r.rtf            is not None]
        all_llm_ttfts = [r.llm_ttft_ms    for r in passed if r.llm_ttft_ms    is not None]
        all_dec_ttfts = [r.decoder_ttft_ms for r in passed if r.decoder_ttft_ms is not None]
        lines.append(f"\n  [all]  {len(passed)} passed")
        lines.append(f"  {'─'*60}")
        lines.append(f"    total latency : {_fmt(all_lats)}")
        if all_ttffs:
            lines.append(f"    time-to-first : {_fmt(all_ttffs)}  (first audio chunk, client-measured)")
        if all_llm_ttfts:
            lines.append(f"    llm ttft      : {_fmt_ms(all_llm_ttfts)}  (first speech token from LLM)")
        if all_dec_ttfts:
            lines.append(f"    decoder ttft  : {_fmt_ms(all_dec_ttfts)}  (first decode_async done)")
        if all_llm_ttfts and all_dec_ttfts and len(all_llm_ttfts) == len(all_dec_ttfts):
            decode_lag = [d - l for d, l in zip(all_dec_ttfts, all_llm_ttfts)]
            lines.append(f"    decode lag    : {_fmt_ms(decode_lag)}  (decoder_ttft - llm_ttft)")
        lines.append(f"    llm           : {_fmt(all_llms)}")
        lines.append(f"    decoder       : {_fmt(all_decs)}")
        if all_llms and all_decs and len(all_llms) == len(all_decs):
            overhead = [l - d for l, d in zip(all_llms, all_decs)]
            lines.append(f"    llm - decode  : {_fmt(overhead)}  (net inference)")
        if all_rtfs:
            over_rt = sum(1 for v in all_rtfs if v > 1.0)
            lines.append(f"    rtf           : {_fmt(all_rtfs, '')}  (realtime factor, <1 = faster than realtime)")
            lines.append(f"    rtf > 1.0     : {over_rt}/{len(all_rtfs)} requests  ({100*over_rt/len(all_rtfs):.1f}% slower than realtime)")

    if failed:
        lines.append(f"\nFailed requests:")
        for r in failed:
            lines.append(f"  req{r.req_id:04d} port={r.port} [{r.language}]: {r.error}")

    lines.append(f"\n{'✓ ALL PASSED' if not failed else f'✗ {len(failed)} FAILED'}")
    lines.append(f"{'='*70}")

    text = "\n".join(lines)
    print(text)

    summary_file = out_dir / "summary.txt"
    summary_file.write_text(text)
    print(f"\n[output] {out_dir}/")

    try:
        with _LLM_LOG.open("a") as f:
            f.write("\n" + text + "\n")
    except OSError:
        pass

    return len(failed) == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main(args: argparse.Namespace) -> None:
    out_dir = _make_out_dir()

    # Resolve which languages to test
    configured = list(_settings.tts_model.language_lora_map.keys())
    if args.languages:
        languages = [l.strip() for l in args.languages.split(",") if l.strip()]
        unknown = [l for l in languages if l not in LANGUAGE_TEXTS]
        if unknown:
            print(f"[ERROR] No text list for language(s): {unknown}", flush=True)
            print(f"        Available: {list(LANGUAGE_TEXTS.keys())}", flush=True)
            sys.exit(1)
        not_in_config = [l for l in languages if l not in configured]
        if not_in_config:
            print(f"[WARN] language(s) {not_in_config} not in config's language_lora_map — "
                  f"server will use default LoRA", flush=True)
    else:
        # Default: all languages that have both a text list and a configured LoRA
        languages = [l for l in configured if l in LANGUAGE_TEXTS]
        if not languages:
            languages = list(LANGUAGE_TEXTS.keys())
        print(f"[lora] no --languages given, testing all configured: {languages}", flush=True)

    streaming = args.streaming if args.streaming is not None else _settings.streaming.enabled

    if args.launch:
        results = await run_test(
            args.requests, out_dir, languages,
            launch=True,
            ctrl_port=args.ctrl_port,
            concurrency=args.concurrency,
            base_port=args.base_port,
            save_audio=args.save_audio,
            streaming=streaming,
            save_chunks=args.save_chunks,
        )
    else:
        if args.ctrl_port and args.ports is None and args.n_ports is None:
            port_list = None
        else:
            port_list = _resolve_ports(args.ports, args.base_port, args.n_ports)
            if port_list:
                print(f"[ports] resolved: {port_list}")
        results = await run_test(
            args.requests, out_dir, languages,
            launch=False,
            ctrl_port=args.ctrl_port,
            concurrency=args.concurrency,
            base_port=args.base_port,
            streaming=streaming,
            save_chunks=args.save_chunks,
            ports=port_list,
        )

    ok = _print_summary(results, out_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FlowTTS LoRA pipeline test — one language tag per request",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--requests", type=int, default=10,
                        help="Total number of requests to send (default: 10)")
    parser.add_argument("--languages", type=str, default=None,
                        help="Comma-separated language tags to test, e.g. 'hi,ta' "
                             "(default: all languages in config's language_lora_map)")

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--launch", dest="launch", action="store_true", default=None,
                     help="(default) Launch flowtts.server, open ports on demand")
    grp.add_argument("--no-launch", dest="launch", action="store_false",
                     help="Connect to an already-running server")

    parser.add_argument("--concurrency", type=int, default=9,
                        help="Number of WS ports to open in managed mode (default: 9)")
    parser.add_argument("--ctrl-port", type=int, default=_DEFAULT_CTRL_PORT,
                        help=f"Server control API port (default: {_DEFAULT_CTRL_PORT})")
    parser.add_argument("--save-audio", type=str, default=None, metavar="DIR",
                        help="Pass --save-audio DIR to the launched server")
    parser.add_argument("--streaming", action="store_true", default=None,
                        help="Use streaming mode (default: settings.streaming.enabled)")
    parser.add_argument("--save-chunks", dest="save_chunks", action="store_true", default=False,
                        help="In streaming mode, also save individual chunk WAVs")

    pg = parser.add_mutually_exclusive_group()
    pg.add_argument("--ports", type=str, default=None,
                    help="Explicit comma-separated port list (--no-launch)")
    pg.add_argument("--n-ports", type=int, default=None,
                    help="Number of sequential ports from --base-port (--no-launch)")
    parser.add_argument("--base-port", "--port", dest="base_port", type=int, default=8765,
                        help="Base WS port (default: 8765)")

    args = parser.parse_args()
    if args.launch is None:
        _external_hints = {"--n-ports", "--ports", "--no-launch"}
        args.launch = not bool(_external_hints.intersection(sys.argv))

    asyncio.run(main(args))
