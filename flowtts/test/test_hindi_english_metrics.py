#!/usr/bin/env python3
"""
Hindi / English quality & latency benchmark using the HuggingFace dataset:
  Shubhangi7/hindi_english_combined_hf_dataset

Pipeline position: CLIENT-SIDE QUALITY + LATENCY TEST
  Loads text samples from the HF dataset, classifies each as Hindi or English
  by script (Devanagari → Hindi, Latin → English), sends them to the FlowTTS
  WebSocket server, and reports per-language metrics:

  Metrics reported (split by language):
    - Synthesis error rate      : fraction of requests that errored / timed out
    - CER (Character Error Rate): ASR round-trip accuracy via Whisper large-v3
        text → TTS → wav → Whisper → transcript → CER vs original
    - TTFF (ms)                 : time to first audio chunk (first token latency)
    - Total latency (wall, ms)  : avg / min / max / p50 / p95 / p99
    - LLM latency (ms)          : avg / min / max / p50 / p95 / p99
    - Decode latency (ms)       : avg / min / max / p50 / p95 / p99
    - RTF                       : avg / min / max
    - Token count               : avg tokens generated
    - Throughput                : successful requests per second

  Language detection:
    A text is classified as Hindi if ≥ HINDI_SCRIPT_THRESHOLD of its
    non-whitespace characters are Devanagari (Unicode block 0900–097F).
    Everything else is treated as English (Hinglish / code-switched).

Usage:
    # Defaults: 50 samples, balanced Hindi/English, with ASR CER
    python flowtts/test/test_hindi_english_metrics.py

    # Skip ASR (faster, no GPU load for Whisper)
    python flowtts/test/test_hindi_english_metrics.py --no-asr

    # Use a smaller/faster Whisper model
    python flowtts/test/test_hindi_english_metrics.py --asr-model openai/whisper-small

    # More samples, save report
    python flowtts/test/test_hindi_english_metrics.py --samples 200 --save-report

    # Dry-run: classify dataset samples without hitting the server
    python flowtts/test/test_hindi_english_metrics.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import socket
import statistics
import time
import unicodedata
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import io
import struct

import numpy as np
import websockets
from websockets.exceptions import WebSocketException

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_NAME            = "Shubhangi7/hindi_english_combined_hf_dataset"
ENGLISH_DATASET_NAME    = "Salesforce/wikitext"   # pure English fallback
ENGLISH_DATASET_CONFIG  = "wikitext-2-raw-v1"
DEFAULT_HF_SPLIT        = "train"
DEFAULT_SAMPLES         = 50       # total samples to draw from the dataset
DEFAULT_BASE_PORT       = 8765
DEFAULT_N_PORTS         = 10
DEFAULT_TIMEOUT_S       = 60       # per-request timeout
WS_MAX_SIZE             = 100 * 1024 * 1024  # 100 MB
# ASR model — whisper-large-v3 handles both Hindi and English well
ASR_MODEL_HINDI         = "openai/whisper-large-v3"
ASR_MODEL_ENGLISH       = "openai/whisper-large-v3"
ASR_SAMPLE_RATE         = 16000

# A character is considered Devanagari if its Unicode codepoint is in 0900–097F
DEVANAGARI_START    = 0x0900
DEVANAGARI_END      = 0x097F

# NOTE: This dataset is a Hindi news + Hinglish corpus — nearly all text contains
# Devanagari.  We split by Devanagari character density:
#   "hindi"   → ≥ HINDI_THRESHOLD of non-space chars are Devanagari (pure Hindi)
#   "english" → < HINDI_THRESHOLD (English-dominant or heavily code-switched)
# This gives a meaningful TTS comparison: pure-script Hindi vs mixed/English text.
HINDI_SCRIPT_THRESHOLD = 0.80   # ≥80% Devanagari → "hindi"; else "english"


# ── Language detection ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Returns 'hindi' if ≥ HINDI_SCRIPT_THRESHOLD of non-whitespace chars are
    Devanagari, otherwise 'english'.
    """
    stripped = text.replace(" ", "").replace("\n", "")
    if not stripped:
        return "english"
    deva_count = sum(
        1 for ch in stripped
        if DEVANAGARI_START <= ord(ch) <= DEVANAGARI_END
    )
    return "hindi" if (deva_count / len(stripped)) >= HINDI_SCRIPT_THRESHOLD else "english"


# ── WAV parsing ───────────────────────────────────────────────────────────────

def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    """
    Decode one or more concatenated WAV blobs (PCM 16-bit, mono, 16 kHz)
    to a single float32 array [-1, 1].

    Each streaming chunk from the server is a complete WAV file with its own
    44-byte header.  We parse each header to find the data payload and strip
    the headers so only PCM samples are concatenated.
    """
    WAV_HEADER = 44
    pcm_parts: list[np.ndarray] = []
    offset = 0

    while offset < len(wav_bytes):
        remaining = len(wav_bytes) - offset
        if remaining < WAV_HEADER:
            break
        # RIFF header: bytes 4-8 = chunk size (file size - 8)
        chunk_size = struct.unpack_from("<I", wav_bytes, offset + 4)[0]
        total_size = chunk_size + 8   # include the 4-byte "RIFF" + 4-byte size
        payload = wav_bytes[offset + WAV_HEADER : offset + total_size]
        if payload:
            pcm_parts.append(np.frombuffer(payload, dtype=np.int16))
        offset += max(total_size, WAV_HEADER + 1)  # guard against infinite loop

    if not pcm_parts:
        return np.zeros(0, dtype=np.float32)
    pcm = np.concatenate(pcm_parts)
    return pcm.astype(np.float32) / 32768.0


# ── ASR (Whisper) ─────────────────────────────────────────────────────────────

class WhisperASR:
    """
    Single whisper-large-v3 model handling both Hindi and English.
    Loaded once; language is passed per-transcription call.

    Avoids the transformers pipeline() wrapper which allocates extra scratch
    buffers and causes bad_alloc when the TTS sglang model occupies most VRAM.
    """

    def __init__(
        self,
        hindi_model_id: str   = ASR_MODEL_HINDI,
        english_model_id: str = ASR_MODEL_ENGLISH,
    ) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        # Both are the same model — load once
        model_id = hindi_model_id  # same as english_model_id

        print(f"\n  Loading ASR model {model_id} …", flush=True)
        t0 = time.time()

        if torch.cuda.is_available():
            self._device = "cuda"
            self._dtype  = torch.float16
        else:
            self._device = "cpu"
            self._dtype  = torch.float32

        self._processor = WhisperProcessor.from_pretrained(model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            dtype=self._dtype,
            low_cpu_mem_usage=True,
        ).to(self._device)
        self._model.eval()

        free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024) if self._device == "cuda" else "n/a"
        print(f"  ASR ready on {self._device}  ({time.time()-t0:.1f}s)  free GPU: {free_mb} MB", flush=True)

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        """
        Transcribe float32 audio array.
        language: 'hindi' or 'english' (as stored in result dicts).
        """
        import torch

        if audio is None or len(audio) == 0:
            return ""

        lang_code = "hi" if language == "hindi" else "en"

        inputs = self._processor(audio, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt")
        inputs = {
            k: (v.to(self._device).to(self._dtype)
                if v.dtype in (torch.float32, torch.float64)
                else v.to(self._device))
            for k, v in inputs.items()
        }

        with torch.no_grad():
            ids = self._model.generate(
                **inputs,
                language=lang_code,
                task="transcribe",
                max_new_tokens=200,
            )
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ── CER computation ───────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for fair CER comparison."""
    import re
    text = text.lower().strip()
    # Remove common punctuation but keep Devanagari + Latin chars and spaces
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate using Levenshtein edit distance at the character level.

    CER = (S + D + I) / N
      S = substitutions, D = deletions, I = insertions
      N = number of characters in reference

    Returns 0.0 if reference is empty, capped at 1.0.
    """
    ref = list(_normalise(reference))
    hyp = list(_normalise(hypothesis))

    if not ref:
        return 0.0

    n, m = len(ref), len(hyp)
    # Standard DP edit distance
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return min(dp[m] / n, 1.0)


# ── Pure English sentence loader ─────────────────────────────────────────────

def load_english_samples(n: int, seed: int = 42) -> list[str]:
    """
    Load n pure English sentences from Salesforce/wikitext (wikitext-2-raw-v1).
    Text-only, no audio — fast and memory-safe.
    Filters to sentences that are 5–25 words, contain only Latin + common punct,
    and are not section headers.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub not found — pip install huggingface_hub")

    import re

    print(f"  Fetching English samples from {ENGLISH_DATASET_NAME} …", flush=True)

    # wikitext-2 train split is a single parquet file on the Hub
    path = hf_hub_download(
        repo_id=ENGLISH_DATASET_NAME,
        filename="wikitext-2-raw-v1/train-00000-of-00001.parquet",
        repo_type="dataset",
    )

    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["text"])
    raw_lines = table.column("text").to_pylist()

    # Filter: non-empty, not a header (= lines), 5–25 words, ASCII-dominant
    pool: list[str] = []
    for line in raw_lines:
        line = (line or "").strip()
        if not line or line.startswith(" ="):
            continue
        words = line.split()
        if not (5 <= len(words) <= 25):
            continue
        # Must be ≥ 90% ASCII printable (true English, not mixed script)
        ascii_ratio = sum(1 for c in line if ord(c) < 128) / len(line)
        if ascii_ratio < 0.90:
            continue
        pool.append(line)

    rng = random.Random(seed)
    selected = rng.sample(pool, min(n, len(pool)))
    print(f"  English pool: {len(pool)} sentences → sampled {len(selected)}", flush=True)
    return selected


# ── Dataset loading ───────────────────────────────────────────────────────────

def _iter_texts_from_arrow_shard(shard_path: str) -> list[str]:
    """
    Read only the 'text' column from a single Arrow IPC stream shard.
    The audio column is never decoded — it stays as raw bytes in the file
    and is skipped entirely by reading only the named column.
    """
    import pyarrow.ipc as ipc

    texts: list[str] = []
    with open(shard_path, "rb") as f:
        reader = ipc.open_stream(f)
        for batch in reader:
            texts.extend(batch.column("text").to_pylist())
    return texts


def load_samples(
    n_samples: int,
    hf_split: str,
    load_audio: bool = False,  # ignored — audio is always skipped to avoid OOM
    seed: int = 42,
    pure_english: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Load `n_samples` text entries from the HF dataset — text-only, no audio.

    Returns:
        texts     : list of raw text strings
        languages : list of 'hindi' | 'english' (one per text)

    Strategy:
        The dataset ships as 113 Arrow IPC shards, each containing both an
        'audio' (binary) and a 'text' column.  Using load_dataset(streaming=True)
        resolves all 113 shard URLs first (network round-trips) and then
        decodes audio bytes into RAM → std::bad_alloc.

        Instead we:
          1. Use hf_hub_download to fetch just shard 0 (~8 s, ~45 MB).
             One shard has ~5 000 rows which is enough for any sample count.
          2. Read only the 'text' column via pyarrow IPC — audio bytes are
             never touched.
          3. If one shard doesn't provide enough of a given language, download
             the next shard (up to MAX_SHARDS).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub not found.\n"
            "Install with:  pip install huggingface_hub\n"
        )

    print(f"\nLoading dataset '{DATASET_NAME}' (text-only, direct arrow) …", flush=True)

    TARGET_PER_LANG = n_samples // 2
    STOP_AT         = max(TARGET_PER_LANG * 4, 200)
    TOTAL_SHARDS    = 113
    MAX_SHARDS      = 3

    hindi_pool: list[str] = []

    # If pure_english mode: fetch English from wikitext, skip Hindi dataset English pool
    if pure_english:
        english_sel_list = load_english_samples(TARGET_PER_LANG, seed=seed)
    else:
        english_sel_list = None

    english_pool: list[str] = []

    for shard_idx in range(MAX_SHARDS):
        fname = f"data-{shard_idx:05d}-of-{TOTAL_SHARDS:05d}.arrow"
        print(f"  Fetching shard {shard_idx}: {fname} …", end=" ", flush=True)
        try:
            path = hf_hub_download(
                repo_id=DATASET_NAME,
                filename=fname,
                repo_type="dataset",
            )
        except Exception as exc:
            print(f"skipped ({exc})")
            continue

        texts = _iter_texts_from_arrow_shard(path)
        print(f"{len(texts)} rows", flush=True)

        for text in texts:
            text = (text or "").strip()
            if not text:
                continue
            if detect_language(text) == "hindi":
                if len(hindi_pool) < STOP_AT:
                    hindi_pool.append(text)
            elif english_sel_list is None:   # only collect if not using wikitext
                if len(english_pool) < STOP_AT:
                    english_pool.append(text)

        english_target_met = (
            (english_sel_list is not None and len(english_sel_list) >= TARGET_PER_LANG)
            or len(english_pool) >= TARGET_PER_LANG
        )
        if len(hindi_pool) >= TARGET_PER_LANG and english_target_met:
            break

    if not hindi_pool:
        raise SystemExit("Dataset returned no usable text rows.")

    rng = random.Random(seed)

    hindi_sel = rng.sample(hindi_pool, min(TARGET_PER_LANG, len(hindi_pool)))

    if english_sel_list is not None:
        english_sel = english_sel_list[:TARGET_PER_LANG]
    else:
        english_sel = rng.sample(english_pool, min(TARGET_PER_LANG, len(english_pool)))

    # Fill remainder from whichever pool has more
    remaining = n_samples - len(hindi_sel) - len(english_sel)
    if remaining > 0:
        extra_pool = [t for t in hindi_pool if t not in set(hindi_sel)]
        extra = rng.sample(extra_pool, min(remaining, len(extra_pool)))
        hindi_sel += extra
        remaining -= len(extra)
    if remaining > 0:
        extra_pool = [t for t in english_pool if t not in set(english_sel)]
        extra = rng.sample(extra_pool, min(remaining, len(extra_pool)))
        english_sel += extra

    all_texts = hindi_sel + english_sel
    all_langs = ["hindi"] * len(hindi_sel) + ["english"] * len(english_sel)

    combined = list(zip(all_texts, all_langs))
    rng.shuffle(combined)
    texts_out, langs_out = zip(*combined) if combined else ([], [])

    print(
        f"  Sampled {len(texts_out)} texts  "
        f"(hindi={sum(1 for l in langs_out if l=='hindi')}  "
        f"english={sum(1 for l in langs_out if l=='english')})",
        flush=True,
    )
    return list(texts_out), list(langs_out)


# ── Port discovery ────────────────────────────────────────────────────────────

def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def discover_ports(host: str, base_port: int, n_ports: int) -> list[int]:
    candidates = [base_port + i for i in range(n_ports)]
    return [p for p in candidates if _port_open(host, p)]


# ── WebSocket request ─────────────────────────────────────────────────────────

async def _synthesize_one(
    host: str,
    port: int,
    text: str,
    language: str,
    idx: int,
    timeout_s: float,
) -> dict:
    """
    Send one synthesis request and return a result dict with timing info.

    Result keys:
        idx, port, ok, language, text
        total_s     – wall time (send → receive last message)
        ttff_s      – time to first audio chunk (first token latency)
        llm_s       – LLM inference time (from server)
        decode_s    – decoder time (from server, if present)
        rtf         – real-time factor (if present)
        token_count – number of speech tokens
        wav_bytes   – concatenated raw WAV bytes (for ASR)
        error       – error message (if ok=False)
    """
    call_id = str(uuid.uuid4())
    text_id = str(uuid.uuid4())
    url = f"ws://{host}:{port}/ws/{call_id}"

    result: dict = {
        "idx":      idx,
        "port":     port,
        "ok":       False,
        "language": language,
        "text":     text,
        "text_id":  text_id,
    }

    t0 = time.perf_counter()
    ttff_s: Optional[float] = None      # time to first audio chunk
    wav_chunks: list[bytes] = []        # accumulate binary WAV frames

    try:
        async with websockets.connect(url, max_size=WS_MAX_SIZE, open_timeout=5) as ws:
            await ws.send(json.dumps({
                "type":    "synthesize",
                "call_id": call_id,
                "text_id": text_id,
                "text":    text,
            }))

            # The server streams:
            #   audio_chunk (JSON) → binary WAV bytes → ... → audio_done (JSON)
            # or for the no-decoder path:
            #   audio (JSON with audio_base64 or audio_tokens)
            # or:
            #   error (JSON)
            final_msg: dict = {}
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    result["error"]   = "timeout"
                    result["total_s"] = time.perf_counter() - t0
                    return result

                # Binary frame = raw WAV audio — collect for ASR, record TTFF
                if isinstance(raw, bytes):
                    if ttff_s is None:
                        ttff_s = time.perf_counter() - t0
                    wav_chunks.append(raw)
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError as exc:
                    result["error"] = f"JSON decode error: {exc}"
                    result["total_s"] = time.perf_counter() - t0
                    return result

                mtype = msg.get("type")

                if mtype == "error":
                    result["error"]   = msg.get("error", "unknown server error")
                    result["total_s"] = time.perf_counter() - t0
                    return result

                if mtype == "audio_done":
                    final_msg = msg
                    break

                if mtype == "audio":
                    # Non-streaming path: audio_base64 or audio_tokens
                    if ttff_s is None:
                        ttff_s = time.perf_counter() - t0
                    final_msg = msg
                    break

                # audio_chunk JSON metadata — keep reading (binary follows)

            total_s = time.perf_counter() - t0
            result["total_s"] = total_s
            result["ttff_s"]  = ttff_s

            result["ok"]       = True
            result["llm_s"]    = final_msg.get("llm_s")
            result["decode_s"] = final_msg.get("decode_s")
            result["rtf"]      = final_msg.get("rtf")

            # Token count
            if "total_tokens" in final_msg:
                result["token_count"] = final_msg["total_tokens"]
            elif "token_count" in final_msg:
                result["token_count"] = final_msg["token_count"]
            else:
                audio_tokens = final_msg.get("audio_tokens") or ""
                result["token_count"] = (
                    audio_tokens.count("<|speech_token_") if audio_tokens else 0
                )

            # Store concatenated WAV bytes for ASR (strips individual WAV headers
            # from chunk 1+ keeping only raw PCM; simplest: keep all bytes and
            # let _wav_bytes_to_float32 handle the first header)
            if wav_chunks:
                result["wav_bytes"] = b"".join(wav_chunks)
            elif "audio_base64" in final_msg:
                import base64
                result["wav_bytes"] = base64.b64decode(final_msg["audio_base64"])

    except (WebSocketException, OSError) as exc:
        result["error"]   = str(exc)
        result["total_s"] = time.perf_counter() - t0

    return result


# ── Concurrent runner ─────────────────────────────────────────────────────────

async def run_tests(
    host: str,
    ports: list[int],
    texts: list[str],
    languages: list[str],
    timeout_s: float,
) -> list[dict]:
    """
    Send all requests concurrently, round-robining across available ports.
    """
    tasks = []
    for idx, (text, lang) in enumerate(zip(texts, languages)):
        port = ports[idx % len(ports)]
        tasks.append(
            _synthesize_one(host, port, text, lang, idx, timeout_s)
        )

    print(
        f"\n  Sending {len(tasks)} requests concurrently across {len(ports)} port(s) …",
        flush=True,
    )
    results = await asyncio.gather(*tasks)
    return list(results)


# ── ASR transcription pass ────────────────────────────────────────────────────

def run_asr_transcription(
    results: list[dict],
    asr: "WhisperASR",
) -> None:
    """
    For each successful result that has wav_bytes, transcribe with Whisper
    and compute CER against the original text.  Mutates results in-place,
    adding 'asr_transcript' and 'cer' keys.
    """
    to_transcribe = [r for r in results if r.get("ok") and r.get("wav_bytes")]
    if not to_transcribe:
        print("  No WAV audio collected — skipping ASR.", flush=True)
        return

    print(f"\n  Running ASR on {len(to_transcribe)} samples …", flush=True)
    t0 = time.time()

    for i, r in enumerate(to_transcribe):
        audio = _wav_bytes_to_float32(r["wav_bytes"])
        transcript = asr.transcribe(audio, language=r["language"])
        cer = compute_cer(r["text"], transcript)
        r["asr_transcript"] = transcript
        r["cer"] = round(cer, 4)
        if (i + 1) % 10 == 0 or (i + 1) == len(to_transcribe):
            print(f"    {i+1}/{len(to_transcribe)} done", flush=True)

    print(f"  ASR complete in {time.time()-t0:.1f}s", flush=True)


# ── Statistics ────────────────────────────────────────────────────────────────

def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _latency_stats(vals_s: list[float]) -> dict:
    if not vals_s:
        return {}
    ms = sorted(v * 1000 for v in vals_s)
    return {
        "avg_ms":  round(statistics.mean(ms),  1),
        "min_ms":  round(min(ms),              1),
        "max_ms":  round(max(ms),              1),
        "std_ms":  round(statistics.stdev(ms) if len(ms) > 1 else 0.0, 1),
        "p50_ms":  round(_percentile(ms, 50),  1),
        "p95_ms":  round(_percentile(ms, 95),  1),
        "p99_ms":  round(_percentile(ms, 99),  1),
    }


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute per-language and overall metrics from a list of result dicts.

    Returns a nested dict:
        {
          "overall": { ... },
          "hindi":   { ... },
          "english": { ... },
        }
    Each sub-dict contains:
        total, errors, error_rate,
        latency_total, latency_llm, latency_decode,
        rtf (if available),
        avg_tokens, throughput_rps,
        errors_detail: list of {idx, text, error}
    """
    groups = {
        "overall": results,
        "hindi":   [r for r in results if r.get("language") == "hindi"],
        "english": [r for r in results if r.get("language") == "english"],
    }

    report = {}
    for label, group in groups.items():
        if not group:
            report[label] = {"total": 0, "note": "no samples"}
            continue

        ok_results  = [r for r in group if r.get("ok")]
        err_results = [r for r in group if not r.get("ok")]

        error_rate = len(err_results) / len(group)

        # Latency metrics (only from successful requests)
        total_s_vals  = [r["total_s"]  for r in ok_results if r.get("total_s")  is not None]
        ttff_s_vals   = [r["ttff_s"]   for r in ok_results if r.get("ttff_s")   is not None]
        llm_s_vals    = [r["llm_s"]    for r in ok_results if r.get("llm_s")    is not None]
        decode_s_vals = [r["decode_s"] for r in ok_results if r.get("decode_s") is not None]
        rtf_vals      = [r["rtf"]      for r in ok_results if r.get("rtf")      is not None]
        token_vals    = [r.get("token_count", 0) for r in ok_results]
        cer_vals      = [r["cer"]      for r in ok_results if r.get("cer")      is not None]

        # Throughput: total successful reqs / (max wall time across the group)
        wall_times = [r["total_s"] for r in ok_results if r.get("total_s") is not None]
        throughput  = len(ok_results) / max(wall_times) if wall_times else 0.0

        entry: dict = {
            "total":       len(group),
            "ok":          len(ok_results),
            "errors":      len(err_results),
            "error_rate":  round(error_rate, 4),
            "error_pct":   f"{error_rate * 100:.1f}%",
        }

        entry["latency_total_ms"]  = _latency_stats(total_s_vals)
        entry["latency_ttff_ms"]   = _latency_stats(ttff_s_vals)
        entry["latency_llm_ms"]    = _latency_stats(llm_s_vals)

        if decode_s_vals:
            entry["latency_decode_ms"] = _latency_stats(decode_s_vals)

        if rtf_vals:
            entry["rtf"] = {
                "avg": round(statistics.mean(rtf_vals),  3),
                "min": round(min(rtf_vals),              3),
                "max": round(max(rtf_vals),              3),
                "below_1_0_pct": f"{sum(1 for v in rtf_vals if v < 1.0) / len(rtf_vals) * 100:.1f}%",
            }

        if cer_vals:
            entry["asr_cer"] = {
                "avg":    round(statistics.mean(cer_vals), 4),
                "median": round(_percentile(sorted(cer_vals), 50), 4),
                "min":    round(min(cer_vals), 4),
                "max":    round(max(cer_vals), 4),
                "pct":    f"{statistics.mean(cer_vals) * 100:.1f}%",
                "n":      len(cer_vals),
            }

        entry["avg_tokens"]      = round(statistics.mean(token_vals), 1) if token_vals else 0
        entry["throughput_rps"]  = round(throughput, 2)

        entry["errors_detail"] = [
            {
                "idx":   r["idx"],
                "lang":  r.get("language"),
                "error": r.get("error", "?"),
                "text":  (r.get("text", "")[:60] + "…") if len(r.get("text", "")) > 60 else r.get("text", ""),
            }
            for r in err_results
        ]

        report[label] = entry

    return report


# ── Pretty printer ────────────────────────────────────────────────────────────

def _fmt_lat(stats: dict) -> str:
    if not stats:
        return "(no data)"
    return (
        f"avg={stats['avg_ms']:.0f}ms  "
        f"p50={stats['p50_ms']:.0f}ms  "
        f"p95={stats['p95_ms']:.0f}ms  "
        f"p99={stats['p99_ms']:.0f}ms  "
        f"min={stats['min_ms']:.0f}ms  "
        f"max={stats['max_ms']:.0f}ms"
    )


def print_report(report: dict) -> None:
    width = 76
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "═" * width)
    print(f"  FlowTTS  Hindi / English Benchmark  —  {now}")
    print("═" * width)

    for label in ("hindi", "english", "overall"):
        data = report.get(label, {})
        if not data or data.get("total", 0) == 0:
            print(f"\n  [{label.upper()}]  no samples")
            continue

        print(f"\n  [{label.upper()}]")
        print(f"  {'─'*60}")
        print(f"  Samples   : {data['total']}  ok={data['ok']}  errors={data['errors']}  "
              f"error_rate={data['error_pct']}")
        print(f"  Tokens    : avg {data['avg_tokens']:.0f} / request")
        print(f"  Throughput: {data['throughput_rps']} req/s")
        print(f"  Latency (total wall)  : {_fmt_lat(data.get('latency_total_ms', {}))}")
        print(f"  Latency (TTFF)        : {_fmt_lat(data.get('latency_ttff_ms', {}))}")
        print(f"  Latency (LLM only)    : {_fmt_lat(data.get('latency_llm_ms', {}))}")

        if "latency_decode_ms" in data:
            print(f"  Latency (decoder)     : {_fmt_lat(data['latency_decode_ms'])}")

        if "rtf" in data:
            r = data["rtf"]
            print(f"  RTF                   : avg={r['avg']}  min={r['min']}  max={r['max']}  "
                  f"<1.0={r['below_1_0_pct']}")

        if "asr_cer" in data:
            c = data["asr_cer"]
            print(f"  ASR CER               : avg={c['pct']}  median={c['median']*100:.1f}%  "
                  f"min={c['min']*100:.1f}%  max={c['max']*100:.1f}%  (n={c['n']})")

        if data["errors_detail"]:
            print(f"\n  Errors ({min(len(data['errors_detail']), 5)} shown):")
            for e in data["errors_detail"][:5]:
                print(f"    [idx={e['idx']}] {e['error']}  text={e['text']!r}")
            if len(data["errors_detail"]) > 5:
                print(f"    … and {len(data['errors_detail'])-5} more")

    print("\n" + "═" * width)


# ── Report saving ─────────────────────────────────────────────────────────────

def save_report(report: dict, raw_results: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag  = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"hindi_english_metrics_{tag}.json"
    path.write_text(
        json.dumps(
            {"summary": report, "raw_results": raw_results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  Report saved → {path}", flush=True)
    return path


# ── Dry-run (no server) ───────────────────────────────────────────────────────

def dry_run(texts: list[str], languages: list[str]) -> None:
    """Print language classification results without hitting the server."""
    hindi   = [t for t, l in zip(texts, languages) if l == "hindi"]
    english = [t for t, l in zip(texts, languages) if l == "english"]

    print(f"\n  DRY RUN — {len(texts)} samples classified (no server)")
    print(f"  Hindi  : {len(hindi)}")
    for t in hindi[:5]:
        print(f"    {t[:80]}")
    if len(hindi) > 5:
        print(f"    … and {len(hindi)-5} more")
    print(f"  English: {len(english)}")
    for t in english[:5]:
        print(f"    {t[:80]}")
    if len(english) > 5:
        print(f"    … and {len(english)-5} more")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "FlowTTS Hindi/English error-rate & latency benchmark\n"
            f"Dataset: {DATASET_NAME}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host", default="localhost",
        help="FlowTTS server host (default: localhost)",
    )
    parser.add_argument(
        "--base-port", type=int, default=DEFAULT_BASE_PORT,
        help=f"Base WebSocket port to scan from (default: {DEFAULT_BASE_PORT})",
    )
    parser.add_argument(
        "--n-ports", type=int, default=DEFAULT_N_PORTS,
        help=f"Number of consecutive ports to scan (default: {DEFAULT_N_PORTS})",
    )
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_SAMPLES,
        help=f"Total samples to draw from dataset (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--hf-split", default=DEFAULT_HF_SPLIT,
        help=f"HuggingFace dataset split (default: {DEFAULT_HF_SPLIT})",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sample selection (default: 42)",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="(deprecated, now always text-only — audio loading is skipped to avoid OOM)",
    )
    parser.add_argument(
        "--pure-english", action="store_true",
        help=(
            "Replace the English half with pure English sentences from "
            f"{ENGLISH_DATASET_NAME} instead of Hinglish from the Hindi dataset"
        ),
    )
    parser.add_argument(
        "--no-asr", action="store_true",
        help="Skip ASR back-transcription and CER computation (faster, no Whisper GPU load)",
    )
    parser.add_argument(
        "--asr-model", default=ASR_MODEL_HINDI,
        help=(
            f"Whisper model ID to use for Hindi ASR "
            f"(default: {ASR_MODEL_HINDI}). "
            f"English always uses {ASR_MODEL_ENGLISH}."
        ),
    )
    parser.add_argument(
        "--save-report", action="store_true",
        help="Save full JSON report to flowtts/test/reports/",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Classify dataset samples and print them, but don't hit the server",
    )
    args = parser.parse_args()

    # 1. Load dataset samples
    texts, languages = load_samples(
        n_samples=args.samples,
        hf_split=args.hf_split,
        load_audio=not args.no_audio,
        seed=args.seed,
        pure_english=args.pure_english,
    )

    if not texts:
        print("No samples loaded. Exiting.")
        return

    # 2. Dry-run shortcut
    if args.dry_run:
        dry_run(texts, languages)
        return

    # 3. Discover live ports
    print(
        f"\n  Scanning {args.base_port}..{args.base_port + args.n_ports - 1} on {args.host} …",
        flush=True,
    )
    ports = discover_ports(args.host, args.base_port, args.n_ports)

    if not ports:
        print(
            "  No live FlowTTS ports found. Is the server running?\n"
            "  Start it with:  python -m flowtts.server\n"
        )
        return

    print(f"  Live ports ({len(ports)}): {ports}", flush=True)

    # 4. Run TTS synthesis tests
    t_wall0 = time.perf_counter()
    raw_results = asyncio.run(
        run_tests(args.host, ports, texts, languages, args.timeout)
    )
    wall_total = time.perf_counter() - t_wall0

    ok_count = sum(1 for r in raw_results if r.get("ok"))
    print(
        f"\n  Synthesis done: {ok_count}/{len(raw_results)} ok  "
        f"wall={wall_total:.2f}s  throughput={ok_count/wall_total:.2f} req/s",
        flush=True,
    )

    # 5. ASR back-transcription → CER (mutates raw_results in-place)
    if not args.no_asr:
        asr = WhisperASR(
            hindi_model_id=args.asr_model if args.asr_model != ASR_MODEL_ENGLISH else ASR_MODEL_HINDI,
            english_model_id=ASR_MODEL_ENGLISH,
        )
        run_asr_transcription(raw_results, asr)
    else:
        print("\n  ASR skipped (--no-asr)", flush=True)

    # 6. Compute & print metrics
    report = compute_metrics(raw_results)
    print_report(report)

    # 7. Optionally save JSON report
    if args.save_report:
        out_dir = Path(__file__).parent / "reports"
        # Drop raw wav_bytes from saved report (too large)
        for r in raw_results:
            r.pop("wav_bytes", None)
        save_report(report, raw_results, out_dir)


if __name__ == "__main__":
    main()
