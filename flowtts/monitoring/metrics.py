"""Pipeline position: OBSERVABILITY — latency metrics for every stage.

Role in pipeline:
  Thin in-memory counters instrumented at two points in the pipeline:

  Worker (worker.py):
    record_synthesis_latency(call_id, text_id, duration_s)
      → tracks time from synthesis_service.synthesize() start to finish
        (= pure sglang GPU inference time, excludes queue wait)

  Gateway (api/websockets.py):
    record_decode_latency(call_id, duration_s)
      → tracks time from ncodec decode start to finish
        (= AudioDecoder.decode_to_wav(), excludes WAV encoding when to_wav=False)
    record_ws_connection_open / close
      → tracks concurrent call count

  All metrics also emit a structlog event so they appear in the log stream
  without needing a separate metrics server.

Scaling note:
  Counters are in-process only — not shared across gateway workers. To
  aggregate across processes, back these with Prometheus/StatsD by replacing
  TimingStat.observe() with a push to an external system.
"""

from __future__ import annotations

import datetime
import json
import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Deque

import structlog
from prometheus_client import Counter, Gauge, Info, REGISTRY
from prometheus_client import disable_created_metrics
disable_created_metrics()

# One JSON line per completed call — written by server.py via record_call().
_CALLS_LOG = Path(__file__).parents[2] / "monitoring" / "calls.jsonl"
_CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
_calls_log_file = _CALLS_LOG.open("a", buffering=1)  # line-buffered, append across restarts


logger = structlog.get_logger("metrics")

# ── Prometheus counters — labeled by gpu_id + voice for multi-GPU visibility ──
_LABELS = ["gpu_id", "voice"]

TTS_REQUESTS  = Counter('tts_requests_total',   'Total successful TTS requests',         _LABELS)
TTS_LLM_MS    = Counter('tts_llm_ms_total',     'Total sum of LLM inference time in ms', _LABELS)
TTS_DECODE_MS = Counter('tts_decode_ms_total',  'Total sum of decoder time in ms',       _LABELS)
TTS_E2E_MS    = Counter('tts_e2e_ms_total',     'Total sum of end-to-end time in ms',    _LABELS)
TTS_TOKENS    = Counter('tts_tokens_total',     'Total sum of generated speech tokens',  _LABELS)
TTS_ERRORS    = Counter('tts_errors_total',     'Total failed TTS requests',             _LABELS)
TTS_CACHE_HITS   = Counter('tts_cache_hits_total',   'Total TTS requests served from WAV cache', _LABELS)
TTS_CACHE_MISSES = Counter('tts_cache_misses_total', 'Total TTS requests that bypassed cache',   _LABELS)

# Short-audio alarm: fired when generated audio is suspiciously brief relative to input text.
# Labels: gpu_id, voice, reason  (reason = "truncated" | "short_text_ok" not fired for the latter)
TTS_SHORT_AUDIO = Counter(
    'tts_short_audio_total',
    'TTS requests where generated audio duration is suspiciously short for the input text length',
    ['gpu_id', 'voice'],
)

# Active connections labeled by gpu_id so multi-GPU deployments are visible
ACTIVE_WEBSOCKETS = Gauge('tts_active_websockets', 'Currently active WebSocket connections',
                          ['gpu_id'])

# Per-call_id active connection gauge — 1 while open, removed on close
WS_ACTIVE_CALL = Gauge('tts_ws_active_call', 'Active WebSocket connection by call_id',
                       ['call_id', 'gpu_id'])

# WebSocket lifetime counters (unlabeled — aggregate totals are enough here)
WS_CONNECTIONS_OPENED = Counter('tts_ws_connections_opened_total', 'Total WebSocket connections opened')
WS_CONNECTIONS_CLOSED = Counter('tts_ws_connections_closed_total', 'Total WebSocket connections closed')

# Port tracking
OPEN_PORTS = Gauge('tts_open_ports', 'Currently open WebSocket ports')
MAX_PORTS  = Gauge('tts_max_ports',  'Maximum WebSocket ports ever open simultaneously')

# Clean disconnect counter (WebSocket close code 1000 = normal closure)
WS_CLEAN_DISCONNECT = Counter('tts_ws_clean_disconnect_total',
                              'WebSocket disconnects via ERROR 1000 (OK) clean close')

# Static engine/GPU info — set once at startup via register_gpu_info()
# Exposes: gpu_id, tp_size, attention_backend, model_gpu_id, decoder_gpu_id
TTS_ENGINE_INFO = Info('tts_engine', 'FlowTTS engine and GPU configuration')

# TTFT Gauges — rolling averages parsed from llm.log stream_done lines
# llm_ttft  = time from request received to first LLM token (pure model latency)
# dec_ttft  = time from request received to first decoded audio chunk sent to client
TTS_LLM_TTFT_AVG_MS = Gauge('tts_llm_ttft_avg_ms',
                             'Rolling average LLM time-to-first-token (ms), from llm.log')
TTS_DEC_TTFT_AVG_MS = Gauge('tts_dec_ttft_avg_ms',
                             'Rolling average decoder time-to-first-chunk (ms), from llm.log')
TTS_TOTAL_AVG_MS    = Gauge('tts_total_avg_ms',
                             'Rolling average total E2E time (ms) including wav_enc, from llm.log')

# Log path — same file server.py writes stream_done lines to
_LLM_LOG = Path(__file__).parents[2] / "llm.log"

# Rolling window for TTFT stats (last N stream_done lines parsed)
_TTFT_WINDOW = 100
_llm_ttft_buf:   deque = deque(maxlen=_TTFT_WINDOW)
_dec_ttft_buf:   deque = deque(maxlen=_TTFT_WINDOW)
_total_ms_buf:   deque = deque(maxlen=_TTFT_WINDOW)
_log_file_pos:   int   = 0   # byte offset — only read new lines on each call

# Regex matching: stream_done lines with llm_ttft/dec_ttft/total fields
_RE_STREAM_DONE = re.compile(
    r'stream_done.*?llm_ttft=(\d+)ms.*?dec_ttft=(\d+)ms.*?total=(\d+)ms'
)


def refresh_ttft_from_log() -> None:
    """Tail llm.log for new stream_done lines and update TTFT gauges.

    Reads only new bytes since last call (tail-follow style).  Safe to call
    frequently — cheap when there are no new lines.
    """
    global _log_file_pos
    if not _LLM_LOG.exists():
        return
    try:
        with _LLM_LOG.open("r", errors="replace") as f:
            f.seek(_log_file_pos)
            new_data = f.read()
            _log_file_pos = f.tell()
    except OSError:
        return

    for line in new_data.splitlines():
        m = _RE_STREAM_DONE.search(line)
        if m:
            _llm_ttft_buf.append(int(m.group(1)))
            _dec_ttft_buf.append(int(m.group(2)))
            _total_ms_buf.append(int(m.group(3)))

    if _llm_ttft_buf:
        TTS_LLM_TTFT_AVG_MS.set(sum(_llm_ttft_buf) / len(_llm_ttft_buf))
    if _dec_ttft_buf:
        TTS_DEC_TTFT_AVG_MS.set(sum(_dec_ttft_buf) / len(_dec_ttft_buf))
    if _total_ms_buf:
        TTS_TOTAL_AVG_MS.set(sum(_total_ms_buf) / len(_total_ms_buf))


class _TtftLogCollector:
    """Prometheus collector that tails llm.log on every scrape to keep TTFT gauges fresh."""
    def describe(self):
        return []   # gauges are already registered; no extra descriptors needed

    def collect(self):
        refresh_ttft_from_log()
        return []   # actual metric values emitted by the registered Gauges above


REGISTRY.register(_TtftLogCollector())


def ttft_snapshot() -> dict:
    """Return TTFT stats dict for use by the dashboard."""
    def _s(buf: deque) -> dict | None:
        if not buf:
            return None
        vals = sorted(buf)
        n = len(vals)
        return {
            "min":  vals[0],
            "avg":  round(sum(vals) / n),
            "p95":  vals[int(n * 0.95)],
            "max":  vals[-1],
            "n":    n,
        }
    return {
        "llm_ttft_ms": _s(_llm_ttft_buf),
        "dec_ttft_ms": _s(_dec_ttft_buf),
        "total_ms":    _s(_total_ms_buf),
    }


@dataclass
class TimingStat:
    count: int = 0
    total: float = 0.0
    max_value: float = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        if value > self.max_value:
            self.max_value = value

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


_lock = threading.Lock()
_synthesis_latency: Dict[str, TimingStat] = defaultdict(TimingStat)  # text_id → stat
_decode_latency: TimingStat = TimingStat()
_ws_connections_opened: int = 0
_ws_connections_closed: int = 0
_active_ws_ids: set = set()  # tracks open conn_ids; gauge derived from len()
_cache_hits: int = 0
_cache_misses: int = 0

# Active GPU id — set at startup by register_gpu_info(); used as default label value
_gpu_id: str = "0"

# Ring buffer of the last N WS events — viewable via GET /ws/log
_WS_LOG_MAX = 20
_ws_log: Deque[dict] = deque(maxlen=_WS_LOG_MAX)


def _ws_log_append(event: dict) -> None:
    event.setdefault("ts", datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3])
    with _lock:
        _ws_log.append(event)


def ws_log_snapshot() -> list:
    """Return a copy of the WS event ring buffer (oldest → newest)."""
    with _lock:
        return list(_ws_log)


def register_gpu_info(
    *,
    model_gpu_id: int = 0,
    decoder_gpu_id: int = 0,
    tp_size: int = 1,
    attention_backend: str = "n/a",
    mem_weight_gb: str = "n/a",
    mem_kvcache_gb: str = "n/a",
) -> None:
    """Set static engine/GPU info in Prometheus. Call once after model load."""
    global _gpu_id
    _gpu_id = str(model_gpu_id)
    TTS_ENGINE_INFO.info({
        "model_gpu_id":      str(model_gpu_id),
        "decoder_gpu_id":    str(decoder_gpu_id),
        "tp_size":           str(tp_size),
        "attention_backend": attention_backend,
        "mem_weight_gb":     str(mem_weight_gb),
        "mem_kvcache_gb":    str(mem_kvcache_gb),
    })


def record_synthesis_latency(call_id: str, text_id: str, duration_seconds: float) -> None:
    """Record time spent in the TTS model (text → audio tokens)."""
    with _lock:
        stat = _synthesis_latency[text_id]
        stat.observe(duration_seconds)

    logger.info(
        "synthesis_latency",
        call_id=call_id,
        text_id=text_id,
        duration_seconds=duration_seconds,
    )


def record_decode_latency(call_id: str, duration_seconds: float) -> None:
    """Record time spent decoding tokens → PCM (plus processing)."""
    with _lock:
        _decode_latency.observe(duration_seconds)

    logger.info(
        "decode_latency",
        call_id=call_id,
        duration_seconds=duration_seconds,
    )


def record_ws_connection_open(call_id: str, *, port: int = 0) -> None:
    """Increment count of WebSocket connections opened."""
    global _ws_connections_opened
    with _lock:
        _ws_connections_opened += 1
        _active_ws_ids.add(call_id)
        active = len(_active_ws_ids)
    ACTIVE_WEBSOCKETS.labels(gpu_id=_gpu_id).set(active)
    WS_ACTIVE_CALL.labels(call_id=call_id, gpu_id=_gpu_id).set(1)
    WS_CONNECTIONS_OPENED.inc()
    _ws_log_append({"event": "open", "call_id": call_id, "port": port, "active_ws": active, "gpu_id": _gpu_id})
    logger.info("ws_connection_open", call_id=call_id)


def record_ws_connection_close(call_id: str, *, port: int = 0) -> None:
    """Increment count of WebSocket connections closed."""
    global _ws_connections_closed
    with _lock:
        already_closed = call_id not in _active_ws_ids
        _active_ws_ids.discard(call_id)
        active = len(_active_ws_ids)
        if not already_closed:
            _ws_connections_closed += 1
    ACTIVE_WEBSOCKETS.labels(gpu_id=_gpu_id).set(active)
    if already_closed:
        logger.warning("ws_connection_close_already_closed", call_id=call_id, port=port)
    else:
        WS_CONNECTIONS_CLOSED.inc()
        WS_ACTIVE_CALL.remove(call_id, _gpu_id)
    _ws_log_append({"event": "close", "call_id": call_id, "port": port, "active_ws": active, "gpu_id": _gpu_id,
                    "already_closed": already_closed})
    logger.info("ws_connection_close", call_id=call_id)


def record_ws_done(
    call_id: str,
    *,
    port: int = 0,
    text_id: str = "",
    token_count: int = 0,
    llm_ms: int = 0,
    decode_ms: int = 0,
    total_ms: int = 0,
    wav_bytes: int = 0,
    ts_text_recv: str = "",
    ts_llm_start: str = "",
    ts_tokens_ready: str = "",
    ts_audio_sent: str = "",
) -> None:
    """Record a successfully completed WS request with per-milestone timestamps."""
    _ws_log_append({
        "event": "done",
        "call_id": call_id,
        "text_id": text_id,
        "port": port,
        "token_count": token_count,
        "llm_ms": llm_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "wav_bytes": wav_bytes,
        "ts_text_recv": ts_text_recv,
        "ts_llm_start": ts_llm_start,
        "ts_tokens_ready": ts_tokens_ready,
        "ts_audio_sent": ts_audio_sent,
    })


def record_ws_error(call_id: str, *, port: int = 0, text_id: str = "", error: str = "",
                    voice_id: str | None = None) -> None:
    """Record a WS request that ended in an error."""
    if "1000" in error and "(OK)" in error:
        WS_CLEAN_DISCONNECT.inc()
    else:
        TTS_ERRORS.labels(gpu_id=_gpu_id, voice=voice_id or "").inc()
    _ws_log_append({
        "event": "error",
        "call_id": call_id,
        "text_id": text_id,
        "port": port,
        "error": error,
        "gpu_id": _gpu_id,
    })


def record_call(
    *,
    call_id: str,
    text_id: str,
    port: int,
    text: str,
    token_count: int,
    llm_s: float,
    decode_s: float,
    wav_bytes: int,
    ts: str,
    voice_id: str | None = None,
    cache_hit: bool = False,
) -> None:
    """Append one JSON line to monitoring/calls.jsonl for every completed TTS call."""
    total_s = round(llm_s + decode_s, 4)
    entry = {
        "ts": ts,
        "call_id": call_id,
        "text_id": text_id,
        "port": port,
        "text": text,
        "voice_id": voice_id,
        "token_count": token_count,
        "llm_s": llm_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "wav_bytes": wav_bytes,
        "cache_hit": cache_hit,
    }
    with _lock:
        _calls_log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    _v = voice_id or ""
    TTS_REQUESTS.labels(gpu_id=_gpu_id,  voice=_v).inc()
    TTS_LLM_MS.labels(gpu_id=_gpu_id,    voice=_v).inc(llm_s * 1000)
    TTS_DECODE_MS.labels(gpu_id=_gpu_id, voice=_v).inc(decode_s * 1000)
    TTS_E2E_MS.labels(gpu_id=_gpu_id,    voice=_v).inc(total_s * 1000)
    TTS_TOKENS.labels(gpu_id=_gpu_id,    voice=_v).inc(token_count)
    global _cache_hits, _cache_misses
    if cache_hit:
        TTS_CACHE_HITS.labels(gpu_id=_gpu_id, voice=_v).inc()
        with _lock:
            _cache_hits += 1
    else:
        TTS_CACHE_MISSES.labels(gpu_id=_gpu_id, voice=_v).inc()
        with _lock:
            _cache_misses += 1

    # ── Short-audio alarm ─────────────────────────────────────────────────────
    # Cache hits always have token_count=0 (audio served from file, not LLM).
    # Skip the alarm for cache hits to avoid false positives.
    if cache_hit:
        return
    # 1 token = 320 samples @ 16 kHz → 0.02 s/token
    audio_s = token_count * 320 / 16000
    # Expected minimum: ~15 chars/s for mixed Hindi/Telugu script (conservative).
    # Only fire if the text is long enough that we'd expect >1.5 s of audio.
    text_chars = len(text.strip())
    expected_s = text_chars / 15.0
    if expected_s > 1.5 and audio_s < 0.8:
        TTS_SHORT_AUDIO.labels(gpu_id=_gpu_id, voice=_v).inc()
        logger.warning(
            "short_audio_alarm",
            call_id=call_id,
            text_id=text_id,
            audio_s=round(audio_s, 3),
            expected_s=round(expected_s, 3),
            token_count=token_count,
            text_chars=text_chars,
            voice=_v,
        )


def record_port_change(open_ports: set) -> None:
    """Update port gauges whenever a WS port is opened. Call from server.py."""
    n = len(open_ports)
    OPEN_PORTS.set(n)
    if n > MAX_PORTS._value.get():
        MAX_PORTS.set(n)


def snapshot_metrics() -> dict:
    """Return a lightweight snapshot of current in-memory metrics."""
    with _lock:
        synthesis = {
            text_id: {
                "count": stat.count,
                "avg": stat.avg,
                "max": stat.max_value,
            }
            for text_id, stat in _synthesis_latency.items()
        }
        decode = {
            "count": _decode_latency.count,
            "avg": _decode_latency.avg,
            "max": _decode_latency.max_value,
        }
        ws = {
            "opened": _ws_connections_opened,
            "closed": _ws_connections_closed,
            "active": len(_active_ws_ids),
            "active_ids": list(_active_ws_ids),
        }
        total_calls = _cache_hits + _cache_misses
        cache = {
            "hits":      _cache_hits,
            "misses":    _cache_misses,
            "hit_rate":  round(_cache_hits / total_calls, 4) if total_calls else 0.0,
        }

    return {
        "ts": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "synthesis_latency": synthesis,
        "decode_latency": decode,
        "ws": ws,
        "cache": cache,
    }

