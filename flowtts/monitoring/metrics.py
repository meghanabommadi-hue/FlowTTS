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

import json
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Deque

import structlog
from prometheus_client import Counter, Gauge
from prometheus_client import disable_created_metrics
disable_created_metrics()

# One JSON line per completed call — written by server.py via record_call().
_CALLS_LOG = Path(__file__).parents[2] / "monitoring" / "calls.jsonl"
_CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
_calls_log_file = _CALLS_LOG.open("a", buffering=1)  # line-buffered, append across restarts


logger = structlog.get_logger("metrics")

# ── Prometheus counters (matching PikoTTS naming) ─────────────────────────────
TTS_REQUESTS      = Counter('tts_requests_total',      'Total successful TTS requests')
TTS_LLM_MS        = Counter('tts_llm_ms_total',        'Total sum of LLM inference time in ms')
TTS_DECODE_MS     = Counter('tts_decode_ms_total',     'Total sum of decoder time in ms')
TTS_E2E_MS        = Counter('tts_e2e_ms_total',        'Total sum of end-to-end time in ms')
TTS_TOKENS        = Counter('tts_tokens_total',        'Total sum of generated speech tokens')
ACTIVE_WEBSOCKETS = Gauge('tts_active_websockets',     'Currently active WebSocket connections')

# WebSocket lifetime counters
WS_CONNECTIONS_OPENED = Counter('tts_ws_connections_opened_total', 'Total WebSocket connections opened')
WS_CONNECTIONS_CLOSED = Counter('tts_ws_connections_closed_total', 'Total WebSocket connections closed')

# Port tracking
OPEN_PORTS = Gauge('tts_open_ports', 'Currently open WebSocket ports')
MAX_PORTS  = Gauge('tts_max_ports',  'Maximum WebSocket ports ever open simultaneously')

# Error counter
TTS_ERRORS = Counter('tts_errors_total', 'Total failed TTS requests')

# Clean disconnect counter (WebSocket close code 1000 = normal closure)
WS_CLEAN_DISCONNECT = Counter('tts_ws_clean_disconnect_total',
                              'WebSocket disconnects via ERROR 1000 (OK) clean close')


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

# Ring buffer of the last N WS events — viewable via GET /ws/log
_WS_LOG_MAX = 20
_ws_log: Deque[dict] = deque(maxlen=_WS_LOG_MAX)


def _ws_log_append(event: dict) -> None:
    import datetime as _dt
    event.setdefault("ts", _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3])
    with _lock:
        _ws_log.append(event)


def ws_log_snapshot() -> list:
    """Return a copy of the WS event ring buffer (oldest → newest)."""
    with _lock:
        return list(_ws_log)


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
        active = _ws_connections_opened - _ws_connections_closed
    ACTIVE_WEBSOCKETS.inc()
    WS_CONNECTIONS_OPENED.inc()
    _ws_log_append({"event": "open", "call_id": call_id, "port": port, "active_ws": active})
    logger.info("ws_connection_open", call_id=call_id)


def record_ws_connection_close(call_id: str, *, port: int = 0) -> None:
    """Increment count of WebSocket connections closed."""
    global _ws_connections_closed
    with _lock:
        _ws_connections_closed += 1
        active = _ws_connections_opened - _ws_connections_closed
    ACTIVE_WEBSOCKETS.dec()
    WS_CONNECTIONS_CLOSED.inc()
    _ws_log_append({"event": "close", "call_id": call_id, "port": port, "active_ws": active})
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


def record_ws_error(call_id: str, *, port: int = 0, text_id: str = "", error: str = "") -> None:
    """Record a WS request that ended in an error."""
    if "1000" in error and "(OK)" in error:
        WS_CLEAN_DISCONNECT.inc()
    else:
        TTS_ERRORS.inc()
    _ws_log_append({
        "event": "error",
        "call_id": call_id,
        "text_id": text_id,
        "port": port,
        "error": error,
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
        "total_s": round(llm_s + decode_s, 4),
        "wav_bytes": wav_bytes,
        "cache_hit": cache_hit,
    }
    with _lock:
        _calls_log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    TTS_REQUESTS.inc()
    TTS_LLM_MS.inc(llm_s * 1000)
    TTS_DECODE_MS.inc(decode_s * 1000)
    TTS_E2E_MS.inc((llm_s + decode_s) * 1000)
    TTS_TOKENS.inc(token_count)


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
        }

    return {"synthesis_latency": synthesis, "decode_latency": decode, "ws": ws}

