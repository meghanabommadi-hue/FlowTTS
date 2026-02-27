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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import structlog

# One JSON line per completed call — written by server.py via record_call().
_CALLS_LOG = Path(__file__).parents[2] / "monitoring" / "calls.jsonl"
_CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
_calls_log_file = _CALLS_LOG.open("a", buffering=1)  # line-buffered, append across restarts


logger = structlog.get_logger("metrics")


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


def record_ws_connection_open(call_id: str) -> None:
    """Increment count of WebSocket connections opened."""
    global _ws_connections_opened
    with _lock:
        _ws_connections_opened += 1
    logger.info("ws_connection_open", call_id=call_id)


def record_ws_connection_close(call_id: str) -> None:
    """Increment count of WebSocket connections closed."""
    global _ws_connections_closed
    with _lock:
        _ws_connections_closed += 1
    logger.info("ws_connection_close", call_id=call_id)


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
) -> None:
    """Append one JSON line to monitoring/calls.jsonl for every completed TTS call."""
    entry = {
        "ts": ts,
        "call_id": call_id,
        "text_id": text_id,
        "port": port,
        "text": text,
        "token_count": token_count,
        "llm_s": llm_s,
        "decode_s": decode_s,
        "total_s": round(llm_s + decode_s, 4),
        "wav_bytes": wav_bytes,
    }
    with _lock:
        _calls_log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


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

