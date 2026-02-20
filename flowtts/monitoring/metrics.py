"""In-process metrics helpers for FlowTTS.

These functions provide a thin abstraction that can later be backed by
Prometheus/StatsD. For now they keep simple in-memory counters and emit
structured logs so you can debug performance without extra infra.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import structlog


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

