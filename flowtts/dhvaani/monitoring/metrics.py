"""Pipeline position: OBSERVABILITY — Prometheus metrics for the DhVaani path.

Role in pipeline:
  Called once per completed request by `engine/engine.py`, plus gauges the
  scheduler and watchdog update continuously.

Namespacing
-----------
Every series here is `dhvaani_*`. The legacy MiraTTS path in
`flowtts/monitoring/metrics.py` owns `tts_*`, and both modules can be imported
into the same process (the WebSocket gateway reuses the legacy recorders so
existing dashboards keep working), so the two name spaces must not overlap.
`_counter`/`_gauge`/`_histogram` below tolerate a duplicate registration rather
than raising, which is what would otherwise happen on a module reload.

Histogram buckets
-----------------
Prometheus' defaults top out at 10 seconds and have their finest resolution
around 5-100 ms in *seconds* units. We record milliseconds and care about the
25-500 ms band, so the buckets are chosen explicitly. Without this, a p99 TTFB
target of 200 ms is unmeasurable -- every observation lands in one bucket.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import structlog
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

from flowtts.dhvaani.types import RequestMetrics

logger = structlog.get_logger(__name__)

# Latency buckets in milliseconds, dense across the range a real-time TTS
# actually lives in.
_MS_BUCKETS = (10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 5000, 10000)
_SHORT_MS_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 1000)

_LABELS = ("voice", "language")


def _get_or_create(cls, name, doc, labels=(), **kw):
    """Register a metric, returning the existing one if the name is taken."""
    existing = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
    if existing is not None:
        return existing
    try:
        return cls(name, doc, labels, **kw) if labels else cls(name, doc, **kw)
    except ValueError:
        return getattr(REGISTRY, "_names_to_collectors", {}).get(name)


REQUESTS = _get_or_create(Counter, "dhvaani_requests_total", "Completed TTS requests", _LABELS)
ERRORS = _get_or_create(Counter, "dhvaani_errors_total", "Failed TTS requests", _LABELS)
SPANS = _get_or_create(Counter, "dhvaani_spans_total", "Spans rendered", ("bucket",))
AUDIO_SECONDS = _get_or_create(
    Counter, "dhvaani_audio_seconds_total", "Seconds of audio generated", _LABELS
)
CHARS = _get_or_create(Counter, "dhvaani_input_chars_total", "Input characters", _LABELS)
OOV_CHARS = _get_or_create(
    Counter, "dhvaani_oov_chars_total", "Characters dropped as out-of-vocabulary", ("language",)
)

TTFB = _get_or_create(
    Histogram, "dhvaani_ttfb_ms", "Time to first audio byte (ms)", _LABELS,
    buckets=_MS_BUCKETS,
)
TOTAL = _get_or_create(
    Histogram, "dhvaani_total_ms", "End-to-end request time (ms)", _LABELS,
    buckets=_MS_BUCKETS,
)
FLOW = _get_or_create(
    Histogram, "dhvaani_flow_ms", "Flow-decoder time per request (ms)", (),
    buckets=_MS_BUCKETS,
)
VOCODE = _get_or_create(
    Histogram, "dhvaani_vocode_ms", "Vocoder time per request (ms)", (),
    buckets=_MS_BUCKETS,
)
QUEUE = _get_or_create(
    Histogram, "dhvaani_queue_ms", "Scheduler queue wait (ms)", (), buckets=_MS_BUCKETS
)
NORMALIZE = _get_or_create(
    Histogram, "dhvaani_normalize_ms", "Text normalisation (ms)", (),
    buckets=_SHORT_MS_BUCKETS,
)
RTF = _get_or_create(
    Histogram, "dhvaani_rtf", "Real-time factor (total / audio seconds)", (),
    buckets=(0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0),
)

ACTIVE_SPANS = _get_or_create(Gauge, "dhvaani_active_spans", "Spans in the flow arena")
QUEUE_DEPTH = _get_or_create(Gauge, "dhvaani_queue_depth", "Spans waiting for a slot")
MEAN_BATCH = _get_or_create(Gauge, "dhvaani_mean_batch", "Mean flow-step batch size")
STEPS_PER_S = _get_or_create(Gauge, "dhvaani_steps_per_second", "Flow steps per second")
ARENA_OCCUPANCY = _get_or_create(
    Gauge, "dhvaani_arena_occupancy", "Occupied slots per bucket", ("bucket",)
)
ARENA_BYTES = _get_or_create(Gauge, "dhvaani_arena_bytes", "Bytes held by the flow arenas")
INFLIGHT = _get_or_create(Gauge, "dhvaani_inflight_requests", "Requests being streamed")

VOICE_CACHE_HITS = _get_or_create(Counter, "dhvaani_voice_cache_hits_total", "Voice GPU cache hits")
VOICE_CACHE_MISSES = _get_or_create(
    Counter, "dhvaani_voice_cache_misses_total", "Voice GPU cache misses"
)
TEXT_CACHE_HITS = _get_or_create(Counter, "dhvaani_text_cache_hits_total", "Normaliser cache hits")
TEXT_CACHE_MISSES = _get_or_create(
    Counter, "dhvaani_text_cache_misses_total", "Normaliser cache misses"
)

_JSONL = Path(__file__).parents[3] / "monitoring" / "dhvaani_calls.jsonl"
_jsonl_file = None


def _jsonl():
    """Append-only per-request log, mirroring the legacy path's calls.jsonl."""
    global _jsonl_file
    if _jsonl_file is None:
        try:
            _JSONL.parent.mkdir(parents=True, exist_ok=True)
            _jsonl_file = _JSONL.open("a", buffering=1)
        except Exception as e:
            logger.warning("dhvaani_jsonl_unavailable", error=str(e))
            _jsonl_file = False
    return _jsonl_file or None


def record_request(m: RequestMetrics) -> None:
    labels = (m.voice_id or "default", m.language or "unknown")
    if m.error:
        ERRORS.labels(*labels).inc()
    else:
        REQUESTS.labels(*labels).inc()
        AUDIO_SECONDS.labels(*labels).inc(m.audio_s)
        CHARS.labels(*labels).inc(m.n_chars)
        if m.ttfb_ms:
            TTFB.labels(*labels).observe(m.ttfb_ms)
        TOTAL.labels(*labels).observe(m.total_ms)
        FLOW.observe(m.flow_ms)
        VOCODE.observe(m.vocode_ms)
        QUEUE.observe(m.queue_ms)
        NORMALIZE.observe(m.normalize_ms)
        if m.audio_s > 0:
            RTF.observe(m.rtf)

    f = _jsonl()
    if f is not None:
        try:
            f.write(
                json.dumps(
                    {
                        "ts": round(time.time(), 3),
                        "request_id": m.request_id,
                        "voice": m.voice_id,
                        "language": m.language,
                        "chars": m.n_chars,
                        "spans": m.n_spans,
                        "ttfb_ms": round(m.ttfb_ms, 2),
                        "total_ms": round(m.total_ms, 2),
                        "flow_ms": round(m.flow_ms, 2),
                        "vocode_ms": round(m.vocode_ms, 2),
                        "normalize_ms": round(m.normalize_ms, 3),
                        "audio_s": round(m.audio_s, 3),
                        "rtf": round(m.rtf, 4),
                        "error": m.error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        except Exception:
            pass


def record_span(bucket: int) -> None:
    SPANS.labels(str(bucket)).inc()


def record_oov(language: str, n: int) -> None:
    if n:
        OOV_CHARS.labels(language or "unknown").inc(n)


def update_from_stats(stats: dict) -> None:
    """Refresh gauges from `DhvaaniEngine.stats()`. Called by the /metrics handler
    so the gauges are accurate at scrape time without a background poller."""
    sched = stats.get("scheduler") or {}
    ACTIVE_SPANS.set(sched.get("active_rows", 0))
    QUEUE_DEPTH.set(sched.get("queue_depth", 0))
    MEAN_BATCH.set(sched.get("mean_batch", 0))
    STEPS_PER_S.set(sched.get("steps_per_s", 0))

    arenas = sched.get("arenas") or {}
    ARENA_BYTES.set(float(arenas.get("total_mib", 0)) * 2**20)
    for bucket, info in (arenas.get("buckets") or {}).items():
        ARENA_OCCUPANCY.labels(str(bucket)).set(info.get("occupied", 0))


def snapshot(stats: dict) -> dict:
    """Compact JSON view for /v1/stats and the control API."""
    sched = stats.get("scheduler") or {}
    return {
        "ready": stats.get("ready"),
        "backend": stats.get("backend"),
        "requests": stats.get("requests"),
        "errors": stats.get("errors"),
        "queue_depth": sched.get("queue_depth"),
        "active_spans": sched.get("active_rows"),
        "mean_batch": sched.get("mean_batch"),
        "steps_per_s": sched.get("steps_per_s"),
        "vram": stats.get("vram"),
        "voices": stats.get("voices"),
        "text_cache": stats.get("text_cache"),
    }
