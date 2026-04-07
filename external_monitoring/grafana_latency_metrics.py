"""LLM / decoder / end-to-end latency histograms parsed from journald log lines.

Matches the structlog 'done' line emitted by flowtts.server:
  done  llm=<N>ms decode=<N>ms <tokens>tok total=<N>ms
"""

import re
from prometheus_client import Histogram

# 100 ms-wide buckets: 0-99, 100-199, ..., 2900-2999, then catch-all
_BUCKETS = list(range(0, 3100, 100))  # [0,100,200,...,3000]

LLM_LATENCY   = Histogram('flowtts_llm_latency_ms',   'LLM inference latency in ms',   buckets=_BUCKETS)
DECODE_LATENCY = Histogram('flowtts_decode_latency_ms', 'Decoder latency in ms',         buckets=_BUCKETS)
E2E_LATENCY   = Histogram('flowtts_e2e_latency_ms',   'End-to-end latency in ms',      buckets=_BUCKETS)

# Matches both log formats:
#   Non-streaming: done  llm=790ms  decode=1328ms  wav_enc=2ms  total=2121ms  ...
#   Streaming:     stream_done  chunks=N  tokens=N  llm_ttft=Xms  dec_ttft=Xms  decode=Xms  wav_enc=Xms  total=Xms  ...
_re_done = re.compile(r"\bdone\s+llm=(\d+)ms\s+decode=(\d+)ms\s+\S+\s+total=(\d+)ms")
_re_stream_done = re.compile(r"\bstream_done\b.*?\bdecode=(\d+)ms\b.*?\btotal=(\d+)ms\b")


def parse_line(line: str) -> None:
    m = _re_done.search(line)
    if m:
        LLM_LATENCY.observe(int(m.group(1)))
        DECODE_LATENCY.observe(int(m.group(2)))
        E2E_LATENCY.observe(int(m.group(3)))
        return
    m = _re_stream_done.search(line)
    if m:
        # Streaming path has no separate llm= field; decode covers decoder time only
        DECODE_LATENCY.observe(int(m.group(1)))
        E2E_LATENCY.observe(int(m.group(2)))
