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
#   done  llm=790ms  decode=1328ms  wav_enc=2ms  total=2121ms  tokens=368  wav=235564B
#   done  llm=221ms  decode=480ms  wav_enc=1ms  total=702ms  tokens=83  wav=53164B
_re_done = re.compile(r"\bdone\s+llm=(\d+)ms\s+decode=(\d+)ms\s+\S+\s+total=(\d+)ms")


def parse_line(line: str) -> None:
    m = _re_done.search(line)
    if m:
        LLM_LATENCY.observe(int(m.group(1)))
        DECODE_LATENCY.observe(int(m.group(2)))
        E2E_LATENCY.observe(int(m.group(3)))
