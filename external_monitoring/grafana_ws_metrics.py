"""WebSocket open/close/active/max metrics parsed from journald log lines.

Regex patterns match the structlog output from flowtts.server:
  ws_connection_open  call_id=<uuid>
  ws_connection_close call_id=<uuid>
"""

import re
from prometheus_client import Counter, Gauge

_re_open       = re.compile(r"ws_connection_open\s+call_id=(\S+)")
_re_close      = re.compile(r"ws_connection_close\s+call_id=(\S+)")
_re_error_1000 = re.compile(r"ERROR:.*\b1000\b.*\(OK\)")

WS_OPENED      = Counter('flowtts_ws_opened_total', 'Total WebSocket connections opened')
WS_CLOSED      = Counter('flowtts_ws_closed_total', 'Total WebSocket connections closed')
WS_ACTIVE      = Gauge('flowtts_ws_active',     'Currently active WebSocket connections')
WS_MAX         = Gauge('flowtts_ws_max_active', 'Max concurrent WebSocket connections seen')
WS_CONCURRENCY     = Counter('flowtts_ws_concurrency_level_total',
                             'How many times each concurrency level was observed',
                             labelnames=['level'])
WS_CLEAN_DISCONNECT = Counter('flowtts_ws_clean_disconnect_total',
                              'WebSocket disconnects via ERROR 1000 (OK) clean close')

_active = 0
_max    = 0


def parse_line(line: str) -> None:
    global _active, _max

    if _re_open.search(line):
        WS_OPENED.inc()
        _active += 1
        WS_ACTIVE.set(_active)
        if _active > _max:
            _max = _active
            WS_MAX.set(_max)
        WS_CONCURRENCY.labels(level=str(_active)).inc()

    elif _re_close.search(line):
        WS_CLOSED.inc()
        _active = max(0, _active - 1)
        WS_ACTIVE.set(_active)
        WS_CONCURRENCY.labels(level=str(_active)).inc()

    elif _re_error_1000.search(line):
        WS_CLOSED.inc()
        WS_CLEAN_DISCONNECT.inc()
        _active = max(0, _active - 1)
        WS_ACTIVE.set(_active)
        WS_CONCURRENCY.labels(level=str(_active)).inc()
