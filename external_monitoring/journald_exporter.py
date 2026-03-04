"""FlowTTS Journald Prometheus Exporter.

Reads flowtts.service logs from journald and exposes Prometheus metrics on :9101.

On startup:
  1. Backfills today's logs (journalctl --since today)
  2. Follows new lines as they arrive (journalctl -f)

Metrics exposed:
  flowtts_ws_opened_total       Counter  - WS connections opened
  flowtts_ws_closed_total       Counter  - WS connections closed
  flowtts_ws_active             Gauge    - Currently active WS connections
  flowtts_ws_max_active         Gauge    - Peak concurrent WS connections
  flowtts_llm_latency_ms        Histogram (100ms buckets)
  flowtts_decode_latency_ms     Histogram (100ms buckets)
  flowtts_e2e_latency_ms        Histogram (100ms buckets)

Run:
  python3 external_monitoring/journald_exporter.py
  # metrics at http://localhost:9101/metrics
"""

import subprocess
import sys
import time
from pathlib import Path

# Ensure imports work whether run from FlowTTS/ root or external_monitoring/ dir
sys.path.insert(0, str(Path(__file__).parent))

import prometheus_client

import grafana_ws_metrics as ws_metrics
import grafana_latency_metrics as latency_metrics

PORT = 9101


def _feed(proc: subprocess.Popen) -> None:
    """Read lines from a journalctl subprocess and feed to parsers."""
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace")
        ws_metrics.parse_line(line)
        latency_metrics.parse_line(line)


def main() -> None:
    prometheus_client.start_http_server(PORT)
    print(f"[flowtts_exporter] metrics on http://localhost:{PORT}/metrics", flush=True)

    # --- Phase 1: backfill today's history ---
    print("[flowtts_exporter] backfilling today's journald logs ...", flush=True)
    backfill = subprocess.Popen(
        ["journalctl", "-u", "flowtts*", "--since", "today", "--output=short", "--no-pager"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    _feed(backfill)
    backfill.wait()
    print("[flowtts_exporter] backfill done, following live logs ...", flush=True)

    # --- Phase 2: follow live ---
    while True:
        follow = subprocess.Popen(
            ["journalctl", "-u", "flowtts*", "-f", "--output=short", "--no-pager"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            _feed(follow)
        except KeyboardInterrupt:
            follow.terminate()
            sys.exit(0)
        except Exception as e:
            print(f"[flowtts_exporter] journalctl error: {e}, restarting in 5s", flush=True)
            follow.terminate()
            time.sleep(5)


if __name__ == "__main__":
    main()
