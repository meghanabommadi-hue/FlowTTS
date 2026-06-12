import subprocess
import time
import os
import re
import urllib.request
import json


_CTRL_PORT = 8764


def get_active_connections() -> dict | None:
    """Poll /ws/active on the ctrl API. Returns None if server unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{_CTRL_PORT}/ws/active", timeout=1
        ) as r:
            return json.loads(r.read())
    except Exception:
        return None


def get_logs() -> str:
    result = subprocess.run(
        ["journalctl", "-u", "flowtts*", "--since", "today"],
        capture_output=True,
        text=True,
    )
    return result.stdout


def _stats(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    return f"min={min(vals):.0f}  avg={sum(vals)/len(vals):.0f}  max={max(vals):.0f} ms"


def parse_logs(logs: str) -> tuple[int, int, int, list[str], list[str], dict[int, int], int, dict]:
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    re_open  = re.compile(r"ws_connection_open\s+call_id=(\S+)")
    re_close = re.compile(r"ws_connection_close\s+call_id=(\S+)")
    re_clean = re.compile(r"ERROR:.*\b1000\b.*\(OK\)")
    re_req = re.compile(rf"({uuid_pattern})\s+(?:req|stream)\s")
    re_done = re.compile(rf"({uuid_pattern})\s+(?:done|stream_done)\s")
    re_error = re.compile(rf"({uuid_pattern})\s+ERROR:")
    re_restart    = re.compile(r"flowtts\.service.*(?:Main process exited|Started flowtts)")
    re_stream_done = re.compile(
        r"stream_done\s+.*?llm_ttft=(\d+)ms\s+dec_ttft=(\d+)ms\s+.*?total=(\d+)ms"
    )

    opened: int = 0
    closed: int = 0
    req_ids: set[str] = set()
    done_ids: set[str] = set()
    error_ids: set[str] = set()
    clean_ids: set[str] = set()
    active_now: set[str] = set()
    level_counts: dict[int, int] = {}
    e2e_ms:      list[float] = []
    llm_ttft_ms: list[float] = []
    dec_ttft_ms: list[float] = []

    for line in logs.splitlines():
        if re_restart.search(line):
            # Service died/restarted — all open connections are implicitly closed
            closed += len(active_now)
            active_now.clear()
            # Any req seen so far without done/error is now orphaned — count as error
            error_ids.update(req_ids - done_ids - error_ids)
            level_counts[0] = level_counts.get(0, 0) + 1
            continue
        m = re_open.search(line)
        if m:
            call_id = m.group(1)
            opened += 1
            active_now.add(call_id)
            level = len(active_now)
            level_counts[level] = level_counts.get(level, 0) + 1
            continue
        m = re_close.search(line)
        if m:
            call_id = m.group(1)
            closed += 1
            active_now.discard(call_id)
            # record level after close so idle periods (0) are counted correctly
            level = len(active_now)
            level_counts[level] = level_counts.get(level, 0) + 1
            continue
        if re_clean.search(line):
            m2 = re.search(uuid_pattern, line)
            if m2:
                clean_ids.add(m2.group(0))
            continue
        m = re_req.search(line)
        if m:
            req_ids.add(m.group(1))
            continue
        m = re_stream_done.search(line)
        if m:
            llm_ttft_ms.append(float(m.group(1)))
            dec_ttft_ms.append(float(m.group(2)))
            e2e_ms.append(float(m.group(3)))
            # fall through to re_done to also register in done_ids
        m = re_done.search(line)
        if m:
            done_ids.add(m.group(1))
            continue
        m = re_error.search(line)
        if m:
            error_ids.add(m.group(1))

    processing_ports = sorted(active_now)                       # connections open but not yet closed
    processing_uuids = sorted(req_ids - done_ids - error_ids - clean_ids)  # in-flight requests
    active = len(active_now)
    call_count = len(done_ids) + len(error_ids)                 # completed calls today
    latency_stats = {
        "e2e":      e2e_ms,
        "llm_ttft": llm_ttft_ms,
        "dec_ttft": dec_ttft_ms,
    }

    return opened, closed, active, processing_ports, processing_uuids, level_counts, call_count, latency_stats


def clear():
    os.system("clear")


def main(interval: float = 0.2):
    session_counts: dict[int, int] = {}

    while True:
        logs = get_logs()
        opened, closed, active, processing_ports, processing_uuids, level_counts, call_count, latency_stats = parse_logs(logs)
        live = get_active_connections()

        max_active_log = max(level_counts) if level_counts else 0
        session_counts[active] = session_counts.get(active, 0) + 1

        lines = []
        lines.append(f"Opened     : {opened}")
        lines.append(f"Closed     : {closed}")
        lines.append(f"Active     : {active}")
        lines.append(f"Max Active : {max_active_log}")
        lines.append(f"Calls Done : {call_count}")
        lines.append("")
        lines.append(f"E2E latency  : {_stats(latency_stats['e2e'])}")
        lines.append(f"LLM TTFT     : {_stats(latency_stats['llm_ttft'])}")
        lines.append(f"Decoder TTFT : {_stats(latency_stats['dec_ttft'])}")
        lines.append("")

        # ── Live active connections panel ──────────────────────────────────────
        if live is not None:
            live_ids = live.get("active_ids", [])
            lines.append(f"Live Active Connections ({live['active']}):")
            lines.append("-" * 40)
            if live_ids:
                for cid in sorted(live_ids):
                    lines.append(f"  {cid}")
            else:
                lines.append("  (none)")
        else:
            lines.append("Live Active Connections: (ctrl API unreachable)")
        lines.append("")
        lines.append(f"Processing ports ({len(processing_ports)}):")
        lines.append("-" * 40)
        if processing_ports:
            for port in processing_ports:
                lines.append(f"  {port}")
        else:
            lines.append("  (none)")
        lines.append("")
        lines.append(f"Processing call_ids ({len(processing_uuids)}):")
        lines.append("-" * 40)
        if processing_uuids:
            for uid in processing_uuids:
                lines.append(f"  {uid}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("Activity graph (this session):")
        lines.append("-" * 40)
        max_session = max(session_counts.values()) if session_counts else 1
        for level in range(max(session_counts) + 1):
            n = session_counts.get(level, 0)
            bar = "█" * int(n / max_session * 30)
            lines.append(f"  {level} | {bar:<30} {n}")

        lines.append("")
        lines.append("Activity graph (today):")
        lines.append("-" * 40)
        if level_counts:
            max_count = max(level_counts.values())
            for level in range(max(level_counts) + 1):
                n = level_counts.get(level, 0)
                bar = "█" * int(n / max_count * 30)
                lines.append(f"  {level} | {bar:<30} {n}")
        else:
            lines.append("  (no data)")

        clear()
        print("\n".join(lines))

        time.sleep(interval)


if __name__ == "__main__":
    # Change interval to 0.1 for 100ms or 0.5 for 500ms
    main(interval=0.5)
