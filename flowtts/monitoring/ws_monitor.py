import subprocess
import time
import os
import re


def get_logs() -> str:
    result = subprocess.run(
        ["journalctl", "-u", "flowtts*", "--since", "today"],
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_logs(logs: str) -> tuple[int, int, int, list[str], list[str], dict[int, int]]:
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    re_open = re.compile(r"ws_connection_open\s+call_id=(\S+)")
    re_close = re.compile(r"ws_connection_close\s+call_id=(\S+)")
    re_req = re.compile(rf"({uuid_pattern})\s+req\s")
    re_done = re.compile(rf"({uuid_pattern})\s+done\s")
    re_error = re.compile(rf"({uuid_pattern})\s+ERROR:")

    port_open: set[str] = set()
    port_close: set[str] = set()
    req_ids: set[str] = set()
    done_ids: set[str] = set()
    error_ids: set[str] = set()
    active_now: set[str] = set()
    level_counts: dict[int, int] = {}

    for line in logs.splitlines():
        m = re_open.search(line)
        if m:
            port = m.group(1)
            port_open.add(port)
            active_now.add(port)
            level = len(active_now)
            level_counts[level] = level_counts.get(level, 0) + 1
            continue
        m = re_close.search(line)
        if m:
            port = m.group(1)
            port_close.add(port)
            active_now.discard(port)
            level = len(active_now)
            level_counts[level] = level_counts.get(level, 0) + 1
            continue
        m = re_req.search(line)
        if m:
            req_ids.add(m.group(1))
            continue
        m = re_done.search(line)
        if m:
            done_ids.add(m.group(1))
            continue
        m = re_error.search(line)
        if m:
            error_ids.add(m.group(1))

    active_uuids = sorted(req_ids - done_ids - error_ids)
    active_ports = sorted(port_open - port_close)
    opened = len(port_open)
    closed = len(port_close)
    active = len(port_open - port_close)

    return opened, closed, active, active_uuids, active_ports, level_counts


def clear():
    os.system("clear")


def main(interval: float = 0.2):
    max_active = 0
    session_counts: dict[int, int] = {}

    while True:
        logs = get_logs()
        opened, closed, active, active_uuids, active_ports, level_counts = parse_logs(logs)

        max_active = max(max_active, active)
        session_counts[active] = session_counts.get(active, 0) + 1

        lines = []
        lines.append(f"Opened     : {opened}")
        lines.append(f"Closed     : {closed}")
        lines.append(f"Active     : {active}")
        lines.append(f"Max Active : {max_active}")
        lines.append("")
        lines.append("Processing UUIDs:")
        lines.append("-" * 40)
        lines.append("  " + ",  ".join(active_uuids) if active_uuids else "  (none)")
        lines.append("")
        lines.append("Active ports:")
        lines.append("-" * 40)
        if active_ports:
            for port in active_ports:
                lines.append(f"  {port}")
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
