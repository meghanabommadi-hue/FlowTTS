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


def parse_logs(logs: str) -> tuple[int, int, int, list[str], list[str], dict[int, int], int]:
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    re_open  = re.compile(r"ws_connection_open\s+call_id=(\S+)")
    re_close = re.compile(r"ws_connection_close\s+call_id=(\S+)")
    re_clean = re.compile(r"ERROR:.*\b1000\b.*\(OK\)")
    re_req = re.compile(rf"({uuid_pattern})\s+req\s")
    re_done = re.compile(rf"({uuid_pattern})\s+done\s")
    re_error = re.compile(rf"({uuid_pattern})\s+ERROR:")
    re_restart = re.compile(r"flowtts\.service.*(?:Main process exited|Started flowtts)")

    port_open: set[str] = set()
    port_close: set[str] = set()
    req_ids: set[str] = set()
    done_ids: set[str] = set()
    error_ids: set[str] = set()
    active_now: set[str] = set()
    level_counts: dict[int, int] = {}

    for line in logs.splitlines():
        if re_restart.search(line):
            # Service died/restarted — all open ports are implicitly closed
            port_close.update(active_now)
            active_now.clear()
            # Any req seen so far without done/error is now orphaned — count as error
            error_ids.update(req_ids - done_ids - error_ids)
            level_counts[0] = level_counts.get(0, 0) + 1
            continue
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
        if re_clean.search(line):
            m2 = re.search(uuid_pattern, line)
            if m2:
                error_ids.add(m2.group(0))
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

    processing_ports = sorted(port_open - port_close)        # ports: open but not yet closed
    processing_uuids = sorted(req_ids - done_ids - error_ids)  # call_ids: req seen, no done/error yet
    opened = len(port_open)
    closed = len(port_close)
    active = len(port_open - port_close)
    call_count = len(done_ids) + len(error_ids)               # completed calls today

    return opened, closed, active, processing_ports, processing_uuids, level_counts, call_count


def clear():
    os.system("clear")


def main(interval: float = 0.2):
    session_counts: dict[int, int] = {}

    while True:
        logs = get_logs()
        opened, closed, active, processing_ports, processing_uuids, level_counts, call_count = parse_logs(logs)

        max_active_log = max(level_counts) if level_counts else 0
        session_counts[active] = session_counts.get(active, 0) + 1

        lines = []
        lines.append(f"Opened     : {opened}")
        lines.append(f"Closed     : {closed}")
        lines.append(f"Active     : {active}")
        lines.append(f"Max Active : {max_active_log}")
        lines.append(f"Calls Done : {call_count}")
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
