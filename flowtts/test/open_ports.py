"""Open N WebSocket ports on a running FlowTTS server via the control API.

Usage:
    python -m flowtts.test.open_ports --n 40
    python -m flowtts.test.open_ports --n 50 --base-port 8805
    python -m flowtts.test.open_ports --ports 8900,8901,8902
    python -m flowtts.test.open_ports --ctrl-port 8764 --n 10
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


_DEFAULT_CTRL_PORT = 8764
_DEFAULT_BASE_PORT = 8765


def _ctrl_get(ctrl_port: int, path: str) -> dict:
    url = f"http://127.0.0.1:{ctrl_port}{path}"
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def _ctrl_post(ctrl_port: int, path: str) -> dict:
    url = f"http://127.0.0.1:{ctrl_port}{path}"
    req = urllib.request.Request(url, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Open WS ports on a running FlowTTS server")
    parser.add_argument("--ctrl-port", type=int, default=_DEFAULT_CTRL_PORT,
                        help=f"Control API port (default: {_DEFAULT_CTRL_PORT})")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--n", type=int, metavar="N",
                     help="Number of ports to open sequentially from --base-port")
    grp.add_argument("--ports", type=str, metavar="LIST",
                     help="Explicit comma-separated port list, e.g. 8900,8901,8902")

    parser.add_argument("--base-port", type=int, default=None,
                        help="Starting port for --n (default: next after highest open port)")
    args = parser.parse_args()

    # Check server is reachable
    try:
        data = _ctrl_get(args.ctrl_port, "/ports")
    except Exception as e:
        print(f"ERROR: cannot reach ctrl API on port {args.ctrl_port}: {e}", file=sys.stderr)
        sys.exit(1)

    existing = set(data.get("ports", []))

    # Resolve port list
    if args.ports:
        targets = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
    else:
        base = args.base_port
        if base is None:
            base = max(existing) + 1 if existing else _DEFAULT_BASE_PORT
        targets = [base + i for i in range(args.n)]

    opened, skipped = [], []
    for port in targets:
        if port in existing:
            skipped.append(port)
            continue
        try:
            result = _ctrl_post(args.ctrl_port, f"/ports/add?port={port}")
            opened.append(port)
        except Exception as e:
            print(f"ERROR: failed to open port {port}: {e}", file=sys.stderr)

    if skipped:
        print(f"skipped (already open): {skipped}")
    if opened:
        print(f"opened {len(opened)} port(s): {opened}")

    # Print current total
    data = _ctrl_get(args.ctrl_port, "/ports")
    all_ports = data.get("ports", [])
    print(f"total open: {len(all_ports)} ports  ({min(all_ports)}–{max(all_ports)})")


if __name__ == "__main__":
    main()
