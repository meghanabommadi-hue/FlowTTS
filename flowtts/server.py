#!/usr/bin/env python3
"""Pipeline position: COMPATIBILITY SHIM — `python -m flowtts.server` → flowtts.service.

Role in pipeline:
  This module used to be the single-process WebSocket server. Everything it did
  now lives in :mod:`flowtts.service`, which serves the identical WebSocket
  protocol on the same raw ports *and* adds the REST / OpenAI-compatible /
  streaming HTTP surface on one shared model load.

  Keeping two implementations of the request path was the real risk: the
  streaming contract, the parameter set and the chunking all changed in this
  version, and a second copy would have drifted silently. So this file forwards
  its arguments and exits.

  The previous implementation is in git history at this path, if you need to
  compare against it.

Old flags map across unchanged:

    python -m flowtts.server --ports 2 --base-port 8080 --ctrl-port 8764
      ≡ python -m flowtts.service --ws-ports 2 --base-port 8080 --ctrl-port 8764

Prefer calling ``flowtts.service`` directly in new deployments — it exposes the
HTTP port, which is what nginx proxies.
"""

from __future__ import annotations

import argparse
import sys

_DEPRECATION = (
    "[flowtts.server] deprecated: forwarding to flowtts.service. "
    "Update your launcher to `python -m flowtts.service` to reach the HTTP API."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlowTTS WebSocket server (deprecated; forwards to flowtts.service)"
    )
    parser.add_argument("--base-port", type=int, default=8080)
    parser.add_argument("--ports", type=int, default=1)
    parser.add_argument("--ctrl-port", type=int, default=None)
    parser.add_argument("--http-port", type=int, default=None)
    parser.add_argument("--profile", default=None)
    # Accepted and ignored: the new service writes audio through the API rather
    # than to a directory, and this flag has no equivalent.
    parser.add_argument("--save-audio", default=None)
    args, extra = parser.parse_known_args()

    print(_DEPRECATION, file=sys.stderr, flush=True)
    if args.save_audio:
        print("[flowtts.server] --save-audio is no longer supported; ignoring.",
              file=sys.stderr, flush=True)

    forwarded = ["--ws-ports", str(args.ports), "--base-port", str(args.base_port)]
    if args.ctrl_port:
        forwarded += ["--ctrl-port", str(args.ctrl_port)]
    if args.http_port:
        forwarded += ["--http-port", str(args.http_port)]
    if args.profile:
        forwarded += ["--profile", args.profile]

    from flowtts import service

    sys.argv = ["flowtts.service", *forwarded, *extra]
    service.main()


if __name__ == "__main__":
    main()
