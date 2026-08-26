#!/usr/bin/env python3
"""Pipeline position: PROCESS ENTRY POINT — one process, every listener.

Role in pipeline:
  Loads OmniVoice once and serves it on all three surfaces from a single event
  loop, so there is one model in VRAM and one batch queue coalescing everything:

      :8000   HTTP   REST + OpenAI-compatible + /ws + docs + metrics  (FastAPI)
      :8080…  WS     the raw-port FlowTTS protocol, for existing clients
      :8764   HTTP   control API: healthz / readyz / stats / metrics / ports

  The raw WS ports exist only for backward compatibility: the FastAPI app serves
  the identical protocol at /ws, which is what nginx proxies. New clients should
  use that; old ones keep working without a change.

Usage:
    python -m flowtts.service --profile balanced
    python -m flowtts.service --http-port 9000 --ws-ports 2 --base-port 9080
    python -m flowtts.service --profile fast --no-legacy-ws
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import uuid

import structlog

from flowtts.core.config import PROFILES, apply_profile, settings

logger = structlog.get_logger(__name__)

_legacy_ports: set[int] = set()


# ---------------------------------------------------------------------------
# Raw-port WebSocket (legacy protocol, same handler as the FastAPI /ws route)
# ---------------------------------------------------------------------------
class _RawWebSocketAdapter:
    """Presents a `websockets` connection with the FastAPI WebSocket interface.

    Lets both transports run the exact same session handler, so the protocol
    cannot drift between the raw port and the proxied /ws route.
    """

    def __init__(self, connection) -> None:
        self._ws = connection

    async def accept(self) -> None:            # already accepted by websockets.serve
        return None

    async def receive_text(self) -> str:
        message = await self._ws.recv()
        return message.decode("utf-8") if isinstance(message, bytes) else message

    async def send_text(self, data: str) -> None:
        await self._ws.send(data)

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._ws.close(code, reason)


async def _serve_legacy_ports(base_port: int, count: int) -> None:
    """Bind `count` consecutive raw WebSocket ports."""
    import websockets
    from websockets.exceptions import ConnectionClosed

    from flowtts.api.http_app import _ws_session
    from flowtts.api.service import service

    async def _handler(connection, port: int) -> None:
        path = getattr(getattr(connection, "request", None), "path", "") or ""
        call_id = path.rsplit("/", 1)[-1] or str(uuid.uuid4())
        try:
            await _ws_session(_RawWebSocketAdapter(connection), call_id)
        except ConnectionClosed:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("legacy_ws_error", port=port, error=str(exc))

    async def _process_request(connection, request):
        """Answer /health and /metrics on the WS port without an upgrade."""
        from websockets.http11 import Headers, Response

        if request.path == "/health":
            ready = service.ready and not service.restarting
            body = json.dumps({"status": "ok" if ready else "loading",
                               "ready": ready}).encode()
            status = (200, "OK") if ready else (503, "Service Unavailable")
        elif request.path == "/metrics":
            from prometheus_client import generate_latest
            body, status = generate_latest(), (200, "OK")
        else:
            return None   # proceed with the WebSocket handshake

        headers = Headers([("Content-Type", "application/json"),
                           ("Content-Length", str(len(body)))])
        return Response(status[0], status[1], headers, body)

    for offset in range(count):
        port = base_port + offset
        await websockets.serve(
            lambda conn, p=port: _handler(conn, p),
            settings.server.host, port,
            ping_interval=30, ping_timeout=30,
            max_size=64 * 1024 * 1024,
            process_request=_process_request,
        )
        _legacy_ports.add(port)
        logger.info("legacy_ws_listening", url=f"ws://{settings.server.host}:{port}")


# ---------------------------------------------------------------------------
# Control API
# ---------------------------------------------------------------------------
async def _serve_control(port: int) -> None:
    """Small aiohttp control plane, separate from the public HTTP port.

    Kept separate so health checks and Prometheus keep answering even when the
    public port is saturated with synthesis requests.
    """
    from aiohttp import web

    from flowtts.api.service import service

    async def healthz(_: web.Request) -> web.Response:
        if service.restarting:
            return web.json_response({"status": "error", "reason": "restarting"}, status=503)
        if not service.ready:
            return web.json_response({"status": "loading"}, status=503)
        return web.json_response({"status": "ok", "ready": True})

    async def readyz(_: web.Request) -> web.Response:
        ok = service.ready and not service.restarting and not service.oom_recovery
        return web.json_response(
            {"ready": ok, "oom_recovery": service.oom_recovery,
             "ports": sorted(_legacy_ports)},
            status=200 if ok else 503,
        )

    async def stats(_: web.Request) -> web.Response:
        return web.json_response(service.stats())

    async def metrics(_: web.Request) -> web.Response:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return web.Response(body=generate_latest(),
                            content_type=CONTENT_TYPE_LATEST.split(";")[0].strip())

    async def ports(_: web.Request) -> web.Response:
        return web.json_response({"ws_ports": sorted(_legacy_ports),
                                  "http_port": settings.server.http_port})

    app = web.Application()
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/health", healthz)
    app.router.add_get("/readyz", readyz)
    app.router.add_get("/ready", readyz)
    app.router.add_get("/stats", stats)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/ports", ports)

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, settings.server.host, port).start()
    logger.info("control_listening", url=f"http://{settings.server.host}:{port}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def _run(args: argparse.Namespace) -> None:
    import uvicorn

    from flowtts.api.http_app import create_app
    from flowtts.api.service import service

    # The service loads the model itself, below, so the control port is
    # already answering "loading" while it happens.
    app = create_app(load_on_startup=False)

    config = uvicorn.Config(
        app,
        host=settings.server.host,
        port=args.http_port,
        log_level="warning",
        access_log=False,
        ws_max_size=64 * 1024 * 1024,
        timeout_keep_alive=75,
        # Streaming responses must not be buffered by the server either.
        h11_max_incomplete_event_size=16 * 1024 * 1024,
    )
    server = uvicorn.Server(config)

    if args.ctrl_port:
        await _serve_control(args.ctrl_port)

    logger.info("loading_model")
    await service.initialize()

    if args.legacy_ws and args.ws_ports > 0:
        await _serve_legacy_ports(args.base_port, args.ws_ports)

    logger.info(
        "flowtts_ready",
        http=f"http://{settings.server.host}:{args.http_port}",
        docs=f"http://{settings.server.host}:{args.http_port}/docs",
        ws=f"ws://{settings.server.host}:{args.http_port}/ws",
        legacy_ws=sorted(_legacy_ports),
        control=args.ctrl_port,
    )
    await server.serve()


def main() -> None:
    parser = argparse.ArgumentParser(description="FlowTTS / OmniVoice server")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None,
                        help="latency profile: fast | balanced | quality")
    parser.add_argument("--http-port", type=int, default=settings.server.http_port)
    parser.add_argument("--ctrl-port", type=int, default=settings.server.ctrl_port)
    parser.add_argument("--base-port", type=int, default=settings.server.ws_base_port,
                        help="first raw WebSocket port (legacy clients)")
    parser.add_argument("--ws-ports", type=int, default=settings.server.ws_ports,
                        help="how many raw WebSocket ports to bind")
    parser.add_argument("--no-legacy-ws", dest="legacy_ws", action="store_false",
                        help="serve WebSocket only at /ws on the HTTP port")
    parser.add_argument("--backend", default=None,
                        choices=["auto", "tensorrt", "trtllm", "torch", "pytorch"],
                        help="backbone backend (overrides config)")
    parser.add_argument("--num-step", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--max-batch", type=int, default=None)
    parser.set_defaults(legacy_ws=True)
    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ]
    )

    if args.profile:
        applied = apply_profile(args.profile)
        logger.info("profile_applied", profile=args.profile, **applied)

    # Explicit flags win over the profile.
    if args.backend:
        settings.omnivoice.backbone_backend = args.backend
    if args.num_step is not None:
        settings.generation.num_step = args.num_step
    if args.guidance_scale is not None:
        settings.generation.guidance_scale = args.guidance_scale
    if args.max_batch is not None:
        settings.omnivoice.max_batch = args.max_batch

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:   # not available on every platform
            pass
    try:
        loop.run_until_complete(_run(args))
    except KeyboardInterrupt:
        logger.info("stopped")


if __name__ == "__main__":
    main()
