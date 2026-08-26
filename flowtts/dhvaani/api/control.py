"""Pipeline position: CONTROL API — operational endpoints on a separate port.

Role in pipeline:
  Same routes `flowtts/server.py` exposes via aiohttp, so existing tooling
  (`flowtts/test/open_ports.py`, the run.sh readiness probe, the Grafana
  exporters) keeps working. Kept off the public REST port so it can be firewalled.

    POST /ports/add?port=N   bind another WebSocket port at runtime
    GET  /ports              list open ports
    GET  /ready              200 once the model is loaded and warm
    GET  /health             liveness
    GET  /metrics            Prometheus scrape
    GET  /ws/log             recent WebSocket events
    GET  /ws/active          live connections
    GET  /stats              full engine stats
"""

from __future__ import annotations

import structlog
from aiohttp import web

logger = structlog.get_logger(__name__)


def build_control_app(engine, gateway) -> web.Application:
    app = web.Application()

    async def add_port(req: web.Request) -> web.Response:
        try:
            port = int(req.rel_url.query["port"])
        except (KeyError, ValueError):
            return web.json_response({"error": "missing or invalid ?port=N"}, status=400)
        if not (1024 <= port <= 65535):
            return web.json_response({"error": "port out of range"}, status=400)
        opened = await gateway.bind_port(port)
        return web.json_response({"port": port, "opened": opened})

    async def list_ports(_req: web.Request) -> web.Response:
        return web.json_response({"ports": sorted(gateway.open_ports)})

    async def ready(_req: web.Request) -> web.Response:
        if engine is None or not engine.ready:
            return web.json_response({"ready": False, "reason": "loading"}, status=503)
        return web.json_response({"ready": True, "ports": sorted(gateway.open_ports)})

    async def health(_req: web.Request) -> web.Response:
        if engine is None or not engine.ready:
            return web.json_response({"status": "error", "reason": "not ready"}, status=503)
        return web.json_response({"status": "ok", "ready": True})

    async def metrics(_req: web.Request) -> web.Response:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        from flowtts.dhvaani.monitoring.metrics import update_from_stats

        if engine is not None and engine.ready:
            try:
                update_from_stats(engine.stats())
            except Exception:
                pass
        return web.Response(
            body=generate_latest(),
            content_type=CONTENT_TYPE_LATEST.split(";")[0].strip(),
        )

    async def ws_log(_req: web.Request) -> web.Response:
        from flowtts.monitoring.metrics import ws_log_snapshot

        return web.json_response(ws_log_snapshot())

    async def ws_active(_req: web.Request) -> web.Response:
        from flowtts.monitoring.metrics import snapshot_metrics

        return web.json_response(snapshot_metrics()["ws"])

    async def stats(_req: web.Request) -> web.Response:
        if engine is None or not engine.ready:
            return web.json_response({"ready": False}, status=503)
        payload = engine.stats()
        payload["gateway"] = gateway.stats()
        return web.json_response(payload)

    app.router.add_post("/ports/add", add_port)
    app.router.add_get("/ports", list_ports)
    app.router.add_get("/ready", ready)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/ws/log", ws_log)
    app.router.add_get("/ws/active", ws_active)
    app.router.add_get("/stats", stats)
    return app


async def start_control_api(engine, gateway, host: str, port: int) -> web.AppRunner:
    runner = web.AppRunner(build_control_app(engine, gateway))
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    logger.info("control_api_started", host=host, port=port)
    return runner
