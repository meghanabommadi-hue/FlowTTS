"""Pipeline position: GATEWAY (Redis-backed multi-process mode).

Role in pipeline:
  1. Accepts WebSocket connections from callers (one connection per call_id).
  2. Receives synthesize requests, publishes them to the Redis TTS queue.
  3. Subscribes to the per-call Redis Pub/Sub channel and forwards audio
     token results (+ optional decoded WAV) back to the caller.
  4. Exposes /health and /ports HTTP endpoints for ops/discovery.

When to use this vs server.py:
  Use main.py (via `python -m flowtts.main`) when you want the full
  Redis-backed multi-process architecture: one gateway process per port,
  separate worker process(es) for GPU inference.

  Use server.py (via `./run.sh`) when you want a simpler single-process
  setup — no Redis, no worker, model loaded once in-process.

Port discovery:
  Set FLOWTTS_KNOWN_PORTS=8765,8766,… so /ports can report which gateway
  ports are live without scanning the full range.
"""

from __future__ import annotations

import os
import socket
from contextlib import asynccontextmanager
from typing import List

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from flowtts.api.websockets import router as websocket_router, manager
from flowtts.core.config import settings
from flowtts.monitoring.logging import configure_logging


logger = structlog.get_logger(__name__)

# Ports that run.sh told us about (FLOWTTS_KNOWN_PORTS=8765,8766,8767)
_KNOWN_PORTS: List[int] = [
    int(p)
    for p in os.environ.get("FLOWTTS_KNOWN_PORTS", "").split(",")
    if p.strip().isdigit()
]


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup; clean up on shutdown."""
    configure_logging()
    logger.info("flowtts_gateway_starting", host=settings.ws.host, port=settings.ws.port)

    # Pre-initialize the Redis connection so the first WebSocket request
    # doesn't pay the connection latency.
    try:
        await manager.initialize_redis()
        logger.info("redis_connection_ready")
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
        # Continue anyway – will retry on first request

    yield

    # Shutdown: close shared Redis client
    if manager.redis_client is not None:
        await manager.redis_client.aclose()
        logger.info("redis_connection_closed")

    logger.info("flowtts_gateway_stopped")


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="FlowTTS",
        description="Simple text-to-speech gateway over WebSocket.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(websocket_router, tags=["websocket"])

    @app.get("/health")
    async def health() -> dict:
        return {
            "service": "FlowTTS",
            "status": "running",
            "ws_host": settings.ws.host,
            "ws_port": settings.ws.port,
        }

    @app.get("/ports")
    async def ports() -> dict:
        """Return all known gateway ports and which ones are currently live."""
        scan = _KNOWN_PORTS or list(range(8765, 8775))  # fallback: scan default range
        live = [p for p in scan if _port_open(p)]
        return {
            "known": scan,
            "live": live,
            "count": len(live),
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "flowtts.main:app",
        host=settings.ws.host,
        port=settings.ws.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
