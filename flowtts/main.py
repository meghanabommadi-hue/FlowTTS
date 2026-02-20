"""Main FastAPI application for FlowTTS.

This mirrors the shape of ``litranscriber.main`` but is intentionally
minimal: a single WebSocket endpoint for text-to-speech.

On-demand model loading:
  The worker loads the sglang model lazily on the first job it receives.
  Gateways register themselves via FLOWTTS_KNOWN_PORTS so /ports can
  report which ports are currently live.
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
