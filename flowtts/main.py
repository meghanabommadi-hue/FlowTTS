"""Main FastAPI application for FlowTTS.

This mirrors the shape of ``litranscriber.main`` but is intentionally
minimal: a single WebSocket endpoint for text-to-speech.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from flowtts.api.websockets import router as websocket_router
from flowtts.core.config import settings


logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="FlowTTS",
        description="Simple text-to-speech gateway over WebSocket.",
        version="0.1.0",
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

