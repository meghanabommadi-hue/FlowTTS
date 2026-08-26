"""Pipeline position: HTTP APP — assembles the FastAPI application.

Role in pipeline:
  Wires the REST and voice routers onto one app, installs CORS, optional API-key
  auth and the DhvaaniError -> HTTP mapping, and holds the engine on
  `app.state`.

The engine is NOT started here. `server.py` starts it once, before any port is
bound, so a request can never arrive at a half-loaded model.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from flowtts.dhvaani.api import rest, voices_api
from flowtts.dhvaani.config import dhv_settings
from flowtts.dhvaani.types import DhvaaniError

logger = structlog.get_logger(__name__)


def create_app(engine, settings=None) -> FastAPI:
    s = settings or dhv_settings

    app = FastAPI(
        title="DhVaani TTS",
        version="0.5",
        description=(
            "Zero-shot multilingual TTS for 27 Indian languages, served with "
            "continuous-batched flow-matching inference. OpenAI-compatible "
            "speech endpoint plus voice-clone management."
        ),
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.engine = engine
    app.state.settings = s

    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.server.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-Id", "X-TTFB-Ms", "X-Total-Ms", "X-Audio-Seconds",
            "X-RTF", "X-Sample-Rate",
        ],
    )

    @app.exception_handler(DhvaaniError)
    async def _dhvaani_error(request: Request, exc: DhvaaniError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": str(exc), "type": "dhvaani_error", "code": exc.code}},
        )

    app.include_router(rest.router)
    app.include_router(voices_api.router)

    @app.get("/")
    async def root():
        return {
            "service": "dhvaani",
            "model": "ARTPARK-IISc/DhVaani-0.5",
            "ready": bool(engine and engine.ready),
            "endpoints": [
                "POST /v1/audio/speech",
                "GET  /v1/models",
                "GET  /v1/languages",
                "POST /v1/voices",
                "GET  /v1/voices",
                "GET  /v1/voices/{voice_id}",
                "DELETE /v1/voices/{voice_id}",
                "POST /v1/voices/{voice_id}/preview",
                "GET  /v1/stats",
                "GET  /metrics",
                "GET  /healthz",
            ],
        }

    return app
