"""WebSocket gateway for FlowTTS.

This is a single-process implementation that:
- accepts text synthesis requests over WebSocket,
- uses ``synthesis.engine`` to obtain audio tokens,
- uses ``decoder.decoder`` to obtain WAV bytes,
- streams the audio back to the client.

It follows the LITranscriber style (FastAPI router + connection handler),
but omits Redis/worker for simplicity.
"""

from __future__ import annotations

import base64
import json
from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import structlog

from flowtts.api.models import (
    SynthesizeRequest,
    AudioMessage,
    ErrorMessage,
    MessageType,
)
from flowtts.core.config import settings
from flowtts.synthesis.engine import synthesis_service
from flowtts.decoder.decoder import decoder as audio_decoder


logger = structlog.get_logger(__name__)

router = APIRouter()


class ConnectionManager:
    """Minimal connection manager for FlowTTS."""

    def __init__(self) -> None:
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, call_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections[call_id] = websocket
        logger.info("websocket_connected", call_id=call_id)

    async def disconnect(self, call_id: str) -> None:
        ws = self.active_connections.pop(call_id, None)
        if ws is not None:
            logger.info("websocket_disconnected", call_id=call_id)

    async def send_error(self, websocket: WebSocket, error: str, call_id: str | None = None, text_id: str | None = None) -> None:
        msg = ErrorMessage(call_id=call_id, text_id=text_id, error=error)
        await websocket.send_json(json.loads(msg.model_dump_json()))


manager = ConnectionManager()


@router.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str) -> None:
    """Handle FlowTTS WebSocket connections."""
    await manager.connect(call_id, websocket)

    try:
        async for raw in websocket.iter_text():
            try:
                payload = json.loads(raw)
                if payload.get("type") != MessageType.SYNTHESIZE:
                    raise ValueError("Unsupported message type")

                req = SynthesizeRequest(**payload)
            except Exception as e:
                await manager.send_error(websocket, f"Bad request: {e}", call_id=call_id)
                continue

            try:
                # Ensure synthesis service is initialized
                if not synthesis_service.is_initialized:
                    await synthesis_service.initialize()

                # 1) Text → audio tokens
                audio_tokens = await synthesis_service.synthesize(req.text)

                # 2) Audio tokens → WAV bytes
                decoded = audio_decoder.decode_to_wav(audio_tokens)

                # 3) Encode as base64 and send back
                audio_b64 = base64.b64encode(decoded.wav_bytes).decode("ascii")

                resp = AudioMessage(
                    call_id=req.call_id,
                    text_id=req.text_id,
                    audio_base64=audio_b64,
                    sample_rate=decoded.sample_rate,
                    is_final=True,
                )
                await websocket.send_json(json.loads(resp.model_dump_json()))

            except Exception as e:
                logger.error("synthesis_failed", call_id=call_id, text_id=req.text_id, error=str(e))
                await manager.send_error(websocket, f"Synthesis failed: {e}", call_id=call_id, text_id=req.text_id)

    except WebSocketDisconnect:
        logger.info("websocket_disconnect", call_id=call_id)
    finally:
        await manager.disconnect(call_id)

