"""Pipeline position: GATEWAY — WebSocket ↔ Redis bridge.

Role in pipeline:
  Sits at both ends of the async pipeline:
    • Inbound:  receives synthesize requests from callers over WebSocket,
                serialises them to JSON and pushes onto the Redis TTS queue.
    • Outbound: subscribes to the per-call Redis Pub/Sub channel and forwards
                the result to the WebSocket client.

Key classes:
  ConnectionManager
    connect()              — accept WebSocket, init Redis, start listener task.
    disconnect()           — cancel listener, close pubsub, release decoder.
    _publish_job_to_queue()— rpush {call_id, text_id, text} to Redis queue.
    _listen_for_results()  — subscribe to the appropriate channel and forward
                             AudioMessage to the WebSocket client.

Decoded path (settings.decoder.enabled = True):
  Worker publishes audio_tokens → flowtts:audio:{call_id}
  DecoderWorker (decoder/decoder.py) subscribes, runs ncodec, publishes WAV
    → flowtts:decoded:{call_id}
  Gateway subscribes to flowtts:decoded:{call_id}, forwards audio_base64 to client.

No-decode path (settings.decoder.enabled = False):
  Worker publishes audio_tokens → flowtts:audio:{call_id}
  Gateway subscribes to flowtts:audio:{call_id}, forwards raw tokens to client.

One ConnectionManager instance (`manager`) is shared across all WebSocket
connections in a single gateway process.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis.asyncio as redis
import structlog

from flowtts.api.models import (
    SynthesizeRequest,
    AudioMessage,
    ErrorMessage,
    MessageType,
)
from flowtts.core.config import settings
from flowtts.decoder.manager import DecoderManager
from flowtts.monitoring.metrics import (
    record_decode_latency,
    record_ws_connection_open,
    record_ws_connection_close,
)


logger = structlog.get_logger(__name__)

router = APIRouter()

_decoder_manager = DecoderManager()


class ConnectionManager:
    """Manages WebSocket connections for FlowTTS.

    TTS ConnectionManager (no Redis/audio buffers here; a future version can add them).
    """

    def __init__(self) -> None:
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_tasks: Dict[str, asyncio.Task] = {}
        self.redis_client: redis.Redis | None = None
        self.redis_pubsub_clients: Dict[str, Any] = {}

    async def initialize_redis(self) -> None:
        """Initialize Redis connection for the manager."""
        if self.redis_client is not None:
            return

        cfg = settings.redis
        redis_url = f"redis://{cfg.host}:{cfg.port}/{cfg.db}"
        self.redis_client = await redis.from_url(  # type: ignore[arg-type]
            redis_url,
            password=cfg.password,
            decode_responses=False,
        )
        logger.info("redis_client_initialized", host=cfg.host, port=cfg.port)

    async def connect(self, call_id: str, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection and register it by call_id."""
        await websocket.accept()
        self.active_connections[call_id] = websocket
        record_ws_connection_open(call_id)
        logger.info("websocket_connected", call_id=call_id)

        # Ensure Redis client is ready and start result listener
        await self.initialize_redis()
        # Register decoder lifecycle
        _decoder_manager.acquire(call_id)
        task = asyncio.create_task(self._listen_for_results(call_id))
        self.connection_tasks[call_id] = task

    async def disconnect(self, call_id: str) -> None:
        """Clean up resources for a disconnected WebSocket."""
        # Cancel listener task
        if call_id in self.connection_tasks:
            task = self.connection_tasks.pop(call_id)
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close pubsub client
        if call_id in self.redis_pubsub_clients:
            pubsub = self.redis_pubsub_clients.pop(call_id)
            await pubsub.unsubscribe()
            await pubsub.aclose()  # type: ignore[func-returns-value]

        # Release decoder lifecycle tracking
        _decoder_manager.release(call_id)

        ws = self.active_connections.pop(call_id, None)
        if ws is not None:
            record_ws_connection_close(call_id)
            logger.info("websocket_disconnected", call_id=call_id)

    async def send_message(self, call_id: str, message: dict) -> None:
        """Send a JSON-serializable message to the WebSocket client."""
        websocket = self.active_connections.get(call_id)
        if websocket is None:
            return
        try:
            await websocket.send_json(message)
        except Exception:
            # On send failure, drop the connection
            await self.disconnect(call_id)

    async def send_audio(self, call_id: str, audio: AudioMessage) -> None:
        """Send an audio message to the WebSocket client."""
        await self.send_message(call_id, audio.model_dump())

    async def send_error(
        self,
        call_id: str | None,
        text_id: str | None,
        error: str,
    ) -> None:
        """Send an error message to the WebSocket client."""
        # When call_id is missing we cannot route to a connection; best-effort only.
        if call_id is None:
            return
        msg = ErrorMessage(call_id=call_id, text_id=text_id, error=error)
        await self.send_message(call_id, msg.model_dump())

    async def _publish_job_to_queue(self, req: SynthesizeRequest) -> None:
        """Publish a TTS job to the Redis queue."""
        if self.redis_client is None:
            await self.initialize_redis()

        payload = {
            "call_id": req.call_id,
            "text_id": req.text_id,
            "text": req.text,
            "published_at": time.time(),
        }

        await self.redis_client.rpush(  # type: ignore[func-returns-value]
            settings.redis.tts_queue_name,
            json.dumps(payload),
        )
        logger.debug(
            "job_published_to_queue",
            call_id=req.call_id,
            text_id=req.text_id,
        )

    async def _listen_for_results(self, call_id: str) -> None:
        """Listen for synthesis results from Redis Pub/Sub and forward to client.

        Two paths depending on settings.decoder.enabled:

        • enabled=True  → subscribe to flowtts:decoded:{call_id}
                           The DecoderWorker (decoder/decoder.py) has already
                           decoded audio_tokens → WAV; we just forward the result.

        • enabled=False → subscribe to flowtts:audio:{call_id}
                           Forward raw audio_tokens to the client (client decodes).
        """
        try:
            if self.redis_client is None:
                await self.initialize_redis()

            pubsub = self.redis_client.pubsub()  # type: ignore[assignment]
            self.redis_pubsub_clients[call_id] = pubsub

            if settings.decoder.enabled:
                channel = f"{settings.redis.decoded_channel_prefix}:{call_id}"
            else:
                channel = f"{settings.redis.results_channel_prefix}:{call_id}"

            await pubsub.subscribe(channel)
            logger.info("subscribed_to_results_channel", call_id=call_id, channel=channel)

            async for message in pubsub.listen():  # type: ignore[assignment]
                if message["type"] != "message":
                    continue

                try:
                    data = json.loads(message["data"])
                    text_id = data.get("text_id", "")
                    is_final = data.get("is_final", True)
                    llm_s = data.get("llm_s")

                    if settings.decoder.enabled:
                        # --- Decoded path: DecoderWorker already produced WAV ---
                        audio_b64 = data.get("audio_base64", "")
                        sample_rate = data.get("sample_rate", settings.decoder.sample_rate)
                        decode_s = data.get("decode_s")

                        if not audio_b64:
                            logger.warning(
                                "empty_decoded_audio_received",
                                call_id=call_id,
                                text_id=text_id,
                            )
                            await self.send_error(call_id, text_id, "Decoder returned empty audio")
                            continue

                        if decode_s is not None:
                            record_decode_latency(call_id, decode_s)

                        resp = AudioMessage(
                            call_id=call_id,
                            text_id=text_id,
                            audio_base64=audio_b64,
                            sample_rate=sample_rate,
                            is_final=is_final,
                            llm_s=llm_s,
                            decode_s=decode_s,
                        )
                        logger.debug(
                            "result_forwarded_to_client",
                            call_id=call_id,
                            text_id=text_id,
                            is_final=is_final,
                            decode_s=decode_s,
                        )
                    else:
                        # --- No-decode path: forward raw LLM tokens directly ---
                        audio_tokens = data.get("audio_tokens", "")
                        if not audio_tokens:
                            logger.warning(
                                "empty_audio_tokens_received",
                                call_id=call_id,
                                text_id=text_id,
                            )
                            await self.send_error(call_id, text_id, "Synthesis returned empty audio tokens")
                            continue

                        resp = AudioMessage(
                            call_id=call_id,
                            text_id=text_id,
                            audio_tokens=audio_tokens,
                            is_final=is_final,
                            llm_s=llm_s,
                        )
                        logger.debug(
                            "result_forwarded_to_client_no_decode",
                            call_id=call_id,
                            text_id=text_id,
                            is_final=is_final,
                        )

                    await self.send_audio(call_id, resp)
                except Exception as e:  # noqa: BLE001
                    logger.error("result_processing_failed", call_id=call_id, error=str(e))

        except asyncio.CancelledError:
            logger.info("result_listener_cancelled", call_id=call_id)
        except Exception as e:  # noqa: BLE001
            logger.error("result_listener_failed", call_id=call_id, error=str(e))


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

                # Cross-validate: call_id in URL must match call_id in body
                if req.call_id != call_id:
                    raise ValueError(
                        f"call_id mismatch: URL has '{call_id}', body has '{req.call_id}'"
                    )
            except Exception as e:  # noqa: BLE001
                await manager.send_error(call_id, None, f"Bad request: {e}")
                continue

            try:
                # Publish a TTS job to Redis; a separate worker+decoder pipeline
                # will produce audio_tokens and publish results to audio:{call_id},
                # which _listen_for_results will forward to the client.
                await manager._publish_job_to_queue(req)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "synthesis_enqueue_failed",
                    call_id=call_id,
                    text_id=req.text_id,
                    error=str(e),
                )
                await manager.send_error(call_id, req.text_id, f"Synthesis enqueue failed: {e}")

    except WebSocketDisconnect:
        logger.info("websocket_disconnect", call_id=call_id)
    finally:
        await manager.disconnect(call_id)
