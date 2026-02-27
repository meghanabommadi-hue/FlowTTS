"""
Real-time transcription client for LITranscriber.

This client captures audio from the microphone, streams it to the
LITranscriber WebSocket server, and displays the transcriptions in real-time.
"""

import asyncio
import json
import uuid
import logging
import time
import base64
import ssl as ssl_module
import os
from typing import Optional
import websockets
from websockets.legacy.client import WebSocketClientProtocol
import audioop
import numpy as np
import resampy

from vocode.streaming.transcriber.base_transcriber import (
    BaseAsyncTranscriber,
    Transcription,
    meter,
)
from xpertcore.models.transcriber import (
    LITranscriberConfig,
)
from xpertcore.models.audio_encoding import AudioEncoding


PUNCTUATION_TERMINATORS = [".", "!", "?"]
NUM_RESTARTS = 5


avg_latency_hist = meter.create_histogram(
    name="transcriber.deepgram.avg_latency",
    unit="seconds",
)
max_latency_hist = meter.create_histogram(
    name="transcriber.deepgram.max_latency",
    unit="seconds",
)
min_latency_hist = meter.create_histogram(
    name="transcriber.deepgram.min_latency",
    unit="seconds",
)
duration_hist = meter.create_histogram(
    name="transcriber.deepgram.duration",
    unit="seconds",
)

class LITranscriber(BaseAsyncTranscriber[LITranscriberConfig]):
    def __init__(
        self,
        transcriber_config: LITranscriberConfig,
        logger: Optional[logging.Logger] = None,
        endpointing_timeout: float = 0.8,
    ):
        super().__init__(transcriber_config)
        self._ended = False
        self.is_ready = False
        self.logger = logger or logging.getLogger(__name__)
        self.audio_cursor = 0.0
        self.call_id = None
        self.endpointing_timeout = endpointing_timeout
        self.logger.debug(f"Initialized LITranscriber with endpointing_timeout={endpointing_timeout}s")
        self.diagnostic_log_file = None

    async def _run_loop(self):
        restarts = 0
        while not self._ended and restarts < NUM_RESTARTS:
            await self.process()
            restarts += 1
            self.logger.debug(
                "LITranscripter connection died, restarting, num_restarts: %s", restarts
            )

    def send_audio(self, chunk):
        # Original chunk is 320 bytes (20ms at 8kHz)
        if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16:
            audio_np = np.frombuffer(chunk, dtype=np.int16)
            # Resample from 8kHz to 16kHz, resulting in 640 bytes (20ms at 16kHz)
            resampled_np = resampy.resample(
                audio_np, self.transcriber_config.sampling_rate, 16000
            )
            chunk = resampled_np.astype(np.int16).tobytes()

        super().send_audio(chunk)

    def terminate(self):
        self._ended = True
        if self.diagnostic_log_file:
            self.diagnostic_log_file.close()
        super().terminate()

    def get_lit_url(self):
        self.call_id = str(uuid.uuid4())
        if self.transcriber_config.diagnostic_mode:
            log_dir = "transcriber_logs"
            os.makedirs(log_dir, exist_ok=True)
            self.diagnostic_log_file = open(
                os.path.join(log_dir, f"lit_transcriber_diagnostics_{self.call_id}.json"),
                "w",
            )
        return f"{self.transcriber_config.stt_endpoint}/{self.call_id}?language={self.transcriber_config.language}"

    async def _sender(self, ws):
        """Send audio to websocket.
        
        This method accumulates 640-byte chunks (20ms at 16kHz) from the input queue
        and assembles them into 1024-byte frames (32ms at 16kHz) required by the
        server's PreciseVAD (Silero VAD model).
        """
        FRAME_SIZE = 1024  # 32ms at 16kHz, 16-bit (required by Silero VAD)
        buffer = b""
        
        while not self._ended:
            try:
                data = await asyncio.wait_for(
                    self.input_queue.get(), 
                    timeout=self.endpointing_timeout
                )
                buffer += data
            except asyncio.TimeoutError:
                if self._ended:
                    break
                
                # If the queue is empty for endpointing_timeout duration, send a flush message to the server
                await ws.send(json.dumps({"type": "flush"}))
                continue
            except asyncio.CancelledError:
                return

            # Send complete frames as they become available
            while len(buffer) >= FRAME_SIZE:
                chunk_to_send = buffer[:FRAME_SIZE]
                buffer = buffer[FRAME_SIZE:]

                num_channels = 1
                sample_width = 2
                self.audio_cursor += len(chunk_to_send) / (
                    16000 * num_channels * sample_width
                )
                
                message = {
                    "type": "audio_frame",
                    "call_id": self.call_id,
                    "sequence": self.sequence,
                    "audio": base64.b64encode(chunk_to_send).decode("utf-8"),
                    "timestamp": int(time.time() * 1000),
                }
                await ws.send(json.dumps(message))
                self.sequence += 1
        
        # Handle remaining audio in buffer when stream ends
        if len(buffer) > 0:
            self.logger.debug(f"Sending remaining audio buffer of size {len(buffer)}")
            
            # Pad to FRAME_SIZE to meet VAD requirements
            padding_needed = FRAME_SIZE - len(buffer)
            if padding_needed > 0:
                buffer += b'\x00' * padding_needed
                self.logger.debug(f"Padded buffer with {padding_needed} bytes of silence")

            num_channels = 1
            sample_width = 2
            self.audio_cursor += len(buffer) / (
                16000 * num_channels * sample_width
            )
            
            message = {
                "type": "audio_frame",
                "call_id": self.call_id,
                "sequence": self.sequence,
                "audio": base64.b64encode(buffer).decode("utf-8"),
                "timestamp": int(time.time() * 1000),
            }
            await ws.send(json.dumps(message))
            self.sequence += 1

    async def _receiver(self, ws):
        while not self._ended:
            try:
                msg = await ws.recv()
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.logger.debug(f"Got error {e} in LIT receiver, terminating")
                break

            data = json.loads(msg)
            msg_type = data.get("type", "")

            if self.diagnostic_log_file:
                self.diagnostic_log_file.write(
                    json.dumps(data, indent=4, ensure_ascii=False) + "\n"
                )

            if msg_type == "transcript":
                is_final = data.get("is_final", False)
                
                if is_final:
                    # This is a final transcript.
                    self.output_queue.put_nowait(
                        Transcription(
                            message=data["text"],
                            confidence=1.0,
                            is_final=True,
                        )
                    )
                # else:
                #     # This is an interim transcript.
                #     self.output_queue.put_nowait(
                #         Transcription(
                #             message=data["text"],
                #             confidence=data["confidence"],
                #             is_final=False,
                #         )
                #     )
            
            elif msg_type == "human_speaking":
                # Human started speaking - send interrupt signal
                self.logger.debug(
                    f"Human started speaking at timestamp: {data.get('timestamp')}"
                )
                self.output_queue.put_nowait(
                    Transcription(
                        message="--",
                        confidence=1.0,
                        is_final=False,
                    )
                )
            
            elif msg_type == "human_stopped":
                # Human stopped speaking - informational event
                self.logger.debug(
                    f"Human stopped speaking at timestamp: {data.get('timestamp')}"
                )

    async def process(self):
        self.audio_cursor = 0.0
        self.sequence = 0

        ssl_context = ssl_module.create_default_context()
        # TODO: Remove this in production
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl_module.CERT_NONE

        # async with websockets.connect(self.get_lit_url(), ssl=ssl_context) as ws:
        async with websockets.connect(self.get_lit_url()) as ws:
            await asyncio.gather(self._sender(ws), self._receiver(ws))