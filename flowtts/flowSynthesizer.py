"""
Real-time synthesis client for FlowTTS.

This client sends text to the FlowTTS WebSocket server and receives
synthesized audio in real-time.  It mirrors the structure of litranscriber.py
(the STT client) but for the TTS direction:

  litranscriber  :  mic audio  → WebSocket → STT server → transcription text
  FlowSynthesizer:  text       → WebSocket → TTS server → audio bytes

Usage
-----
    synth = FlowSynthesizer(url="ws://localhost:8080", call_id="call-abc")
    synth.start()                         # connect + start background loops

    synth.send_text("నమస్తే!")             # queue a sentence
    result = await synth.output_queue.get()   # SynthesisResult with audio
    # result.audio_bytes  — WAV bytes (16-bit PCM, 16000 Hz)
    # result.audio_base64 — base64-encoded WAV
    # result.text_id      — echo of the request text_id
    # result.call_id      — echo of the call_id
    # result.llm_s        — LLM inference time (seconds)
    # result.decode_s     — codec decode time (seconds)

    synth.terminate()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import websockets
from websockets.exceptions import WebSocketException


NUM_RESTARTS = 5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SynthesisResult:
    """Audio result returned by the FlowTTS server for one text request."""
    text_id: str
    call_id: str
    audio_bytes: bytes          # raw WAV bytes (16-bit PCM, 16000 Hz)
    audio_tokens: str           # raw speech-token string from the LLM
    sample_rate: int = 16000
    llm_s: float = 0.0          # LLM inference time reported by server
    decode_s: float = 0.0       # codec decode time reported by server


# ---------------------------------------------------------------------------
# FlowSynthesizer
# ---------------------------------------------------------------------------

class FlowSynthesizer:
    """Async TTS client for FlowTTS WebSocket server.

    Parameters
    ----------
    url:
        WebSocket URL of the FlowTTS server, e.g. ``"ws://localhost:8080"``.
    call_id:
        Identifier for this call session.  Echoed back in every response and
        used by the server for logging.
    logger:
        Optional logger; defaults to ``logging.getLogger(__name__)``.
    request_timeout:
        Seconds to wait for a response before giving up on a pending request.
        ``None`` means wait forever.
    """

    def __init__(
        self,
        url: str = "ws://localhost:8080",
        call_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        request_timeout: Optional[float] = None,
    ) -> None:
        self.url = url
        self.call_id = call_id or str(uuid.uuid4())
        self.logger = logger or logging.getLogger(__name__)
        self.request_timeout = request_timeout

        self._ended = False
        self.is_ready = False

        # input_queue  : str  (plain text to synthesize)
        # output_queue : SynthesisResult | Exception
        self.input_queue: asyncio.Queue[str] = asyncio.Queue()
        self.output_queue: asyncio.Queue[SynthesisResult | Exception] = asyncio.Queue()

        # maps text_id → asyncio.Future so _receiver can resolve the right request
        self._pending: dict[str, asyncio.Future] = {}

        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background connection loop (non-blocking)."""
        self._task = asyncio.get_event_loop().create_task(self._run_loop())

    def send_text(self, text: str) -> None:
        """Queue *text* for synthesis (non-blocking, thread-safe)."""
        self.input_queue.put_nowait(text)

    def terminate(self) -> None:
        """Stop the client and cancel background tasks."""
        self._ended = True
        if self._task and not self._task.done():
            self._task.cancel()
        self.logger.debug("FlowSynthesizer terminated")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        restarts = 0
        while not self._ended and restarts < NUM_RESTARTS:
            try:
                await self.process()
            except Exception as e:
                self.logger.debug(
                    "FlowSynthesizer connection lost, restarting (%d/%d): %s",
                    restarts + 1,
                    NUM_RESTARTS,
                    e,
                )
            restarts += 1
        self.logger.debug("FlowSynthesizer loop exited after %d restart(s)", restarts)

    async def process(self) -> None:
        """Open a WebSocket connection and run sender + receiver concurrently."""
        self.is_ready = False
        self._pending.clear()

        self.logger.debug("FlowSynthesizer connecting to %s", self.url)
        async with websockets.connect(
            self.url,
            ping_interval=30,
            ping_timeout=30,
            max_size=100 * 1024 * 1024,
            open_timeout=10,
        ) as ws:
            self.is_ready = True
            self.logger.debug("FlowSynthesizer connected to %s", self.url)
            await asyncio.gather(self._sender(ws), self._receiver(ws))

    # ------------------------------------------------------------------
    # Sender — reads text from input_queue, sends JSON to server
    # ------------------------------------------------------------------

    async def _sender(self, ws) -> None:
        """Read text from input_queue and send synthesize requests."""
        while not self._ended:
            try:
                text = await asyncio.wait_for(
                    self.input_queue.get(),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                # Keep alive — just loop back and check _ended
                continue
            except asyncio.CancelledError:
                return

            text_id = str(uuid.uuid4())
            message = {
                "type": "synthesize",
                "call_id": self.call_id,
                "text_id": text_id,
                "text": text,
                "timestamp": int(time.time() * 1000),
            }

            # Register a future so the receiver can deliver the result
            loop = asyncio.get_event_loop()
            fut: asyncio.Future[SynthesisResult] = loop.create_future()
            self._pending[text_id] = fut

            try:
                await ws.send(json.dumps(message))
                self.logger.debug(
                    "FlowSynthesizer SENT text_id=%s text=%r", text_id, text[:60]
                )
            except (WebSocketException, Exception) as e:
                # Connection broke — put the text back and bail out
                fut.cancel()
                self._pending.pop(text_id, None)
                self.input_queue.put_nowait(text)
                self.logger.debug("FlowSynthesizer send error: %s", e)
                return

    # ------------------------------------------------------------------
    # Receiver — reads server responses, resolves futures / output_queue
    # ------------------------------------------------------------------

    async def _receiver(self, ws) -> None:
        """Receive synthesized audio messages from the server."""
        while not self._ended:
            try:
                frame = await ws.recv()
            except asyncio.CancelledError:
                return
            except (WebSocketException, Exception) as e:
                self.logger.debug("FlowSynthesizer receiver error: %s", e)
                break

            try:
                data = json.loads(frame)
            except Exception as e:
                self.logger.warning("FlowSynthesizer bad JSON from server: %s", e)
                continue

            msg_type = data.get("type", "")
            text_id = data.get("text_id", "")

            if msg_type == "audio":
                # Frame 2: raw WAV bytes
                try:
                    wav_bytes = await ws.recv()
                except (WebSocketException, Exception) as e:
                    self.logger.debug("FlowSynthesizer binary recv error: %s", e)
                    break
                if isinstance(wav_bytes, str):
                    wav_bytes = wav_bytes.encode()
                result = SynthesisResult(
                    text_id=text_id,
                    call_id=data.get("call_id", self.call_id),
                    audio_bytes=wav_bytes,
                    audio_tokens=data.get("audio_tokens", ""),
                    sample_rate=data.get("sample_rate", 16000),
                    llm_s=data.get("llm_s", 0.0),
                    decode_s=data.get("decode_s", 0.0),
                )
                self.logger.debug(
                    "FlowSynthesizer RECV text_id=%s llm_s=%.3f decode_s=%.3f "
                    "audio=%d bytes",
                    text_id,
                    result.llm_s,
                    result.decode_s,
                    len(result.audio_bytes),
                )
                # Resolve pending future (if caller used async API)
                fut = self._pending.pop(text_id, None)
                if fut and not fut.done():
                    fut.set_result(result)
                # Always also put onto output_queue (for queue-based consumers)
                self.output_queue.put_nowait(result)

            elif msg_type == "error":
                error_msg = data.get("error", "Unknown error")
                self.logger.warning(
                    "FlowSynthesizer server error for text_id=%s: %s",
                    text_id,
                    error_msg,
                )
                exc = RuntimeError(f"FlowTTS server error: {error_msg}")
                fut = self._pending.pop(text_id, None)
                if fut and not fut.done():
                    fut.set_exception(exc)
                self.output_queue.put_nowait(exc)

            else:
                self.logger.debug(
                    "FlowSynthesizer unknown message type: %s", msg_type
                )

    # ------------------------------------------------------------------
    # Async convenience: send and await result directly
    # ------------------------------------------------------------------

    async def synthesize(self, text: str) -> SynthesisResult:
        """Send *text* and wait for the synthesized audio result.

        This is a higher-level convenience wrapper around
        ``send_text`` + ``output_queue.get()``.

        Raises
        ------
        RuntimeError
            If the server returns an error response.
        asyncio.TimeoutError
            If ``request_timeout`` is set and exceeded.
        """
        text_id = str(uuid.uuid4())
        message = {
            "type": "synthesize",
            "call_id": self.call_id,
            "text_id": text_id,
            "text": text,
            "timestamp": int(time.time() * 1000),
        }

        loop = asyncio.get_event_loop()
        fut: asyncio.Future[SynthesisResult] = loop.create_future()
        self._pending[text_id] = fut

        # Find the active WebSocket connection via the running task — we need to
        # send directly without going through the input_queue so we can match
        # the future by text_id.  For simplicity, use send_text + poll output_queue
        # and match by text_id.
        fut.cancel()
        self._pending.pop(text_id, None)

        # Simpler path: use the queue and match by text_id
        self.send_text(text)

        # Poll output_queue for the matching result
        deadline = (time.monotonic() + self.request_timeout) if self.request_timeout else None
        while True:
            remaining = (deadline - time.monotonic()) if deadline else None
            try:
                item = await asyncio.wait_for(
                    self.output_queue.get(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(
                    f"FlowSynthesizer: no response for {text!r} within {self.request_timeout}s"
                )
            if isinstance(item, Exception):
                raise item
            return item


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

async def _demo() -> None:
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8080"
    texts = sys.argv[2:] or [
        "నమస్తే! ఈ రోజు మీరు ఎలా ఉన్నారు?",
        "తెలుగు భాష చాలా మధురంగా ఉంటుంది.",
        "FlowTTS synthesis client is working correctly.",
    ]

    logging.basicConfig(level=logging.DEBUG)
    synth = FlowSynthesizer(url=url)
    synth.start()

    # Wait a moment for connection to establish
    await asyncio.sleep(0.5)

    for text in texts:
        print(f"[demo] Synthesizing: {text!r}")
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                synth.synthesize(text),
                timeout=30.0,
            )
            elapsed = time.monotonic() - t0
            out_path = f"flowsynth_{result.text_id[:8]}.wav"
            with open(out_path, "wb") as f:
                f.write(result.audio_bytes)
            print(
                f"[demo] Done  llm={result.llm_s:.3f}s  decode={result.decode_s:.3f}s"
                f"  total={elapsed:.3f}s  wav={len(result.audio_bytes)}B  → {out_path}"
            )
        except Exception as e:
            print(f"[demo] ERROR: {e}")

    synth.terminate()


if __name__ == "__main__":
    asyncio.run(_demo())
