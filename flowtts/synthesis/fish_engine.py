"""Pipeline position: SYNTHESIS ENGINE — text → 24 kHz waveform (backend client).

Role in pipeline:
  A GPU-less async client for the **Fish Audio S2 Pro** model served by
  **sglang-omni** (`sgl-omni serve`, OpenAI-compatible ``POST /v1/audio/speech``).
  All GPU work — Dual-AR token generation, EVA-GAN codec decode, continuous
  batching, RadixAttention prefix caching — happens in the sglang backend. This
  process only builds requests, forwards streamed PCM, and manages the voice
  registry.

  server.py → engine.synthesize[/_stream](text, voice_id, speed, language)
            → POST backend /v1/audio/speech (response_format=pcm)
            → int16 PCM → np.float32 waveform(24k) → (resample) → WebSocket

Why no in-process batcher (unlike the old OmniVoice engine):
  S2 Pro is a standard autoregressive LLM, so sglang's continuous batching +
  paged KV cache do the throughput multiplication far better than a request-level
  queue could. Concurrent WebSocket requests simply issue concurrent HTTP calls;
  sglang coalesces them server-side. Streaming is true token-by-token AR streaming.

Voice cloning:
  A voice is a reference clip + transcript (see voices/registry.py). We send it as
  ``references=[{"audio_path", "text"}]``; sglang encodes it into VQ codes once and
  caches the KV, so reused voices hit the prefix cache (~86–90%).
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path

import numpy as np
import structlog

from flowtts.core.config import settings
from flowtts.voices.registry import VoiceRegistry
from flowtts.voices.store import manifest_path, save_voice

logger = structlog.get_logger(__name__)

_MIME_BY_EXT = {
    ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4", ".ogg": "audio/ogg",
}


class FishSpeechEngine:
    """Async client to the sglang-omni Fish S2 Pro backend + voice registry."""

    # AR backend streams one contiguous PCM stream → callers must NOT crossfade
    # between streamed chunks (only a final fade-out is appropriate).
    continuous_stream: bool = True

    def __init__(self) -> None:
        cfg = settings.fish
        self._cfg = cfg
        self.registry: VoiceRegistry | None = None
        self.sampling_rate: int = int(cfg.sample_rate)
        self.engine_info: dict = {}
        self._session = None  # aiohttp.ClientSession, created in initialize()
        self._speech_url = cfg.backend_url.rstrip("/") + cfg.speech_path
        self._health_url = cfg.backend_url.rstrip("/") + cfg.health_path

    # ------------------------------------------------------------------ load
    async def initialize(self) -> None:
        if self._session is not None:
            return
        import aiohttp

        cfg = self._cfg
        # Voice registry — cheap (loads tiny json manifests, no GPU).
        self.registry = VoiceRegistry(settings.voices.voices_dir, settings.voices.default_voice)

        timeout = aiohttp.ClientTimeout(total=cfg.request_timeout_s, connect=cfg.connect_timeout_s)
        self._session = aiohttp.ClientSession(timeout=timeout)

        logger.info("fish_backend_connecting", url=cfg.backend_url, model=cfg.model)
        await self._wait_for_backend()

        self.engine_info = {
            "backend_url": cfg.backend_url,
            "model": cfg.model,
            "response_format": cfg.response_format,
            "reference_mode": cfg.reference_mode,
            "sampling_rate": self.sampling_rate,
            "voices": self.registry.aliases() if self.registry else [],
        }
        logger.info("fish_engine_ready", **{k: self.engine_info[k] for k in
                    ("backend_url", "model", "sampling_rate", "voices")})

        if cfg.warmup:
            await self._warmup()

    async def _wait_for_backend(self, attempts: int = 30, delay_s: float = 2.0) -> None:
        """Probe the backend /health until it responds (backend may still be booting)."""
        last_err: Exception | None = None
        for i in range(attempts):
            try:
                async with self._session.get(self._health_url) as resp:
                    if resp.status == 200:
                        logger.info("fish_backend_healthy", attempt=i + 1)
                        return
                    last_err = RuntimeError(f"health status {resp.status}")
            except Exception as e:  # noqa: BLE001
                last_err = e
            await asyncio.sleep(delay_s)
        # Don't hard-fail — surface a clear warning and let requests error later if
        # the backend never comes up (keeps the gateway serving /health as 503).
        logger.warning("fish_backend_unreachable", url=self._health_url, error=str(last_err))

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------ request build
    def _reference_field(self, voice_id: str | None) -> list[dict] | None:
        """Build the `references` list for a voice_id, or None to use the default voice."""
        if self.registry is None:
            return None
        ref = self.registry.reference(voice_id)
        if ref is None:
            return None
        audio_path, ref_text = ref
        if self._cfg.reference_mode == "base64":
            audio_uri = _to_data_uri(audio_path)
        else:
            audio_uri = _map_backend_path(audio_path, self._cfg.backend_voices_dir,
                                          settings.voices.voices_dir)
        return [{"audio_path": audio_uri, "text": ref_text}]

    def _build_payload(
        self, text: str, voice_id: str | None, speed: float | None,
        language: str | None, stream: bool,
    ) -> dict:
        cfg = self._cfg
        payload: dict = {
            "input": text,
            "model": cfg.model,
            "response_format": cfg.response_format,
            "stream": stream,
            "speed": float(speed) if speed is not None else cfg.speed,
        }

        # Language precedence: explicit request > voice's preferred > global default.
        if language is not None:
            lang = language
        elif self.registry is not None and self.registry.language(voice_id):
            lang = self.registry.language(voice_id)
        else:
            lang = settings.voices.default_language
        if lang:
            payload["language"] = lang

        references = self._reference_field(voice_id)
        if references is not None:
            payload["references"] = references
        else:
            payload["voice"] = "default"

        # Optional generation knobs — only sent when explicitly configured.
        for key, val in (
            ("temperature", cfg.temperature),
            ("top_p", cfg.top_p),
            ("top_k", cfg.top_k),
            ("repetition_penalty", cfg.repetition_penalty),
            ("max_new_tokens", cfg.max_new_tokens),
            ("initial_codec_chunk_frames", cfg.initial_codec_chunk_frames),
        ):
            if val is not None:
                payload[key] = val
        return payload

    # ------------------------------------------------------------------ public API
    async def synthesize(
        self, text: str, *, voice_id: str | None = None,
        speed: float | None = None, language: str | None = None,
    ) -> np.ndarray:
        """Return the full 24 kHz float32 waveform for *text* (non-streaming)."""
        if self._session is None:
            raise RuntimeError("FishSpeechEngine not initialized")
        payload = self._build_payload(text, voice_id, speed, language, stream=False)
        async with self._session.post(self._speech_url, json=payload) as resp:
            body = await resp.read()
            if resp.status != 200:
                raise RuntimeError(f"backend {resp.status}: {body[:200]!r}")
            return _decode_audio_bytes(body)

    async def synthesize_stream(
        self, text: str, *, voice_id: str | None = None,
        speed: float | None = None, language: str | None = None,
    ):
        """Yield (chunk_index, waveform_float32, is_final) as PCM streams from the backend.

        The backend streams one contiguous 16-bit PCM stream; we decode byte
        fragments into float32 (buffering any odd trailing byte across fragments)
        and use a one-chunk look-ahead so the FINAL yielded chunk is flagged
        is_final=True (letting server.py fade only the true tail).
        """
        if self._session is None:
            raise RuntimeError("FishSpeechEngine not initialized")
        payload = self._build_payload(text, voice_id, speed, language, stream=True)

        async with self._session.post(self._speech_url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.read()
                raise RuntimeError(f"backend {resp.status}: {body[:200]!r}")

            idx = 0
            leftover = b""
            pending: np.ndarray | None = None  # one-chunk look-ahead for is_final
            async for frag in resp.content.iter_any():
                if not frag:
                    continue
                buf = leftover + frag
                n_even = len(buf) - (len(buf) % 2)
                if n_even == 0:
                    leftover = buf
                    continue
                leftover = buf[n_even:]
                wav = _pcm_bytes_to_float32(buf[:n_even])
                if wav.size == 0:
                    continue
                if pending is not None:
                    yield idx, pending, False
                    idx += 1
                pending = wav

            # Flush the final buffered chunk (marked is_final).
            if pending is not None:
                yield idx, pending, True
            elif idx == 0:
                # Nothing streamed — emit an empty final so callers finalize cleanly.
                yield 0, np.zeros(0, dtype=np.float32), True

    async def create_voice(
        self, voice_id: str, audio_path: str, ref_text: str, language: str | None = None,
    ) -> dict:
        """Clone a voice: convert the clip to mono WAV in voices_dir + write its manifest.

        No GPU/model work — the sglang backend encodes the reference on first use.
        The voice is usable immediately (no restart). If the voices_dir is a shared
        volume the backend also mounts, the reference resolves there directly.
        """
        if not ref_text or not ref_text.strip():
            raise ValueError("ref_text is required for voice cloning")
        if self.registry is None:
            raise RuntimeError("FishSpeechEngine not initialized")

        voices_dir = Path(settings.voices.voices_dir)
        out_wav = voices_dir / f"{voice_id}.wav"

        loop = asyncio.get_event_loop()
        sr, dur = await loop.run_in_executor(None, _to_mono_wav, str(audio_path), str(out_wav))

        save_voice(voices_dir, alias=voice_id, ref_text=ref_text,
                   audio_file=out_wav.name, language=language)
        self.registry.add(voice_id, manifest_path(voices_dir, voice_id))
        logger.info("voice_cloned", voice_id=voice_id, sr=sr, dur_s=round(dur, 2), language=language)
        return {
            "voice_id": voice_id,
            "ref_text": ref_text,
            "language": language,
            "sample_rate": sr,
            "duration_s": round(dur, 2),
            "audio_file": out_wav.name,
            "npz": str(out_wav),  # legacy key kept so existing clients that read it don't break
        }

    async def _warmup(self) -> None:
        cfg = self._cfg
        voice = settings.voices.default_voice if (self.registry and self.registry.has(settings.voices.default_voice)) else None
        logger.info("fish_warmup", voice=voice)
        t0 = time.perf_counter()
        try:
            await self.synthesize(cfg.warmup_sentence, voice_id=voice)
            logger.info("fish_warmup_done", ms=round((time.perf_counter() - t0) * 1000))
        except Exception as e:  # noqa: BLE001
            logger.warning("fish_warmup_failed", error=str(e))


# ---------------------------------------------------------------------------
# Helpers (module-level: no self, easy to unit-test)
# ---------------------------------------------------------------------------
def _pcm_bytes_to_float32(pcm: bytes) -> np.ndarray:
    """Decode raw little-endian int16 mono PCM bytes into float32 in [-1, 1)."""
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(pcm, dtype="<i2")
    return (arr.astype(np.float32) / 32768.0)


def _decode_audio_bytes(body: bytes) -> np.ndarray:
    """Decode a non-streaming response body — raw int16 PCM, or a WAV container.

    With response_format="pcm" the backend returns raw PCM, but be defensive: if it
    returns a RIFF/WAVE container, parse it with soundfile so a header isn't
    mis-read as samples (a click at the start).
    """
    if body[:4] == b"RIFF" and body[8:12] == b"WAVE":
        import io
        import soundfile as sf
        data, _sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
        return np.asarray(data, dtype=np.float32).reshape(-1)
    return _pcm_bytes_to_float32(body)


def _to_data_uri(audio_path: str) -> str:
    """Encode a reference clip as a data:audio/...;base64 URI for the backend."""
    p = Path(audio_path)
    mime = _MIME_BY_EXT.get(p.suffix.lower(), "audio/wav")
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _map_backend_path(audio_path: str, backend_voices_dir: str | None, voices_dir: str) -> str:
    """Rewrite a gateway-local clip path to the backend's mount point, if they differ."""
    if not backend_voices_dir:
        return audio_path
    vd = str(Path(voices_dir))
    ap = str(Path(audio_path))
    if ap.startswith(vd):
        return str(Path(backend_voices_dir) / Path(ap).relative_to(vd))
    return audio_path


def _to_mono_wav(src: str, dst: str) -> tuple[int, float]:
    """Read any supported audio file (via ffmpeg), downmix to mono, write a 16-bit WAV.

    Keeps the native sample rate (the sglang backend resamples internally). Returns
    (sample_rate, duration_s). Uses pydub → no heavy numba/librosa in the gateway.
    """
    from pydub import AudioSegment

    seg = AudioSegment.from_file(src).set_channels(1).set_sample_width(2)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    seg.export(dst, format="wav")
    return int(seg.frame_rate), float(len(seg) / 1000.0)
