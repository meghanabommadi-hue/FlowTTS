"""Pipeline position: SYNTHESIS — miotts (Indic-Mio + MioCodec) backend.

Implements BaseSynthesizer for the miotts project (SPRINGLab/Indic-Mio LLM +
MioCodec audio decoder, see ~/miotts). miotts's own checkout splits into two
venvs (.venv_vllm for vllm==0.8.5+transformers==4.51.3 on Python 3.10, .venv
for miocodec on Python >=3.12, since miocodec's package metadata hard-requires
Python >=3.12). FlowTTS does NOT reuse either venv -- it has its own single
unified venv, .venv_mio (see setup_mio.sh), verified to actually run
vllm+transformers+miocodec together in one Python 3.12 interpreter with no
version conflicts. See core/config.py's MiottsSettings docstring for the
verification details.

This backend still spawns TWO child processes (not one) -- both launched
from the same .venv_mio interpreter -- mirroring miotts's own run.sh, which
keeps the LLM server and codec server separate on purpose (independent GPU
residency / restart / latency isolation), not because of a Python-version
conflict:
  - vLLM server   (.venv_mio)  — text -> speech-token generation.
  - codec server  (.venv_mio, miotts.codec_server) — speech tokens ->
    waveform via MioCodec, and voice-embedding resolution (preset/reference
    audio -> global_embedding).

MioCodec's decode() has no incremental/causal mode (confirmed against
~/miotts/miotts/model.py: forward_wave() sizes every interpolation/upsampling
step off the FULL target audio length computed from the complete token
sequence up front — decoding a token prefix changes those target sizes and
therefore the content itself, not just a boundary seam; an earlier empirical
test found max abs waveform diff ~0.66 on a ~[-1,1] scale between prefix- and
full-decode of the same overlapping tokens). So, like OmniVoiceSynthesizer,
synthesize_stream() below yields exactly one final SynthChunk — there is no
safe way to produce genuine partial audio chunks from this codec.

Post-decode smoothing (~/miotts/miotts/postprocess.py:smooth_glitches) is
applied here, and ONLY here — it crossfades isolated codec glitches (outlier
speech tokens rendered as sharp transients) out of the single waveform this
backend returns. No other FlowTTS backend's audio is touched by this.

Requires:
  - ~/miotts checked out, with .venv_vllm and .venv already set up per its own
    commands.md ("Faster inference with vLLM" section).
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import io
import json
import os
import re
import struct
import subprocess
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import aiohttp
import numpy as np
import structlog

from flowtts.core.config import settings
from flowtts.synthesis.base import BaseSynthesizer, SynthChunk, SynthResult

logger = structlog.get_logger(__name__)

SPEECH_TOKEN_RE = re.compile(r"<\|s_(\d+)\|>")

# miotts/postprocess.py itself depends only on numpy, but the miotts PACKAGE's
# __init__.py eagerly imports synthesizer.py -> transformers/torch, which we
# do NOT want pulled into FlowTTS's own process (same reason the vLLM/codec
# servers are proxied over HTTP instead of imported: separate, conflicting
# venvs). So load postprocess.py directly by file path by importlib, bypassing
# the miotts package's __init__.py entirely.
import importlib.util as _importlib_util


def _load_smooth_glitches():
    postprocess_path = Path(settings.miotts.repo_dir) / "miotts" / "postprocess.py"
    spec = _importlib_util.spec_from_file_location("_miotts_postprocess", postprocess_path)
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.smooth_glitches


smooth_glitches = _load_smooth_glitches()


def _pcm_to_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
    """float32 PCM array (~[-1,1]) -> 16-bit WAV bytes."""
    pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    data_size = len(pcm_int16)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_int16)
    return buf.getvalue()


class MiottsSynthesizer(BaseSynthesizer):
    """miotts TTS backend, hosted in two FlowTTS-managed child processes
    (vLLM server + codec server), proxied over HTTP — see module docstring."""

    def __init__(self) -> None:
        self._vllm_proc: Optional[subprocess.Popen] = None
        self._codec_proc: Optional[subprocess.Popen] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._vllm_base_url: str = ""
        self._codec_base_url: str = ""
        self._sample_rate: int = 44100

    # ------------------------------------------------------------------
    # BaseSynthesizer interface
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._vllm_proc is not None:
            return

        cfg = settings.miotts
        repo_dir = Path(cfg.repo_dir)
        venv_python = Path(cfg.venv_python)

        if not venv_python.is_file():
            raise FileNotFoundError(
                f"miotts venv interpreter not found: {venv_python}\n"
                f"Run ./setup_mio.sh first, or point "
                f"FLOWTTS_MIOTTS__VENV_PYTHON elsewhere."
            )

        self._vllm_base_url = f"http://127.0.0.1:{cfg.vllm_port}"
        self._codec_base_url = f"http://127.0.0.1:{cfg.codec_port}"
        self._session = aiohttp.ClientSession()

        await self._spawn_vllm(repo_dir, venv_python, cfg)
        await self._spawn_codec(repo_dir, venv_python, cfg)

        atexit.register(self._terminate_children_sync)

        logger.info(
            "miotts_ready", vllm_base_url=self._vllm_base_url, codec_base_url=self._codec_base_url
        )

        if cfg.warmup_sentence:
            t0 = time.monotonic()
            try:
                await self.synthesize(cfg.warmup_sentence)
                logger.info("miotts_warmup_done", ms=round((time.monotonic() - t0) * 1000))
            except Exception as e:
                logger.warning("miotts_warmup_failed", error=str(e))

    async def _spawn_vllm(self, repo_dir: Path, venv_python: Path, cfg) -> None:
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        logger.info("miotts_vllm_spawning", port=cfg.vllm_port)
        self._vllm_proc = subprocess.Popen(
            [
                str(venv_python), "-m", "vllm", "serve", cfg.model_name,
                "--max-model-len", str(cfg.max_model_len),
                "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
                "--port", str(cfg.vllm_port),
            ],
            cwd=str(repo_dir),
            env=env,
        )
        await self._wait_healthy(
            f"{self._vllm_base_url}/v1/models", self._vllm_proc, cfg.startup_timeout_s, "vllm"
        )
        logger.info("miotts_vllm_ready", port=cfg.vllm_port)

    async def _spawn_codec(self, repo_dir: Path, venv_python: Path, cfg) -> None:
        logger.info("miotts_codec_spawning", port=cfg.codec_port)
        self._codec_proc = subprocess.Popen(
            [str(venv_python), "-m", "miotts.codec_server", "--port", str(cfg.codec_port)],
            cwd=str(repo_dir),
            env=os.environ.copy(),
        )
        await self._wait_healthy(
            f"{self._codec_base_url}/health", self._codec_proc, cfg.startup_timeout_s, "codec"
        )
        logger.info("miotts_codec_ready", port=cfg.codec_port)

    async def _wait_healthy(
        self, health_url: str, proc: subprocess.Popen, timeout_s: float, name: str
    ) -> None:
        deadline = time.monotonic() + timeout_s
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"miotts {name} server exited early (code={proc.returncode})")
            try:
                async with self._session.get(
                    health_url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
            await asyncio.sleep(2.0)
        await self.shutdown()
        raise TimeoutError(f"miotts {name} server did not become healthy within {timeout_s}s") from last_err

    def _terminate_children_sync(self) -> None:
        """Best-effort child-process teardown; safe to call from atexit."""
        for proc in (self._vllm_proc, self._codec_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        self._vllm_proc = None
        self._codec_proc = None

    async def shutdown(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._terminate_children_sync()

    def _resolve_voice(self, voice_id: str | None) -> tuple[Optional[str], Optional[str]]:
        """Return (voice_preset, reference_audio) for the given voice_id."""
        cfg = settings.miotts
        if voice_id and voice_id in cfg.voices:
            return cfg.voices[voice_id]
        if voice_id:
            logger.warning("miotts_unknown_voice_id", voice_id=voice_id, available=list(cfg.voices.keys()))
        return cfg.voice_preset, cfg.reference_audio

    async def _stream_speech_tokens(self, text: str, timeout_s: float):
        """POSTs a streaming chat completion to miotts's vLLM server, yielding
        each SSE delta's text as it arrives (mirrors miotts's own
        vllm_synthesizer.py:_stream_speech_tokens, async instead of sync)."""
        cfg = settings.miotts
        payload = {
            "model": cfg.model_name,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "repetition_penalty": cfg.repetition_penalty,
            "stream": True,
        }
        async with self._session.post(
            f"{self._vllm_base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

    async def _decode(self, audio_codes: list[int], voice_id: str | None, timeout_s: float) -> np.ndarray:
        voice_preset, reference_audio = self._resolve_voice(voice_id)
        async with self._session.post(
            f"{self._codec_base_url}/resolve_embedding",
            json={"voice_preset": voice_preset, "reference_audio": reference_audio},
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            resp.raise_for_status()
            embedding = (await resp.json())["embedding"]

        async with self._session.post(
            f"{self._codec_base_url}/decode",
            json={"audio_codes": audio_codes, "embedding": embedding},
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            resp.raise_for_status()
            body = await resp.json()
            self._sample_rate = body["sample_rate"]
            waveform_bytes = base64.b64decode(body["waveform_b64"])
            return np.frombuffer(waveform_bytes, dtype=np.float32).copy()

    async def synthesize(self, text: str, voice_id: str | None = None) -> SynthResult:
        if self._session is None:
            raise RuntimeError("MiottsSynthesizer not initialized")

        cfg = settings.miotts
        t0 = time.perf_counter()
        content = ""
        async for chunk_text in self._stream_speech_tokens(text, cfg.request_timeout_s):
            content += chunk_text
        llm_s = time.perf_counter() - t0

        audio_codes = [int(m) for m in SPEECH_TOKEN_RE.findall(content)]
        if not audio_codes:
            raise RuntimeError(
                "No speech tokens were generated for the given text; try again or "
                "increase miotts.max_new_tokens."
            )

        t1 = time.perf_counter()
        waveform = await self._decode(audio_codes, voice_id, cfg.request_timeout_s)

        if cfg.smooth_glitches:
            waveform = smooth_glitches(waveform, self._sample_rate)
        decode_s = time.perf_counter() - t1

        wav_bytes = _pcm_to_wav(waveform, self._sample_rate)

        return SynthResult(
            wav_bytes=wav_bytes,
            sample_rate=self._sample_rate,
            n_tokens=len(audio_codes),
            llm_s=round(llm_s, 4),
            decode_s=round(decode_s, 4),
        )

    async def synthesize_stream(self, text: str, voice_id: str | None = None) -> AsyncGenerator[SynthChunk, None]:
        # MioCodec has no partial/incremental decode (see module docstring) —
        # one full-response chunk, marked final, mirrors how OmniVoiceSynthesizer's
        # stream degenerates when the underlying engine has no chunked output.
        result = await self.synthesize(text, voice_id=voice_id)
        yield SynthChunk(
            wav_bytes=result.wav_bytes,
            is_final=True,
            sample_rate=result.sample_rate,
            n_tokens=result.n_tokens,
            meta={"llm_s": result.llm_s, "decode_s": result.decode_s},
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
