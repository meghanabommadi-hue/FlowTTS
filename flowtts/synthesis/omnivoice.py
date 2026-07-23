"""Pipeline position: SYNTHESIS — OmniVoice backend.

Implements BaseSynthesizer for the OmniVoice model family (k2-fsa/OmniVoice),
vendored from the omnivoice_scaled project.

OmniVoice's own Python dependencies (transformers>=5.3.0) conflict with the
transformers==4.57.3 pin sglang/Mira need in this same venv, so the model is
NOT imported in-process. Instead this module spawns omnivoice_scaled's own
FastAPI server (src/model.py) as a child process running under ITS OWN
virtualenv (created by setup_omni.sh), and talks to it over loopback HTTP.
FlowTTS owns the child process's lifecycle (start at initialize(), terminate
at shutdown()) — it is not a standalone always-on service.

Like OmniVoice's own microbatch_server.py explains: generate() is a
synchronous, non-streaming, whole-batch call with no partial output, so
synthesize_stream() below yields exactly one final SynthChunk (mirrors how
VoxCpmSynthesizer's shape is consumed by server.py's generic streaming path).

Requires:
  - omnivoice_scaled repo checked out (default: ~/omnivoice_scaled)
  - its venv set up via setup_omni.sh (default: ~/omnivoice_scaled/.venv)
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import aiohttp
import structlog

from flowtts.core.config import settings
from flowtts.synthesis.base import BaseSynthesizer, SynthChunk, SynthResult

logger = structlog.get_logger(__name__)


class OmniVoiceSynthesizer(BaseSynthesizer):
    """OmniVoice TTS backend, hosted in a FlowTTS-managed child process.

    FlowTTS launches omnivoice_scaled/src/model.py using that project's own
    venv interpreter, waits for it to report healthy on a loopback port, and
    proxies synthesize()/synthesize_stream() calls to it over HTTP. The child
    process is torn down when shutdown() is called (or the parent exits).
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._base_url: str = ""
        self._sample_rate: int = 24000

    # ------------------------------------------------------------------
    # BaseSynthesizer interface
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._proc is not None:
            return

        cfg = settings.omnivoice
        repo_dir = Path(cfg.repo_dir)
        venv_python = Path(cfg.venv_python)
        model_script = repo_dir / "src" / "model.py"

        if not venv_python.is_file():
            raise FileNotFoundError(
                f"OmniVoice venv interpreter not found: {venv_python}\n"
                f"Run setup_omni.sh first (see FLOWTTS_OMNIVOICE__VENV_PYTHON)."
            )
        if not model_script.is_file():
            raise FileNotFoundError(
                f"OmniVoice server script not found: {model_script}\n"
                f"Set FLOWTTS_OMNIVOICE__REPO_DIR to the omnivoice_scaled checkout."
            )

        self._base_url = f"http://127.0.0.1:{cfg.port}"

        env = os.environ.copy()
        env["OMNIVOICE_MODEL_ID"] = cfg.model_id
        env["OMNIVOICE_DEVICE"] = cfg.device
        env["OMNIVOICE_NUM_STEP"] = str(cfg.num_step)
        env["OMNIVOICE_MAX_BATCH_SIZE"] = str(cfg.max_batch_size)

        logger.info("omnivoice_spawning", script=str(model_script), port=cfg.port)
        self._proc = subprocess.Popen(
            [
                str(venv_python), str(model_script),
                "--host", "127.0.0.1",
                "--port", str(cfg.port),
            ],
            cwd=str(repo_dir),
            env=env,
        )

        # server.py has no synthesizer-shutdown hook today (Mira/VoxCPM release
        # GPU memory purely by process exit) — register a fallback so this
        # child process doesn't outlive an OOM-triggered sys.exit()/restart.
        atexit.register(self._terminate_child_sync)

        self._session = aiohttp.ClientSession()

        deadline = time.monotonic() + cfg.startup_timeout_s
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"omnivoice server process exited early (code={self._proc.returncode})"
                )
            try:
                async with self._session.get(f"{self._base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        if body.get("model_loaded"):
                            break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
            await asyncio.sleep(1.0)
        else:
            await self.shutdown()
            raise TimeoutError(
                f"omnivoice server did not become healthy within {cfg.startup_timeout_s}s"
            ) from last_err

        logger.info("omnivoice_ready", base_url=self._base_url)

        if cfg.warmup_sentence:
            t0 = time.monotonic()
            try:
                await self.synthesize(cfg.warmup_sentence)
                logger.info("omnivoice_warmup_done", ms=round((time.monotonic() - t0) * 1000))
            except Exception as e:
                logger.warning("omnivoice_warmup_failed", error=str(e))

    def _terminate_child_sync(self) -> None:
        """Best-effort child-process teardown; safe to call from atexit."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        self._proc = None

    async def shutdown(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._terminate_child_sync()

    def _resolve_voice(self, voice_id: str | None) -> tuple[Optional[str], Optional[str]]:
        """Return (ref_audio_path, ref_text) for the given voice_id."""
        cfg = settings.omnivoice
        if voice_id and voice_id in cfg.voices:
            return cfg.voices[voice_id]
        if voice_id:
            logger.warning("omnivoice_unknown_voice_id", voice_id=voice_id,
                           available=list(cfg.voices.keys()))
        return cfg.ref_audio, cfg.ref_audio_text

    async def _generate(self, text: str, voice_id: str | None) -> dict:
        if self._session is None:
            raise RuntimeError("OmniVoiceSynthesizer not initialized")

        ref_audio, ref_text = self._resolve_voice(voice_id)
        cfg = settings.omnivoice
        payload = {
            "text": text,
            "language": cfg.language,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
        }
        async with self._session.post(
            f"{self._base_url}/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=cfg.request_timeout_s),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def synthesize(self, text: str, voice_id: str | None = None) -> SynthResult:
        t0 = time.perf_counter()
        body = await self._generate(text, voice_id)
        wav_bytes = base64.b64decode(body["audio_b64"])
        self._sample_rate = body["sample_rate"]

        return SynthResult(
            wav_bytes=wav_bytes,
            sample_rate=self._sample_rate,
            n_tokens=0,
            llm_s=round(body.get("generate_s", 0.0), 4),
            decode_s=0.0,
        )

    async def synthesize_stream(self, text: str, voice_id: str | None = None) -> AsyncGenerator[SynthChunk, None]:
        # OmniVoice has no partial/token-by-token output (see module docstring
        # and omnivoice_scaled/src/microbatch_server.py) — one full-response
        # chunk, marked final, mirrors how VoxCPM's stream degenerates when
        # there's only one PCM chunk from the underlying engine.
        result = await self.synthesize(text, voice_id=voice_id)
        yield SynthChunk(
            wav_bytes=result.wav_bytes,
            is_final=True,
            sample_rate=result.sample_rate,
            n_tokens=0,
            meta={"llm_s": result.llm_s},
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

