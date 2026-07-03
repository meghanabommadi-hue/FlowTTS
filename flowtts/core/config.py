"""Pipeline position: CONFIGURATION (read by every module at import time).

Role in pipeline:
  Single source of truth for all tunable parameters. Every pipeline stage
  imports `settings` from here rather than reading env-vars directly.

Model / serving:
  This gateway synthesizes speech with **Fish Audio S2 Pro** (`fishaudio/s2-pro`),
  served out-of-process by **sglang-omni** (`sgl-omni serve`, OpenAI-compatible
  `POST /v1/audio/speech`). S2 Pro is a Dual-AR (Qwen3-4B slow-AR + 400M fast-AR)
  model with an EVA-GAN / RVQ codec (10 codebooks, ~21 Hz, 24 kHz output). All GPU
  work lives in the sglang backend; this process is a CPU-only WebSocket proxy.

  This replaces the previous in-process k2-fsa/OmniVoice diffusion stack; the
  serving framework (WebSocket gateway, streaming protocol, metrics, WAV cache,
  voice registry, control API) is preserved.

Key sections and their pipeline consumers:
  FishSpeechSettings → synthesis/fish_engine.py  (backend URL, request + gen params)
  VoiceSettings      → voices/registry.py         (reference-clip registry + aliases)
  OutputSettings     → server.py, decoder/decoder.py (stream sample rate, encoding)
  StreamingSettings  → server.py                  (boundary fades; chunking retained
                                                   for the non-streaming Redis path)
  WebSocketSettings  → main.py / server.py        (host/port)
  RedisSettings      → worker.py, api/websockets.py (secondary Redis-backed path)

All values can be overridden via environment variables (FLOWTTS_ prefix,
nested via `__`, e.g. FLOWTTS_FISH__BACKEND_URL=http://127.0.0.1:8000).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
_HOME = Path.home()
# Repo root (…/FlowTTS) — used to resolve relative paths regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_FILES_DIR = str(_HOME / "FlowTTS/sample_files")
_VOICES_DIR = str(_HOME / "FlowTTS/voices")          # reference clips + <alias>.json manifests
_WAV_CACHE_DIR = str(_HOME / "FlowTTS/cached_data")  # sha256(text).wav cache

# Fish Audio S2 Pro's native output sample rate (24 kHz). The authoritative value
# is echoed by the backend in the `x-sample-rate` response header at runtime.
FISH_NATIVE_SR = 24000


class FishSpeechSettings(BaseModel):
    """Fish Audio S2 Pro backend (sglang-omni) connection + generation config."""

    # --- Backend connection ---
    # URL of the sglang-omni server (`sgl-omni serve ... --port 8000`). In the
    # two-service compose setup this is the service DNS name; override for local
    # dev or an external/licensed endpoint via FLOWTTS_FISH__BACKEND_URL.
    backend_url: str = "http://fish-s2pro:8000"
    model: str = "fishaudio/s2-pro"        # echoed as the OpenAI `model` field / in metrics
    speech_path: str = "/v1/audio/speech"  # OpenAI-compatible TTS endpoint
    health_path: str = "/health"           # backend liveness probe
    request_timeout_s: float = 300.0       # total per-request budget (long-form safety)
    connect_timeout_s: float = 10.0

    # --- Output wire format from the backend ---
    # "pcm" → raw 16-bit mono PCM (streaming-friendly, what we forward). "wav" is
    # only useful for debugging non-streaming calls.
    response_format: Literal["pcm", "wav"] = "pcm"
    sample_rate: int = FISH_NATIVE_SR      # native backend rate (echoed via x-sample-rate)

    # --- Voice reference passing (zero-shot cloning) ---
    # "local"  → send references=[{audio_path: <clip on the shared voices volume>}].
    #            Lowest overhead; relies on the sglang container mounting the same
    #            clip path (see backend_voices_dir for path remapping).
    # "base64" → send the clip inline as a data:audio/...;base64 URI (no shared
    #            volume needed; ~MBs per request, but SGLang's prefix cache means
    #            only the first same-voice request re-prefills).
    reference_mode: Literal["local", "base64"] = "local"
    # If the sglang container mounts the voices dir at a DIFFERENT path than this
    # gateway, set this to the backend's mount point; the gateway rewrites the
    # voices_dir prefix of each clip path accordingly. None → identical path.
    backend_voices_dir: str | None = None

    # --- Generation defaults (only sent to the backend when not None) ---
    speed: float = 1.0                     # per-request speed overrides this
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int | None = None
    # Frames buffered before the first codec decode when streaming; smaller ⇒ lower
    # TTFB, larger ⇒ smoother. None ⇒ backend default.
    initial_codec_chunk_frames: int | None = None

    # --- Warmup ---
    # Fire one synthesis at startup to prime the backend (CUDA graphs) and warm the
    # default voice's RadixAttention prefix cache before real traffic arrives.
    warmup: bool = True
    warmup_sentence: str = (
        "नमस्ते. मैं बजाज finance से बोल रही हूं, एक recorded line के माध्यम से. "
        "क्या मैं customer name से बात कर रही हूं?"
    )


class VoiceSettings(BaseModel):
    """Voice-clone registry: reference clips + transcripts addressed by alias.

    A voice is `voices_dir/<alias>.json` (ref_text, language, audio filename) beside
    its reference clip `voices_dir/<alias>.<ext>`. sglang encodes the clip into VQ
    codes on first use and caches the KV via RadixAttention.
    """

    voices_dir: str = _VOICES_DIR
    # Alias used when a request omits voice_id (must exist in voices_dir).
    default_voice: str = "priya"
    # Language passed to the backend when a request omits it (None = auto-detect).
    default_language: str | None = None


class OutputSettings(BaseModel):
    """Audio output format for the WebSocket stream.

    Fish S2 Pro is natively 24 kHz. If sample_rate < native, chunks are resampled
    before encoding. The `sample_rate` value is always echoed in the audio_chunk
    JSON header so compliant clients adapt automatically.
    """

    sample_rate: int = FISH_NATIVE_SR   # 24000 native; set 16000/8000 to resample (telephony)
    encoding: Literal["pcm_int16"] = "pcm_int16"


class StreamingSettings(BaseModel):
    """Streaming behaviour + boundary smoothing.

    Fish S2 Pro is autoregressive, so the backend streams one CONTIGUOUS PCM stream
    (no per-chunk boundaries). We therefore do NOT crossfade between streamed
    chunks; only a final fade-out is applied. The text-chunk knobs below are kept
    for the secondary (non-streaming) Redis path and are unused on the AR stream.
    """

    enabled: bool = True

    # Text-chunk caps (secondary/non-streaming path only; the AR stream ignores them).
    first_chunk_max_chars: int = 60
    chunk_max_chars: int = 160
    min_chunk_chars: int = 12

    # Boundary smoothing (samples at output rate). crossfade_samples is NOT applied
    # mid-AR-stream (see server.py: gated on synth.continuous_stream); fade_out
    # still tidies the final tail.
    crossfade_samples: int = 480       # 20 ms @ 24 kHz
    fade_out_samples: int = 240        # fade the final chunk tail


class WebSocketSettings(BaseModel):
    """Gateway WebSocket server settings."""

    host: str = "0.0.0.0"
    port: int = 8080


class Settings(BaseSettings):
    """Top-level FlowTTS settings."""

    model_config = SettingsConfigDict(env_prefix="FLOWTTS_", env_nested_delimiter="__")

    ws: WebSocketSettings = WebSocketSettings()
    fish: FishSpeechSettings = FishSpeechSettings()
    voices: VoiceSettings = VoiceSettings()
    output: OutputSettings = OutputSettings()
    streaming: StreamingSettings = StreamingSettings()

    # Directory of pre-generated WAV files named by SHA256 of raw transcript.
    # Cache hits bypass the backend entirely (huge win for repeated call-centre prompts).
    wav_cache_dir: str | None = _WAV_CACHE_DIR

    # Redis queue / pubsub configuration (secondary, Redis-backed multi-process path).
    class RedisSettings(BaseModel):
        host: str = "localhost"
        port: int = 6379
        db: int = 0
        password: str | None = None

        tts_queue_name: str = "flowtts:tts_queue"
        results_channel_prefix: str = "flowtts:audio"
        decoded_channel_prefix: str = "flowtts:decoded"
        worker_concurrency: int = 32

    redis: RedisSettings = RedisSettings()


settings = Settings()
