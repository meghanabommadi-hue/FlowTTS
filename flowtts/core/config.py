"""Pipeline position: CONFIGURATION (read by every module at import time).

Role in pipeline:
  Single source of truth for all tunable parameters. Every pipeline stage
  imports `settings` from here rather than reading env-vars directly.

Model:
  This server runs **k2-fsa/OmniVoice** — a non-autoregressive discrete-diffusion
  TTS language model (Qwen3-0.6B backbone + Higgs-Audio-v2 neural codec, 24 kHz).
  It replaces the previous sglang/ncodec MiraTTS stack entirely.

Key sections and their pipeline consumers:
  OmniVoiceSettings  → synthesis/omnivoice_engine.py (model load, generation, batching, accel)
  VoiceSettings      → voices/registry.py            (npz voice-clone registry + aliases)
  OutputSettings     → server.py, decoder/decoder.py (stream sample rate, encoding)
  StreamingSettings  → server.py, synthesis/models.py (text-chunk streaming for low TTFB)
  WebSocketSettings  → main.py / server.py           (host/port)
  RedisSettings      → worker.py, api/websockets.py  (secondary Redis-backed path)

All values can be overridden via environment variables (FLOWTTS_ prefix,
nested via `__`, e.g. FLOWTTS_OMNIVOICE__NUM_STEP=8).
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
# Repo root (…/FlowTTS) — used to resolve a relative local model path regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_FILES_DIR = str(_HOME / "FlowTTS/sample_files")
_VOICES_DIR = str(_HOME / "FlowTTS/voices")          # *.npz voice-clone artifacts
_WAV_CACHE_DIR = str(_HOME / "FlowTTS/cached_data")  # sha256(text).wav cache

# OmniVoice's native output sample rate. Read `engine.sampling_rate` at runtime
# for the authoritative value; this constant is the documented default.
OMNIVOICE_NATIVE_SR = 24000


class OmniVoiceSettings(BaseModel):
    """k2-fsa/OmniVoice model, generation, batching, and acceleration config."""

    # --- Model load ---
    model_repo: str = "k2-fsa/OmniVoice"     # HF repo id (used only if no local weights)
    # Local weights dir (relative to repo root, or absolute). If it exists it is used
    # INSTEAD of downloading from HuggingFace — set to your local snapshot, e.g.
    # "model_dir/base". Override with FLOWTTS_OMNIVOICE__MODEL_PATH.
    model_path: str = "model_dir/base"
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"  # bf16 preferred on Hopper/H200
    # Whisper ASR auto-transcribes ref audio when ref_text is missing. Not needed
    # at serve time because voice prompts are precomputed to npz — keep it off to
    # save GPU memory. The offline voice-clone builder turns it on explicitly.
    load_asr: bool = False
    trust_remote_code: bool = True

    # --- Generation (OmniVoiceGenerationConfig knobs) ---
    # num_step is the dominant latency lever: latency scales ~linearly with it.
    # Default 32 (quality); 16 documented as the fast setting; try 8-12 on H200.
    num_step: int = 16
    guidance_scale: float = 2.0        # classifier-free guidance; note CFG ≈ 2× compute
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0  # set 0.0 for deterministic position selection
    class_temperature: float = 0.0     # 0.0 = greedy token values (already deterministic)
    denoise: bool = True
    # Long-form chunking inside generate(); we do our own streaming chunking, so
    # keep these high to avoid double-chunking within a single (already-short) chunk.
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0

    # --- Dynamic in-flight batching (request-level, length-bucketed) ---
    # Concurrent synthesize() calls (and streamed chunks) are coalesced into one
    # model.generate([...]) call, mirroring the proven async batch-queue pattern.
    max_batch: int = 32
    batch_timeout_ms: float = 8.0      # collection window before dispatch
    # Estimated-token buckets used to group similar-length items in one batch,
    # minimizing padding waste and keeping the set of compiled/captured shapes
    # small (important for torch.compile + CUDA-graph reuse). Values are audio
    # frames (≈ frame_rate * seconds); tune on the H200.
    length_buckets: list[int] = [64, 128, 256, 400, 600]

    # --- Acceleration (all opt-in + graceful fallback if unavailable) ---
    # torch.compile the diffusion backbone (and codec) submodules. "reduce-overhead"
    # enables CUDA graphs per captured shape; "max-autotune" tunes kernels harder.
    compile_model: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    # Run the Higgs codec decode on a dedicated CUDA stream so it can overlap the
    # next diffusion batch (pipelining). Falls back to the default stream if unset.
    overlap_codec_decode: bool = True

    # --- Warmup ---
    # Synthesize a few sentences per length bucket at startup to trigger
    # compilation / CUDA-graph capture and prime caches before real traffic.
    warmup: bool = True
    warmup_sentence: str = (
        "नमस्ते. मैं बजाज finance से बोल रही हूं, एक recorded line के माध्यम से. "
        "क्या मैं customer name से बात कर रही हूं?"
    )


class VoiceSettings(BaseModel):
    """Voice-clone registry: precomputed npz prompts addressed by alias."""

    voices_dir: str = _VOICES_DIR
    # Alias used when a request omits voice_id (must exist in voices_dir).
    default_voice: str = "priya"
    # Language passed to OmniVoice when a request omits it (None = auto-detect).
    default_language: str | None = None


class OutputSettings(BaseModel):
    """Audio output format for the WebSocket stream.

    OmniVoice is natively 24 kHz. If sample_rate < native, chunks are resampled
    before encoding. The `sample_rate` value is always echoed in the audio_chunk
    JSON header so compliant clients adapt automatically.
    """

    sample_rate: int = OMNIVOICE_NATIVE_SR   # 24000 native; set 16000/8000 to resample
    encoding: Literal["pcm_int16"] = "pcm_int16"


class StreamingSettings(BaseModel):
    """Text-chunk streaming — the only way to stream a non-autoregressive model.

    Long text is split into chunks; each chunk is generated (batched with other
    requests) and its PCM streamed as soon as it is ready. A short first chunk
    minimizes time-to-first-byte.
    """

    enabled: bool = True

    # First chunk kept short so the first audio frame streams fast (low TTFB).
    first_chunk_max_chars: int = 60
    # Subsequent chunks larger for smoother prosody / fewer boundaries.
    chunk_max_chars: int = 160
    # Never emit a chunk shorter than this unless it is the last one.
    min_chunk_chars: int = 12

    # Boundary smoothing between decoded chunks (samples at output rate).
    crossfade_samples: int = 480       # 20 ms @ 24 kHz
    fade_out_samples: int = 240        # fade the final chunk tail


class WebSocketSettings(BaseModel):
    """Gateway WebSocket server settings."""

    host: str = "0.0.0.0"
    port: int = 8080


class Settings(BaseSettings):
    """Top-level FlowTTS/OmniVoice settings."""

    model_config = SettingsConfigDict(env_prefix="FLOWTTS_", env_nested_delimiter="__")

    ws: WebSocketSettings = WebSocketSettings()
    omnivoice: OmniVoiceSettings = OmniVoiceSettings()
    voices: VoiceSettings = VoiceSettings()
    output: OutputSettings = OutputSettings()
    streaming: StreamingSettings = StreamingSettings()

    # Directory of pre-generated WAV files named by SHA256 of raw transcript.
    # Cache hits bypass the model entirely (huge win for repeated call-centre prompts).
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


def resolve_model_source() -> str:
    """Return the OmniVoice weights source to load.

    Prefers a local weights directory (settings.omnivoice.model_path, resolved
    relative to the repo root if not absolute) when it exists; otherwise falls
    back to the HuggingFace repo id (settings.omnivoice.model_repo).
    """
    p = Path(settings.omnivoice.model_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if p.is_dir():
        return str(p)
    return settings.omnivoice.model_repo


def is_local_model() -> bool:
    """True if a local weights directory will be used (no HF download needed)."""
    p = Path(settings.omnivoice.model_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p.is_dir()
