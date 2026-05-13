"""Pipeline position: CONFIGURATION (read by every module at import time).

Role in pipeline:
  Single source of truth for all tunable parameters. Every pipeline stage
  imports `settings` from here rather than reading env-vars directly.

Key sections and their pipeline consumers:
  TtsModelSettings  → synthesis/models.py (sglang engine init, sampling params)
                       synthesis/engine.py (model_dir, ref_audio paths)
  DecoderSettings   → api/websockets.py   (enabled flag, to_wav flag)
                       decoder/decoder.py  (sample_rate)
  RedisSettings     → api/websockets.py   (queue publish, pub/sub subscribe)
                       worker.py           (queue consume, pub/sub publish)
  WebSocketSettings → main.py             (uvicorn host/port)

All values can be overridden via environment variables (FLOWTTS_ prefix).
No env-var → sensible defaults used (greedy decoding, 48 kHz, Redis localhost).
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field
from typing import Dict, Literal

_MODELS_DIR      = str(Path.home() / "models")
_SAMPLE_FILES_DIR = str(Path.home() / "FlowTTS/sample_files")
_LORA_DIR = "/root/mira_lora_setup/lora_switching"

# Per-voice reference audio paths. Keys match voice_id values sent by clients.
VOICE_REF_AUDIO: dict[str, str] = {
    "simran":       f"{_SAMPLE_FILES_DIR}/simran.wav",
    "tara":         f"{_SAMPLE_FILES_DIR}/tara.wav",
    "vikram":       f"{_SAMPLE_FILES_DIR}/vikram.wav",
    "daya":         f"{_SAMPLE_FILES_DIR}/daya.wav",
    "british_rose": f"{_SAMPLE_FILES_DIR}/british_rose.wav",
    "rani": f"{_SAMPLE_FILES_DIR}/rani.wav",
    "sana":  f"{_SAMPLE_FILES_DIR}/sana.wav",
    "anita":  f"{_SAMPLE_FILES_DIR}/anita.wav",
    "vanita": f"{_SAMPLE_FILES_DIR}/vanita.wav",
    "sunita": f"{_SAMPLE_FILES_DIR}/sunita.wav",
}


class TtsModelSettings(BaseModel):
    """TTS model configuration."""

    # Base model — LoRA adapters are loaded on top for each language.
    # Accepts a local directory path or a HuggingFace repo ID.
    model_dir: str = "Shubhangi7/mira-english-1-epoch"
    warmup_sentence: str = "नमस्ते. मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से. क्या मैं customer name से बात कर रही हूं?"
    ref_audio: str = "/root/mira_lora_setup/vaani_24k.wav"

    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # sglang engine parameters — flashinfer + disabled cuda graph required for LoRA
    mem_fraction_static: float = 0.70
    attention_backend: str = "flashinfer"
    chunked_prefill_size: int = -1
    disable_radix_cache: bool = True
    disable_cuda_graph: bool = True         # required for LoRA

    # Generation / sampling parameters
    max_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.3
    min_p: float = 0.05

    # LoRA language switching — maps language tag → local adapter path.
    # All adapters are registered at engine init time; switching is zero-cost at inference.
    language_lora_map: Dict[str, str] = Field(default_factory=lambda: {
        "hi": "/root/mira_lora_setup/lora_switching/hi-checkpoint-50000-lora-only",
        "ta": "/root/mira_lora_setup/lora_switching/ta-checkpoint-50000-lora-only",
    })
    default_language: str = "hi"


class DecoderSettings(BaseModel):
    """Decoder / vocoder configuration."""

    # Logical GPU ids – in a real multi-GPU deployment these would be used
    # to route work to distinct devices. For the simple prototype we keep
    # them as configuration only.
    model_gpu_id: int = 0
    decoder_gpu_id: int = 0

    sample_rate: int = 16000

    # Set to False to skip ncodec decoding and forward raw LLM tokens instead.
    # Useful for latency profiling and when decoder is not yet ready.
    enabled: bool = False

    # Set to False to decode to raw PCM bytes only (skip WAV encoding).
    # Saves ~1-5ms per request. Use when client handles raw float32 PCM.
    to_wav: bool = True

    # TTSCodec batch queue settings — scaled up for H200's larger memory bandwidth
    max_batch: int = 256
    batch_timeout_ms: float = 0.7
    gpu_chunk_size: int = 150
    onnx_workers: int = 1
    use_trt: bool = True             # load pre-compiled TRT .ep engine for decoder


class StreamingSettings(BaseModel):
    """Streaming audio chunk configuration."""

    # Set to True to make streaming the default mode (no --streaming flag needed).
    enabled: bool = True

    # --- Chunk size ---
    # Tokens per chunk for the first two chunks (low latency warm-up).
    # At ~50 tokens/sec: 20 tokens ≈ 400ms of audio.
    chunk_tokens_early: int = 15

    # Tokens per chunk from chunk 3 onward (larger = fewer boundaries = smoother).
    chunk_tokens_late: int = 50

    # --- Codec overlap ---
    # Tail tokens from the previous chunk prepended to the next decode for context.
    # Their decoded audio is discarded. Higher = smoother codec boundary quality.
    # 12 tokens = 240ms of context.
    overlap_tokens: int = 12

    # --- Server-side crossfade ---
    # Samples held back from each chunk's tail and blended into the next chunk's head.
    # 1280 = 80ms at 16kHz. Set to 0 to disable crossfade entirely.
    crossfade_samples: int = 1280

    # Unused — fade-out disabled to prevent gaps.
    fade_out_samples: int = 0


class WebSocketSettings(BaseModel):
    """Gateway WebSocket server settings."""

    host: str = "0.0.0.0"
    port: int = 8080


class Settings(BaseSettings):
    """Top-level FlowTTS settings."""

    model_config = SettingsConfigDict(env_prefix="FLOWTTS_", env_nested_delimiter="__")

    ws: WebSocketSettings = WebSocketSettings()
    tts_model: TtsModelSettings = TtsModelSettings()
    decoder: DecoderSettings = DecoderSettings()
    streaming: StreamingSettings = StreamingSettings()

    # Directory of pre-generated WAV files named by SHA256 of raw transcript.
    # Set via env var: FLOWTTS_WAV_CACHE_DIR=/path/to/wav/folder
    wav_cache_dir: str | None = str(Path.home() / "FlowTTS/cached_data_simran") 

    # Redis queue / pubsub configuration
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
