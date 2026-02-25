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

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from typing import Literal


class TtsModelSettings(BaseModel):
    """TTS model configuration."""

    model_dir: str = "/root/models/MeghanaKap-MiraTTSTelugu"
    ref_audio: str = "/root/models/tel_male_audio.wav"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # sglang engine parameters
    mem_fraction_static: float = 0.8       # more KV cache for concurrent requests
    attention_backend: str = "flashinfer"   # fastest for decode-heavy TTS
    chunked_prefill_size: int = 128         # small prefill chunks → decode starts sooner
    max_running_requests: int = 100         # allow all ports to run concurrently in sglang scheduler
    schedule_policy: str = "lpm"            # longest-prefix-match: reuse KV cache across requests
    cuda_graph_max_bs: int = 160            # pre-capture CUDA graphs up to this batch size
    disable_radix_cahce: bool = True
    # Warmup sentence — run once after model load to prime the GPU
    warmup_sentence: str = "నమస్తే! ఎలా ఉన్నారు?"

    # Generation / sampling parameters
    # temperature=0.0 → greedy decode (top_p/top_k/min_p are ignored in greedy mode)
    max_tokens: int = 512                   # TTS tokens rarely exceed 300; cuts tail latency
    temperature: float = 0.0               # greedy — fastest, deterministic
    top_p: float = 0.7
    top_k: int = 50
    repetition_penalty: float = 1.2
    min_p: float = 0.05


class DecoderSettings(BaseModel):
    """Decoder / vocoder configuration."""

    # Logical GPU ids – in a real multi-GPU deployment these would be used
    # to route work to distinct devices. For the simple prototype we keep
    # them as configuration only.
    model_gpu_id: int = 0
    decoder_gpu_id: int = 0

    sample_rate: int = 48000

    # Set to False to skip ncodec decoding and forward raw LLM tokens instead.
    # Useful for latency profiling and when decoder is not yet ready.
    enabled: bool = False

    # Set to False to decode to raw PCM bytes only (skip WAV encoding).
    # Saves ~1-5ms per request. Use when client handles raw float32 PCM.
    to_wav: bool = True

    # TTSCodec batch queue settings
    max_batch: int = 128             # batch queue max size
    batch_timeout_ms: float = 10.0   # ms to wait collecting a batch
    gpu_chunk_size: int = 90         # max items per GPU forward pass
    onnx_workers: int = 1            # parallel ONNX worker threads
    use_trt: bool = False            # compile decoder with TensorRT FP16


class WebSocketSettings(BaseModel):
    """Gateway WebSocket server settings."""

    host: str = "0.0.0.0"
    port: int = 8765


class Settings(BaseSettings):
    """Top-level FlowTTS settings."""

    model_config = SettingsConfigDict(env_prefix="FLOWTTS_", env_nested_delimiter="__")

    ws: WebSocketSettings = WebSocketSettings()
    tts_model: TtsModelSettings = TtsModelSettings()
    decoder: DecoderSettings = DecoderSettings()

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
