"""Configuration settings for FlowTTS.

This is intentionally similar in spirit to ``litranscriber.core.config``,
but trimmed down to the essentials needed for a single-process prototype.
Uses sglang Engine for inference (see TTSIntegration/ws_server.py).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from typing import Literal


class TtsModelSettings(BaseModel):
    """TTS model configuration."""

    model_dir: str = "/root/CleanTTSData/inference/models/MeghanaKap-MiraTTSTelugu"
    ref_audio: str = "/root/CleanTTSData/data/cropped_20260206output.wav"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # sglang engine parameters (mirrors TTSIntegration/ws_server.py)
    mem_fraction_static: float = 0.8
    attention_backend: str = "flashinfer"
    chunked_prefill_size: int = -1

    # Generation / sampling parameters
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.95
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
        worker_concurrency: int = 32

    redis: RedisSettings = RedisSettings()


settings = Settings()
