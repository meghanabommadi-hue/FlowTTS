"""Configuration settings for FlowTTS.

This is intentionally similar in spirit to ``litranscriber.core.config``,
but trimmed down to the essentials needed for a single-process prototype.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from typing import Literal


class TtsModelSettings(BaseModel):
    """TTS model configuration."""

    model_dir: str = "Shubhangi7/mira_hindi_second_round"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"


class DecoderSettings(BaseModel):
    """Decoder / vocoder configuration."""

    # Logical GPU ids – in a real multi-GPU deployment these would be used
    # to route work to distinct devices. For the simple prototype we keep
    # them as configuration only.
    model_gpu_id: int = 0
    decoder_gpu_id: int = 0

    # ncodec / TTSCodec settings could go here if needed.


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

