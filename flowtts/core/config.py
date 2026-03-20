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
from pydantic import BaseModel
from typing import ClassVar, Literal

_MODELS_DIR = str(Path.home() / "models")


class TtsModelSettings(BaseModel):
    """TTS model configuration."""
    checkpoint_lg: ClassVar[str] = "hindi"
    if checkpoint_lg == "telugu":
        model_dir: str = f"{_MODELS_DIR}/MeghanaKap-MiraTTSTelugu"
        warmup_sentence: str = "వర్షం పడుతున్న సాయంత్రంలో చిన్న గ్రామం మొత్తం మట్టి వాసనతో నిండిపోయి అందరినీ ఆనందంగా ముంచెత్తింది."
        ref_audio: str = f"{_MODELS_DIR}/MeghanaKap-MiraTTSTelugu/tel_male_audio.wav"
    else:
        model_dir: str = f"{_MODELS_DIR}/Shubhangi7-mira_hindi_second_round"
        warmup_sentence: str = "नमस्ते. मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से. क्या मैं customer name से बात कर रही हूं?"
        # ref_audio: str = f"{_MODELS_DIR}/MeghanaKap-MiraTTSTelugu/vaani_fast.wav"
        # ref_audio: str = f"{_MODELS_DIR}/MeghanaKap-MiraTTSTelugu/simran_eleven_labs.wav"
        ref_audio: str = f"{_MODELS_DIR}/MeghanaKap-MiraTTSTelugu/friendly_simran.wav"
        
    
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # sglang engine parameters
    # 0.83 × 139.72 GiB ≈ 116 GiB to sglang → ~23 GiB free for decoder ONNX sessions,
    # AudioTokenizer FP16 weights, and batch tensors (~10-12 GiB needed).
    # At 0.85 only 3.9 GiB remained → OOM at req 122. All other values restored from
    # the proven A100 baseline that achieved 0.9s TTFT at 200 requests.
    mem_fraction_static: float = 0.83
    attention_backend: str = "triton"
    chunked_prefill_size: int = 8192
    max_running_requests: int = 200
    schedule_policy: str = "lpm"
    cuda_graph_max_bs: int = 160
    disable_radix_cache: bool = False
    num_continuous_decode_steps: int = 10

    # Generation / sampling parameters
    # temperature=0.0 → greedy decode (top_p/top_k/min_p are ignored in greedy mode)
    max_tokens: int = 600                 # ~5 audio tokens/char × 120 char max sentence; 700 gives EOS headroom without 1024-step worst case
    temperature: float = 0.1
    top_p: float = 0.7
    top_k: int = 50
    repetition_penalty: float = 1.6
    min_p: float = 0.05


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
    max_batch: int = 256             # H200 handles larger batches without OOM
    batch_timeout_ms: float = 0.5   # ms to wait collecting a batch (longer = better packing)
    gpu_chunk_size: int = 160        # H200 has 2x HBM bandwidth vs A100 — double the chunk
    onnx_workers: int = 2            # two ONNX threads to keep GPU fed between batches
    use_trt: bool = False             # load pre-compiled TRT .ep engine for decoder


class StreamingSettings(BaseModel):
    """Streaming audio chunk configuration."""

    # Set to True to make streaming the default mode (no --streaming flag needed).
    enabled: bool = True

    # Number of speech tokens accumulated before decoding and sending a chunk.
    # Lower = more chunks, lower latency to first audio; higher = fewer round-trips.
    # At 50 tokens/sec: 20 tokens ≈ 400ms of audio per chunk.
    chunk_tokens: int = 15

    # Linear crossfade overlap between consecutive chunks (samples at 16 kHz).
    # 320 = 20ms. Set to 0 to disable crossfade.
    crossfade_samples: int = 400

    # Linear fade-out applied to the tail of each non-final chunk to suppress
    # codec boundary noise (samples at 16 kHz). 480 = 30ms.
    fade_out_samples: int = 160


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
    streaming: StreamingSettings = StreamingSettings()

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
