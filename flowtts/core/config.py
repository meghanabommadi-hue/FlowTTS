"""Pipeline position: CONFIGURATION (read by every module at import time).

Role in pipeline:
  Single source of truth for all tunable parameters. Every pipeline stage
  imports `settings` from here rather than reading env-vars directly.

Key sections and their pipeline consumers:
  TtsModelSettings  → synthesis/models.py (sglang engine init, sampling params)
                       synthesis/engine.py (model_dir, ref_audio paths)
  VoxCpmSettings    → synthesis/voxcpm_synthesizer.py (VoxCPM2 engine init)
  DecoderSettings   → api/websockets.py   (enabled flag, to_wav flag)
                       decoder/decoder.py  (sample_rate)
  RedisSettings     → api/websockets.py   (queue publish, pub/sub subscribe)
                       worker.py           (queue consume, pub/sub publish)
  WebSocketSettings → main.py             (uvicorn host/port)

All values can be overridden via environment variables (FLOWTTS_ prefix).
No env-var → sensible defaults used (greedy decoding, 48 kHz, Redis localhost).

Model selection:
  Set FLOWTTS_MODEL_TYPE=voxcpm  to use VoxCPM2 instead of Mira.
  Set FLOWTTS_MODEL_TYPE=mira    (default) to use the MiraITS / sglang path.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from typing import ClassVar, Literal

_MODELS_DIR      = str(Path.home() / "models")
_SAMPLE_FILES_DIR = str(Path.home() / "FlowTTS/sample_files")

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
    "anika":  f"{_SAMPLE_FILES_DIR}/anika_vb.mp3",
    "anika2": f"{_SAMPLE_FILES_DIR}/anika2_vb.mp3",
    "monika": f"{_SAMPLE_FILES_DIR}/monika_vb.mp3",
    "saavi": f"{_SAMPLE_FILES_DIR}/saavi_vb.mp3",
    "zara": f"{_SAMPLE_FILES_DIR}/zara_vb.mp3",
    "gargi": f"{_SAMPLE_FILES_DIR}/gargi_vb.mp3",
}


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
        # ref_audio: str = f"{_SAMPLE_FILES_DIR}/simran.wav"
        ref_audio: str = "/home/ubuntu/FlowTTS/sample_files/angry_tara_slow_17.wav"
        
    
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # sglang engine parameters
    mem_fraction_static: float = 0.70      # more KV cache for concurrent requests
    attention_backend: str = "triton"      # fastest for decode-heavy TTS
    chunked_prefill_size: int = 16384
    max_running_requests: int = 200        # allow all ports to run concurrently in sglang scheduler
    schedule_policy: str = "lpm"           # longest-prefix-match: reuse KV cache across requests
    cuda_graph_max_bs: int = 160           # pre-capture CUDA graphs up to this batch size
    disable_radix_cache: bool = False
    num_continuous_decode_steps: int = 10

    # Generation / sampling parameters
    # temperature=0.0 → greedy decode (top_p/top_k/min_p are ignored in greedy mode)
    max_tokens: int = 600                 # ~5 audio tokens/char × 120 char max sentence; 700 gives EOS headroom without 1024-step worst case
    temperature: float = 0.1
    top_p: float = 0.5
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
    max_batch: int = 256
    batch_timeout_ms: float = 0.7
    gpu_chunk_size: int = 150
    onnx_workers: int = 1
    use_trt: bool = True             # load pre-compiled TRT .ep engine for decoder


class StreamingSettings(BaseModel):
    """Streaming audio chunk configuration."""

    # Set to True to make streaming the default mode (no --streaming flag needed).
    enabled: bool = True

    # Tokens per chunk for the first two chunks (low latency warm-up).
    # At ~50 tokens/sec: 20 tokens ≈ 400ms of audio.
    chunk_tokens_early: int = 15
    chunk_tokens: int = 15  # alias kept for voxcpm path compatibility

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


class VoxCpmSettings(BaseModel):
    """VoxCPM2 model configuration.

    Used when Settings.model_type == "voxcpm".
    All knobs map directly to AsyncVoxCPM2ServerPool / VoxCPM2ServerImpl.
    """

    # Path to the VoxCPM2 model checkpoint directory.
    # Must contain config.json, tokenizer.json and *.safetensors weights.
    model_dir: str = "/home/ubuntu/voxcpm/model"

    # Default (simran) reference voice — used when no voice_id is specified.
    ref_audio: str = "/home/ubuntu/FlowTTS/sample_files/friendly_simran.wav"
    ref_audio_text: str = "नमस्ते मैं बजाज फिनानके की तरफ से बात कर रही हूँ।"

    # Tara voice — used when voice_id="tara" is passed in the WS request.
    tara_ref_audio: str = "/home/ubuntu/FlowTTS/sample_files/tara_vox.wav"
    tara_ref_audio_text: str = "Your EMI of three thousand seven hundred fifty rupees is due on the fifteenth of this month."

    # Warmup sentence — synthesized once at startup to prime CUDA graphs.
    warmup_sentence: str = (
        "नमस्ते, मैं आपकी कैसे मदद कर सकती हूं?"
    )

    # ------- VoxCPM2 engine knobs -------
    inference_timesteps: int = 6        # diffusion ODE steps (fewer = faster, lower quality)
    max_num_batched_tokens: int = -1 # must be >= max_model_len; prefill batch size
    max_num_seqs: int = 150             # max sequences decoded in parallel
    max_model_len: int = 4096           # max KV sequence length
    gpu_memory_utilization: float = 0.80

    # Disable CUDA graph compilation — can be faster for small models / low batch.
    enforce_eager: bool = False

    # Request coalescing window in ms (0 = off for best TTFA, >0 for throughput).
    coalesce_ms: float = 0.0

    # ------- generation params -------
    temperature: float = 0.1
    cfg_value: float = 1.5
    max_generate_length: int = 2000


class WebSocketSettings(BaseModel):
    """Gateway WebSocket server settings."""

    host: str = "0.0.0.0"
    port: int = 8080


class Settings(BaseSettings):
    """Top-level FlowTTS settings."""

    model_config = SettingsConfigDict(env_prefix="FLOWTTS_", env_nested_delimiter="__")

    # ── Model selector ──────────────────────────────────────────────────────
    # "mira"   → use MiraITS / sglang path  (synthesis/models.py)
    # "voxcpm" → use VoxCPM2 path           (synthesis/voxcpm_synthesizer.py)
    model_type: Literal["mira", "voxcpm"] = "mira"

    ws: WebSocketSettings = WebSocketSettings()
    tts_model: TtsModelSettings = TtsModelSettings()
    voxcpm: VoxCpmSettings = VoxCpmSettings()
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
