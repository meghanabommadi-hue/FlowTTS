from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from typing import ClassVar, Literal

_MODELS_DIR = str(Path.home() / "models")


class TtsModelSettings(BaseModel):
    checkpoint_lg: ClassVar[str] = "hindi"
    if checkpoint_lg == "telugu":
        model_dir: str = f"{_MODELS_DIR}/MeghanaKap-MiraTTSTelugu"
        warmup_sentence: str = "వర్షం పడుతున్న సాయంత్రంలో చిన్న గ్రామం మొత్తం మట్టి వాసనతో నిండిపోయి అందరినీ ఆనందంగా ముంచెత్తింది."
        ref_audio: str = f"{_MODELS_DIR}/MeghanaKap-MiraTTSTelugu/tel_male_audio.wav"
    else:
        model_dir: str = f"{_MODELS_DIR}/Shubhangi7-mira_hindi_second_round"
        warmup_sentence: str = "नमस्ते. मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से. क्या मैं customer name से बात कर रही हूं?"
        ref_audio: str = f"{_MODELS_DIR}/MeghanaKap-MiraTTSTelugu/friendly_simran.wav"

    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # sglang engine
    mem_fraction_static: float = 0.65
    attention_backend: str = "triton"        # fastest for decode-heavy TTS
    chunked_prefill_size: int = 512          # caps prefill per step; prevents long prefills starving short-sentence decode
    schedule_policy: str = "fcfs"            # lpm gave no benefit (no shared prefixes across unique TTS texts)
    cuda_graph_max_bs: int = 160
    disable_radix_cache: bool = False
    num_continuous_decode_steps: int = 1     # check EOS every step so short sentences exit immediately

    # Sampling (temperature=0 → greedy; top_p/top_k/min_p ignored)
    max_tokens: int = 600
    temperature: float = 0.0
    top_p: float = 0.7
    top_k: int = 50
    repetition_penalty: float = 1.6
    min_p: float = 0.05


class DecoderSettings(BaseModel):
    model_gpu_id: int = 0
    decoder_gpu_id: int = 0
    sample_rate: int = 16000
    enabled: bool = False    # False → skip ncodec, forward raw tokens
    to_wav: bool = True      # False → raw PCM bytes only (saves ~1-5ms)

    # TTSCodec batch queue
    max_batch: int = 128
    batch_timeout_ms: float = 50.0   # 50ms window to coalesce concurrent LLM completions into one GPU decode pass
    gpu_chunk_size: int = 90
    onnx_workers: int = 2            # parallel ONNX threads
    use_trt: bool = False


class WebSocketSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8765


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLOWTTS_", env_nested_delimiter="__")

    ws: WebSocketSettings = WebSocketSettings()
    tts_model: TtsModelSettings = TtsModelSettings()
    decoder: DecoderSettings = DecoderSettings()

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
