"""Pipeline position: CONFIGURATION (read by every module at import time).

Role in pipeline:
  Single source of truth for all tunable parameters. Every pipeline stage
  imports `settings` from here rather than reading env-vars directly.

Key sections and their pipeline consumers:
  TtsModelSettings  → synthesis/models.py (sglang engine init, sampling params)
                       synthesis/engine.py (model_dir, ref_audio paths)
  VoxCpmSettings    → synthesis/voxcpm_synthesizer.py (VoxCPM2 engine init)
  OmniVoiceSettings → synthesis/omnivoice.py (child-process spawn + HTTP proxy)
  DecoderSettings   → api/websockets.py   (enabled flag, to_wav flag)
                       decoder/decoder.py  (sample_rate)
  RedisSettings     → api/websockets.py   (queue publish, pub/sub subscribe)
                       worker.py           (queue consume, pub/sub publish)
  WebSocketSettings → main.py             (uvicorn host/port)

All values can be overridden via environment variables (FLOWTTS_ prefix).
No env-var → sensible defaults used (greedy decoding, 48 kHz, Redis localhost).

Model selection:
  Set FLOWTTS_MODEL_TYPE=voxcpm     to use VoxCPM2.
  Set FLOWTTS_MODEL_TYPE=omnivoice  to use OmniVoice (runs in a child process
                                    under its own venv — see setup_omni.sh).
  Set FLOWTTS_MODEL_TYPE=miotts     to use miotts (Indic-Mio + MioCodec; runs
                                    in two child processes under their own
                                    venvs — see ~/miotts/commands.md).
  Set FLOWTTS_MODEL_TYPE=mira       to use the MiraITS / sglang path.
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
        ref_audio: str = f"{_SAMPLE_FILES_DIR}/angry_tara_slow_17.wav"
        
    
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # sglang engine parameters
    # Values below match origin/neysa_h200, the config this box previously hit
    # <1s TTFT at 60 concurrent requests with.
    mem_fraction_static: float = 0.70      # more KV cache for concurrent requests
    attention_backend: str = "triton"      # fastest for decode-heavy TTS
    chunked_prefill_size: int = 16384
    max_running_requests: int = 200        # allow all ports to run concurrently in sglang scheduler
    schedule_policy: str = "lpm"           # longest-prefix-match: reuse KV cache across requests
    cuda_graph_max_bs: int = 160           # pre-capture CUDA graphs up to this batch size
    disable_radix_cache: bool = False
    num_continuous_decode_steps: int = 4   # batch N decode steps before re-scheduling → less scheduler overhead

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
    batch_timeout_ms: float = 25  # ONNX Phase-1 cost is ~fixed per call (~280ms, barely
                                   # scales with batch size) — a longer collection window
                                   # lets concurrent requests consolidate into fewer, larger
                                   # sequential batches instead of many near-singleton ones
    gpu_chunk_size: int = 150
    onnx_workers: int = 1
    use_trt: bool = True             # compiled TRT engine present at decoder_trt_b150.ep


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

    # Default (saavi) reference voice — used when no voice_id is specified.
    ref_audio: str = "/home/ubuntu/FlowTTS/sample_files/saavi_vb.mp3"
    ref_audio_text: str = "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun"

    zara_ref_audio: str = "/home/ubuntu/FlowTTS/sample_files/zara_vb.mp3"
    zara_ref_audio_text: str = "theek hai, bas pehele ek choti si confirmation chahiye. haan bas wahi detail, ab aap ka request smoothly aage badh jaayega"

    # Tara voice — used when voice_id="tara" is passed in the WS request.
    tara_ref_audio: str = "/home/ubuntu/FlowTTS/sample_files/tara_vox.wav"
    tara_ref_audio_text: str = "Your EMI of three thousand seven hundred fifty rupees is due on the fifteenth of this month."

    # Warmup sentence — synthesized once at startup to prime CUDA graphs.
    warmup_sentence: str = (
        "नमस्ते, मैं आपकी कैसे मदद कर सकती हूं?"
    )

    # ------- VoxCPM2 engine knobs -------
    inference_timesteps: int = 6        # diffusion ODE steps (fewer = faster, lower quality)
    max_num_batched_tokens: int = 16384  # must be >= max_model_len
    max_num_seqs: int = 150             # max sequences decoded in parallel
    max_model_len: int = 4096           # max KV sequence length
    gpu_memory_utilization: float = 0.80

    # Disable CUDA graph compilation — can be faster for small models / low batch.
    enforce_eager: bool = False

    # Request coalescing window in ms (0 = off for best TTFA, >0 for throughput).
    coalesce_ms: float = 0.0

    # ------- generation params -------
    temperature: float = 0.2
    cfg_value: float = 2.0
    max_generate_length: int = 2000


class OmniVoiceSettings(BaseModel):
    """OmniVoice model configuration.

    Used when Settings.model_type == "omnivoice". OmniVoice's own deps
    (transformers>=5.3.0) conflict with the transformers==4.57.3 pin sglang
    needs in this venv, so the model runs in a child process under its own
    venv (created by setup_omni.sh) and is proxied over loopback HTTP —
    see synthesis/omnivoice.py for details.

    The server code itself is vendored in-tree at FlowTTS/omnivoice/ (no
    dependency on a separate omnivoice_scaled checkout); only its Python
    deps live in a dedicated venv alongside it.
    """

    # Location of the vendored omnivoice server code and its own venv interpreter.
    repo_dir: str = str(Path.home() / "FlowTTS" / "omnivoice")
    venv_python: str = str(Path.home() / "FlowTTS" / "omnivoice" / ".venv" / "bin" / "python")

    # Loopback port the child process's FastAPI server binds to.
    port: int = 8090
    startup_timeout_s: float = 300.0
    request_timeout_s: float = 60.0

    # Passed straight through as OMNIVOICE_* env vars to the child process.
    model_id: str = "k2-fsa/OmniVoice"
    device: str = "cuda:0"
    num_step: int = 16
    max_batch_size: int = 24

    language: str | None = None

    # Playback speed multiplier (1.0 = normal, <1.0 = slower, >1.0 = faster).
    # None → let the model use its own default.
    speed: float | None = None

    # Target duration in seconds for the generated audio. None → model
    # decides length naturally from the text/generation config.
    duration: float | None = None

    # Free-text style/emotion instruction (e.g. "speak angrily", "whisper").
    # None → no instruction, model uses its default delivery.
    instruct: str | None = None

    # Default reference voice.
    ref_audio: str | None = f"{_SAMPLE_FILES_DIR}/saavi_vb.mp3"
    ref_audio_text: str | None = (
        "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun"
    )

    # Named voices — voice_id -> (ref_audio_path, ref_text).
    voices: dict[str, tuple[str, str]] = {}

    # Warmup sentence — synthesized once at startup to prime the child process.
    warmup_sentence: str = "नमस्ते, मैं आपकी कैसे मदद कर सकती हूं?"


class MiottsSettings(BaseModel):
    """miotts model configuration (SPRINGLab/Indic-Mio LLM + MioCodec decoder).

    Used when Settings.model_type == "miotts". miotts's own checkout splits
    into two venvs (.venv_vllm for vllm==0.8.5+transformers==4.51.3 on
    Python 3.10, .venv for miocodec on Python >=3.12) because miocodec's
    package metadata declares Requires-Python >=3.12, which is simply
    incompatible with .venv_vllm's 3.10 interpreter in that checkout.

    FlowTTS does NOT reuse either of those -- it has its own single venv,
    .venv_mio (Python 3.12, see setup_mio.sh), with vllm==0.8.5,
    transformers==4.51.3, AND miocodec all installed together. This was
    verified to actually work (not just resolve without pip conflicts):
    torch/transformers/vllm/miocodec all import successfully in the same
    process, with matching pinned versions and no flashinfer pulled in.
    So unlike OmniVoice (two genuinely incompatible dependency sets), miotts's
    vLLM server and codec server here are still two separate child processes
    (for the same reason miotts's own run.sh keeps them separate -- one
    holds the LLM, one holds the codec, and running two GPU-resident
    processes at all is a deliberate memory/latency choice, not a Python-
    version workaround) -- but both launch from the SAME .venv_mio
    interpreter. See synthesis/miotts.py for details, and README.md's
    "Model dependency matrix" section for why each FlowTTS backend needs
    the venv it needs.
    """

    # Location of the miotts checkout (for its postprocess.py + codec_server.py
    # source, loaded/launched from there) and FlowTTS's own unified venv.
    repo_dir: str = str(Path.home() / "miotts")
    venv_python: str = str(Path("/home/jovyan/FlowTTS") / ".venv_mio" / "bin" / "python")

    # Loopback ports the two child processes bind to.
    vllm_port: int = 8100
    codec_port: int = 8101
    startup_timeout_s: float = 300.0
    request_timeout_s: float = 60.0

    # vLLM server launch params (mirrors miotts/run.sh defaults).
    model_name: str = "SPRINGLab/Indic-Mio"
    max_model_len: int = 2560
    gpu_memory_utilization: float = 0.6

    # Generation params (mirrors miotts/config.py's MioConfig defaults).
    max_new_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.85
    repetition_penalty: float = 1.1

    # Default voice (priority: reference_audio > voice_preset).
    voice_preset: str | None = "en_female"
    reference_audio: str | None = None

    # Named voices -- voice_id -> (voice_preset, reference_audio); either may
    # be None. Resolution follows miotts's own VoiceResolver.resolve() order.
    voices: dict[str, tuple[str | None, str | None]] = {}

    # Post-decode glitch smoothing (miotts/postprocess.py:smooth_glitches).
    # Only applies to this backend -- no other model's audio is touched.
    smooth_glitches: bool = True

    # Warmup sentence -- synthesized once at startup to prime both servers.
    warmup_sentence: str = "Hello, how are you today?"


class WebSocketSettings(BaseModel):
    """Gateway WebSocket server settings."""

    host: str = "0.0.0.0"
    port: int = 8080


class Settings(BaseSettings):
    """Top-level FlowTTS settings."""

    model_config = SettingsConfigDict(env_prefix="FLOWTTS_", env_nested_delimiter="__")

    # ── Model selector ──────────────────────────────────────────────────────
    # "mira"      → use MiraITS / sglang path  (synthesis/models.py)
    # "voxcpm"    → use VoxCPM2 path           (synthesis/voxcpm_synthesizer.py)
    # "omnivoice" → use OmniVoice path         (synthesis/omnivoice.py)
    # model_type: Literal["mira", "voxcpm"] = "mira"
    model_type: Literal["mira", "voxcpm", "omnivoice", "miotts"] = "voxcpm"

    ws: WebSocketSettings = WebSocketSettings()
    tts_model: TtsModelSettings = TtsModelSettings()
    voxcpm: VoxCpmSettings = VoxCpmSettings()
    omnivoice: OmniVoiceSettings = OmniVoiceSettings()
    miotts: MiottsSettings = MiottsSettings()
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
