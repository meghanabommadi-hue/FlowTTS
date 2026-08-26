"""Pipeline position: CONFIGURATION — single source of truth for the DhVaani stack.

Every DhVaani module imports ``dhv_settings`` from here instead of reading
environment variables directly.  All fields are overridable via environment
variables using the ``DHVAANI_`` prefix and ``__`` nesting delimiter, e.g.::

    DHVAANI_FLOW__NUM_STEP=8
    DHVAANI_ENGINE__MAX_BATCH_SIZE=64
    DHVAANI_BACKEND__KIND=trt

Consumers
---------
ModelSettings    -> model/loader.py         (weights, vocoder, tokenizer)
FlowSettings     -> engine/scheduler.py     (ODE steps, CFG, t_shift)
EngineSettings   -> engine/*                (batching, buckets, admission)
BackendSettings  -> backends/*              (torch / trt / triton selection)
ChunkSettings    -> text/chunker.py         (streaming chunk schedule)
TextSettings     -> text/normalizer.py      (normalisation + cache)
VoiceSettings    -> voices/*                (clone store, prompt limits)
AudioSettings    -> engine/vocode.py        (output sample rate, crossfade)
MemorySettings   -> engine/memory.py        (VRAM arenas + GC watchdog)
ServerSettings   -> api/*, server.py        (ports, limits, timeouts)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_HOME = Path.home()

# Frames-per-second of the Vocos mel front-end used by DhVaani:
#   sampling_rate / hop_length = 24000 / 256 = 93.75
FRAME_RATE_HZ: float = 24000.0 / 256.0
HOP_LENGTH: int = 256
N_MELS: int = 100
MODEL_SAMPLE_RATE: int = 24000


class ModelSettings(BaseModel):
    """Where the DhVaani weights live and how they are loaded."""

    # HuggingFace repo id (gated -- needs HF_TOKEN) or a local snapshot dir.
    repo_id: str = "ARTPARK-IISc/DhVaani-0.5"
    local_dir: str | None = Field(
        default=str(_HOME / "models" / "DhVaani-0.5"),
        description="Local snapshot dir. Downloaded on first run if absent.",
    )
    hf_token_env: str = "HF_TOKEN"

    # Vocos mel vocoder (24 kHz). Downloaded from the Hub on first run.
    vocoder_repo: str = "charactr/vocos-mel-24khz"
    vocoder_local_dir: str | None = None

    device: str = "cuda:0"
    # bfloat16 is numerically safer for the Zipformer's large activations;
    # float16 is faster on Ada (L40S) and is what the TRT engines are built in.
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"

    # Run the text encoder in fp32 regardless of `dtype`. It is tiny (192-dim,
    # 4 layers) and its output feeds every ODE step, so precision here is cheap
    # insurance against drift.
    text_encoder_fp32: bool = True


class FlowSettings(BaseModel):
    """Flow-matching (Euler ODE) sampling parameters."""

    # Number of Euler steps. 16 = model-card quality, 8 = the throughput default,
    # 4 = aggressive. Cost is exactly linear in this number.
    num_step: int = 8

    # Classifier-free guidance. NOTE: any non-zero value DOUBLES the batch that
    # goes through the flow decoder (uncond + cond), i.e. 2x the FLOPs.
    guidance_scale: float = 1.0

    # Skip CFG once t exceeds this value; the uncond branch matters most at low
    # t. `1.0` == CFG on for every step (upstream behaviour).
    # e.g. 0.5 keeps CFG for the first half of the trajectory only.
    cfg_until_t: float = 1.0

    # Shift the timestep grid toward low-SNR (small t). Upstream generate_sentence
    # uses 0.5; the raw-evaluation path uses 1.0.
    t_shift: float = 0.5

    # Feature/RMS scaling constants baked into the checkpoint. Do not change
    # unless the checkpoint changes.
    feat_scale: float = 0.1
    target_rms: float = 0.1

    # Speaking rate multiplier (>1 = faster). Applied to predicted durations.
    speed: float = 1.0

    # Deterministic sampling: fixed seed makes x0 reproducible per request.
    # `None` -> per-request random noise (recommended for production variety).
    seed: int | None = None


class BucketSettings(BaseModel):
    """Length bucketing for the padded batch arenas."""

    # Frame counts are rounded UP to a multiple of this before bucketing.
    # 64 keeps padding waste under ~8% for typical chunk lengths while giving
    # TensorRT a small set of shapes to specialise on.
    granularity: int = 64

    min_frames: int = 128
    max_frames: int = 1536

    @property
    def buckets(self) -> Tuple[int, ...]:
        g = self.granularity
        lo = ((self.min_frames + g - 1) // g) * g
        hi = ((self.max_frames + g - 1) // g) * g
        return tuple(range(lo, hi + g, g))


class EngineSettings(BaseModel):
    """Continuous-batching scheduler configuration."""

    # Hard cap on flow streams resident in the scheduler at once, summed across
    # all buckets. This is the primary VRAM knob.
    max_active_streams: int = 192

    # Max rows in a single fm_decoder forward pass (pre-CFG). The effective GPU
    # batch is 2x this when CFG is on.
    max_batch_size: int = 64

    # Scheduler tick budget. The loop runs flat-out while work exists; this is
    # the idle sleep when the pool is empty.
    idle_sleep_s: float = 0.0005

    # How long a newly-arrived request may wait to be merged into a running
    # batch. 0 = admit at the very next step boundary (lowest TTFB).
    admission_delay_ms: float = 0.0

    # Requests queued beyond this are rejected with a 503 / ws error rather than
    # silently building an unbounded backlog.
    max_queue_depth: int = 512

    # Per-request wall-clock ceiling for the whole synthesis.
    request_timeout_s: float = 60.0

    # Prefer packing a bucket to `max_batch_size` before stepping it, up to this
    # many microseconds of extra wait. Trades a little TTFB for GPU efficiency.
    batch_fill_wait_us: float = 250.0

    # Number of chunks of a single request that may be in flight simultaneously.
    # >1 lets chunk N+1 start its ODE while chunk N is still vocoding.
    chunk_lookahead: int = 2


class BackendSettings(BaseModel):
    """Which compute backend executes the flow-decoder step."""

    kind: Literal["torch", "trt", "triton"] = "torch"

    # --- torch backend ---
    use_cuda_graphs: bool = True
    use_torch_compile: bool = False        # compile() the fm_decoder step
    compile_mode: str = "max-autotune-no-cudagraphs"

    # Fused OpenAI-Triton kernels for the elementwise glue in the ODE loop
    # (Euler update, CFG combine, condition concat). Cuts ~6 kernel launches
    # per step; matters most at high batch counts.
    use_triton_kernels: bool = True

    # --- TensorRT backend (in-process) ---
    trt_engine_dir: str = str(_HOME / "models" / "DhVaani-0.5" / "trt")
    trt_fp16: bool = True
    trt_fp8: bool = False                  # Ada/L40S supports FP8; needs calibration
    trt_workspace_gb: int = 8
    trt_build_on_missing: bool = False     # build engines at startup if absent

    # --- NVIDIA Triton Inference Server backend ---
    triton_url: str = "localhost:8001"
    triton_protocol: Literal["grpc", "http"] = "grpc"
    triton_use_cuda_shm: bool = True
    triton_model_fm_step: str = "dhvaani_fm_step"
    triton_model_text_encoder: str = "dhvaani_text_encoder"
    triton_model_vocoder: str = "dhvaani_vocoder"
    triton_client_timeout_s: float = 30.0


class ChunkSettings(BaseModel):
    """Smart streaming chunk schedule.

    DhVaani is non-autoregressive: it renders a whole span at once. Streaming is
    therefore achieved by splitting the text and rendering spans back to back.
    Every span pays the prompt's frames as fixed overhead, so the first span is
    kept small (fast TTFB) and later spans grow (amortise the prompt).
    """

    enabled: bool = True

    # Target audio seconds for the first / second / steady-state spans.
    first_chunk_seconds: float = 1.2
    second_chunk_seconds: float = 2.5
    steady_chunk_seconds: float = 4.5

    # Never emit a span shorter than this unless it is the only/last one --
    # very short spans sound clipped and waste prompt compute.
    min_chunk_seconds: float = 0.6

    # Hard ceiling; ZipVoice quality degrades past roughly 25 s of
    # (prompt + generated) audio in one span.
    max_span_seconds: float = 20.0

    # Prefer breaking at these, in descending priority: sentence end, clause,
    # then whitespace. Handled by text/chunker.py.
    prefer_sentence_boundaries: bool = True

    # If the whole utterance fits in this many seconds, do not chunk at all --
    # one span is always higher quality than two.
    single_span_max_seconds: float = 2.0


class TextSettings(BaseModel):
    """Text normalisation and tokenisation."""

    normalize: bool = True

    # LRU cache of normalised text keyed by (text, lang, config-hash). IVR-style
    # traffic repeats heavily, so this keeps normalisation off the hot path.
    cache_size: int = 20000

    # Run normalisation in a thread pool so a pathological input can never block
    # the event loop. 0 = inline (normalisation is sub-ms for typical inputs).
    normalize_threads: int = 2

    # indic-tts-normalizer lowercases by default. DhVaani's vocab contains both
    # ASCII cases, so preserving case is safe and helps proper nouns.
    lowercase: bool = False

    # Stages -- mirrors indic_tts_normalizer.NormalizerConfig
    contractions: bool = True
    datetime: bool = True
    urls_emails: bool = True
    otp_digit_splitting: bool = True
    numbers: bool = True
    abbreviations: bool = True
    symbols: bool = True

    # Fall back to this language when the request omits one and script
    # detection is inconclusive.
    default_language: str = "hi"

    # Drop characters that are not in tokens.txt before tokenising, and log a
    # counter. DhVaani silently drops OOV; we make it observable.
    report_oov: bool = True


class VoiceSettings(BaseModel):
    """Voice-clone registry."""

    store_dir: str = str(_HOME / "FlowTTS" / "dhvaani_voices")

    # Reference clips are trimmed to this length. Prompt frames are pure
    # overhead on EVERY span, so shorter prompts are dramatically cheaper:
    # a 3 s prompt costs 281 frames of flow compute per span.
    max_prompt_seconds: float = 3.0
    min_prompt_seconds: float = 1.0

    # Strip long internal silences and edge silence from the reference clip
    # once, at creation time (pydub; ~100-300 ms of CPU -- never on the hot path).
    remove_silence: bool = True
    trailing_silence_ms: int = 200

    # Max reference-audio upload size.
    max_upload_bytes: int = 25 * 1024 * 1024

    # Voices kept resident on the GPU. Others live on disk as .npz and are
    # paged in on demand (LRU). Each cached voice costs
    # max_prompt_seconds * 93.75 * 100 * 2 bytes ~= 56 KB in fp16 -- cheap.
    gpu_cache_size: int = 512

    default_voice: str | None = None


class AudioSettings(BaseModel):
    """Output audio format and span stitching."""

    # 24000 is native. 16000/8000 resample on the GPU for telephony.
    output_sample_rate: int = 24000

    encoding: Literal["pcm_int16", "pcm_float32"] = "pcm_int16"

    # Overlap-add crossfade between consecutive spans, in seconds. The tail of
    # each span is held back and blended into the next span's head, so nothing
    # is repeated and no click is audible. 0 disables (fade-in only).
    crossfade_seconds: float = 0.06

    # Trim edge silence from each generated span (cheap, numpy RMS scan).
    trim_edge_silence: bool = True
    silence_threshold_db: float = -45.0

    # Fade the very last span's tail so the utterance ends cleanly.
    final_fade_seconds: float = 0.02

    # Sub-chunk PCM writes so the client starts playing before the whole span
    # is vocoded. 0 = emit whole spans.
    emit_slice_ms: int = 0


class MemorySettings(BaseModel):
    """VRAM stability: pre-allocated arenas plus a GC watchdog.

    The scheduler never allocates per-request GPU tensors on the hot path; it
    writes into arenas sized once at startup. The watchdog exists to catch
    fragmentation from the vocoder and from out-of-profile shapes.
    """

    # Pre-allocate bucket arenas at startup rather than on first use. Costs
    # startup time and VRAM but removes all allocator jitter from steady state.
    preallocate_arenas: bool = True

    # Fraction of total VRAM the arenas may occupy. The remainder is left for
    # weights, the vocoder, TRT contexts and CUDA overhead.
    arena_vram_fraction: float = 0.45

    # Run torch.cuda.empty_cache() + gc.collect() when reserved-but-unallocated
    # memory exceeds this fraction of total VRAM.
    gc_reserved_slack_fraction: float = 0.15

    # Minimum seconds between watchdog collections -- empty_cache() synchronises
    # the device, so it must never run in a tight loop.
    gc_min_interval_s: float = 30.0

    # Also force a collection every N completed requests regardless of slack.
    gc_every_n_requests: int = 20000

    # Emit VRAM gauges to Prometheus every N seconds.
    vram_poll_interval_s: float = 5.0

    # Abort admission when allocated VRAM exceeds this fraction; sheds load
    # instead of OOMing.
    admission_vram_ceiling: float = 0.92

    # expandable_segments cuts fragmentation from the variable-length vocoder
    # outputs substantially. Applied at import time if unset in the env.
    set_pytorch_cuda_alloc_conf: bool = True
    pytorch_cuda_alloc_conf: str = "expandable_segments:True"


class ServerSettings(BaseModel):
    """Gateway: WebSocket + REST + control plane."""

    host: str = "0.0.0.0"

    # WebSocket ports, matching the existing FlowTTS multi-port model.
    ws_base_port: int = 8080
    ws_num_ports: int = 1

    # REST (OpenAI-compatible + voice CRUD + metrics + health).
    http_port: int = 8000

    # Control API (port add/list, readiness) -- kept separate from the REST API
    # so it can be firewalled off.
    ctrl_port: int | None = 8764

    ws_idle_timeout_s: float = 300.0
    ws_max_message_bytes: int = 16 * 1024 * 1024
    ws_ping_interval_s: float = 30.0

    # Reject texts longer than this outright.
    max_text_chars: int = 5000

    cors_allow_origins: list[str] = ["*"]

    # Warm the GPU with synthetic requests before binding ports.
    warmup_enabled: bool = True
    warmup_batch: int = 32
    warmup_rounds: int = 2

    api_keys: list[str] = []   # empty = auth disabled


class DhvaaniSettings(BaseSettings):
    """Top-level DhVaani settings object."""

    model_config = SettingsConfigDict(
        env_prefix="DHVAANI_",
        env_nested_delimiter="__",
        extra="ignore",
        protected_namespaces=(),
    )

    model: ModelSettings = ModelSettings()
    flow: FlowSettings = FlowSettings()
    buckets: BucketSettings = BucketSettings()
    engine: EngineSettings = EngineSettings()
    backend: BackendSettings = BackendSettings()
    chunk: ChunkSettings = ChunkSettings()
    text: TextSettings = TextSettings()
    voice: VoiceSettings = VoiceSettings()
    audio: AudioSettings = AudioSettings()
    memory: MemorySettings = MemorySettings()
    server: ServerSettings = ServerSettings()

    # ---- derived helpers -------------------------------------------------
    def seconds_to_frames(self, seconds: float) -> int:
        return max(1, int(round(seconds * FRAME_RATE_HZ)))

    def frames_to_seconds(self, frames: int) -> float:
        return frames / FRAME_RATE_HZ

    def bucket_for(self, frames: int) -> int:
        """Smallest bucket that fits `frames`; clamps to the largest bucket."""
        g = self.buckets.granularity
        want = ((int(frames) + g - 1) // g) * g
        bs = self.buckets.buckets
        if want <= bs[0]:
            return bs[0]
        if want >= bs[-1]:
            return bs[-1]
        return want


PROFILES: dict[str, dict] = {
    # Lowest latency / highest throughput. CFG off halves flow FLOPs outright.
    "fast": {
        "flow": {"num_step": 4, "guidance_scale": 0.0},
        "chunk": {"first_chunk_seconds": 1.0, "steady_chunk_seconds": 5.0},
    },
    # The default: 8 steps with CFG on the low-t half of the trajectory.
    "balanced": {
        "flow": {"num_step": 8, "guidance_scale": 1.0, "cfg_until_t": 0.5},
        "chunk": {"first_chunk_seconds": 1.2, "steady_chunk_seconds": 4.5},
    },
    # Model-card settings. ~4x the flow FLOPs of "fast".
    "quality": {
        "flow": {"num_step": 16, "guidance_scale": 1.0, "cfg_until_t": 1.0},
        "chunk": {"first_chunk_seconds": 1.5, "steady_chunk_seconds": 6.0},
    },
}


def apply_profile(s: "DhvaaniSettings", name: str) -> "DhvaaniSettings":
    """Overlay a named preset. Explicit env vars still win -- they are re-applied
    on top of the preset by re-reading the environment for the touched fields."""
    import os

    prof = PROFILES.get(name)
    if prof is None:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    for section, fields in prof.items():
        sub = getattr(s, section)
        for k, v in fields.items():
            env_key = f"DHVAANI_{section.upper()}__{k.upper()}"
            if env_key in os.environ:
                continue  # explicit env override beats the preset
            setattr(sub, k, v)
    return s


dhv_settings = DhvaaniSettings()

# Applied before the first CUDA allocation. expandable_segments keeps the
# variable-length vocoder outputs from fragmenting the caching allocator, which
# is the main source of "VRAM slowly grows" in a long-running TTS server.
if dhv_settings.memory.set_pytorch_cuda_alloc_conf:
    import os as _os

    _os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", dhv_settings.memory.pytorch_cuda_alloc_conf
    )
