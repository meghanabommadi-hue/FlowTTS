"""Pipeline position: CONFIGURATION (read by every module at import time).

Role in pipeline:
  Single source of truth for all tunable parameters. Every pipeline stage
  imports `settings` from here rather than reading env-vars directly.

Model:
  This server runs **k2-fsa/OmniVoice** — a non-autoregressive discrete-diffusion
  TTS language model (Qwen3-0.6B backbone + Higgs-Audio-v2 neural codec, 24 kHz),
  with the Qwen3 backbone optionally served from a TensorRT / TensorRT-LLM engine
  (see flowtts/trt/, after github.com/tlitech/omnivoice-trtllm).

Key sections and their pipeline consumers:
  OmniVoiceSettings  → synthesis/omnivoice_engine.py, trt/patcher.py
  GenerationDefaults → synthesis/omnivoice_engine.py  (every OmniVoice knob)
  TextSettings       → text/pipeline.py               (multilingual normalizer)
  VoiceSettings      → voices/registry.py             (npz voice-clone registry)
  OutputSettings     → api/*, decoder/decoder.py      (stream rate + encoding)
  StreamingSettings  → synthesis/chunker.py, processing/stitch.py
  ServerSettings     → service.py                     (WS / HTTP / control ports)

All values can be overridden via environment variables (FLOWTTS_ prefix,
nested via `__`, e.g. FLOWTTS_GENERATION__NUM_STEP=8), and the three latency
profiles in ``apply_profile`` set a coherent group of them at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
_HOME = Path.home()
# Repo root (…/FlowTTS) — used to resolve relative paths regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VOICES_DIR = str(_HOME / "FlowTTS/voices")          # *.npz voice-clone artifacts
_WAV_CACHE_DIR = str(_HOME / "FlowTTS/cached_data")  # sha256(text).wav cache

# OmniVoice's native output sample rate. Read `engine.sampling_rate` at runtime
# for the authoritative value; this constant is the documented default.
OMNIVOICE_NATIVE_SR = 24000


class GenerationDefaults(BaseModel):
    """Every field of OmniVoice's ``OmniVoiceGenerationConfig``, plus the
    generate() arguments that are not per-request identity.

    These are *defaults*. Each one is also overridable per request, so a caller
    can trade latency for quality on a single utterance without touching the
    server (see api/models.py → SynthesisParams).
    """

    # --- Denoising schedule ---
    # num_step is the dominant latency lever: cost scales ~linearly with it.
    # Upstream's default is 32; 16 is the documented fast setting.
    num_step: int = 16
    t_shift: float = 0.1                # smaller → emphasise low-SNR steps
    denoise: bool = True                # prepend the <|denoise|> token

    # --- Classifier-free guidance ---
    # Counter to the usual intuition, this is NOT a throughput lever on
    # OmniVoice. _generate_iterative() builds the 2B cond+uncond batch and runs
    # the backbone over all of it on every step regardless of the value, so
    # guidance_scale=0 costs exactly the same as 2.0 — measured at 186 ms vs
    # 190 ms per utterance on an L40S. What it does change is robustness: the
    # model is trained with CFG, and at 0 it collapses to near-silence on short
    # chunks (peak 0.0002 against a normal 0.5). Leave it on.
    guidance_scale: float = 2.0

    # --- Token selection ---
    layer_penalty_factor: float = 5.0   # encourages earlier codebooks to unmask first
    position_temperature: float = 5.0   # 0.0 → deterministic position choice
    class_temperature: float = 0.0      # 0.0 → greedy token values

    # --- Prompt / output processing ---
    preprocess_prompt: bool = True      # trim + punctuate the reference prompt
    postprocess_output: bool = True     # silence removal, fades, edge padding
    pad_duration: float = 0.1           # silence padded per side (seconds)
    fade_duration: float = 0.1          # fade-in/out curve (seconds)

    # --- OmniVoice's own long-form chunking (inside generate()) ---
    # We do our own duration-aware chunking upstream of this, so these stay high
    # to avoid double-chunking an already-short chunk.
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0

    # --- generate() arguments ---
    speed: float | None = None          # >1 faster, <1 slower; None = model default
    duration: float | None = None       # fixed output seconds; overrides speed
    normalize_text: bool = False        # OmniVoice's own normalizer; ours runs first

    def as_generation_kwargs(self) -> dict:
        """The subset that constructs an ``OmniVoiceGenerationConfig``."""
        exclude = {"speed", "duration", "normalize_text"}
        return {k: v for k, v in self.model_dump().items() if k not in exclude}


class OmniVoiceSettings(BaseModel):
    """Model load, batching, and backbone-acceleration config."""

    # --- Model load ---
    model_repo: str = "k2-fsa/OmniVoice"     # HF repo id (used if no local weights)
    # Local weights dir (relative to repo root, or absolute). Used INSTEAD of
    # downloading when it exists. Override with FLOWTTS_OMNIVOICE__MODEL_PATH.
    model_path: str = "model_dir/base"
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "float16"
    # ASR (Whisper) stays off: ref_text is mandatory for cloning, so there is
    # nothing to auto-transcribe, and loading it costs startup time and VRAM.
    load_asr: bool = False
    trust_remote_code: bool = True

    # --- Backbone acceleration (flowtts/trt) ---
    # "auto"     — best available: tensorrt → trtllm → compiled torch → pytorch
    # "tensorrt" — engine from `python -m flowtts.trt.build_trt`
    # "trtllm"   — engine from `python -m flowtts.trt.build_trtllm`
    # "torch"    — the PyTorch mirror (honours compile_model)
    # "pytorch"  — no patching at all; stock transformers forward
    backbone_backend: Literal["auto", "tensorrt", "trtllm", "torch", "pytorch"] = "auto"
    trt_engine_dir: str = "engines/omnivoice-backbone"
    trtllm_engine_dir: str = "engines/omnivoice-trtllm"
    # Compare the accelerated backbone against the real llm before installing it.
    # Leave this on: it is the guard that stops a bad engine reaching live audio.
    backbone_validate: bool = True
    backbone_min_cosine: float = 0.99

    # --- torch.compile (used by the "torch" backend) ---
    compile_model: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"

    # --- Dynamic in-flight batching (request-level, length-bucketed) ---
    # Concurrent synthesize() calls — including every streamed chunk from every
    # connection — are coalesced into batched model.generate([...]) calls.
    #
    # These numbers come from flowtts.test.bench_batching, and they are smaller
    # than an LLM server's for a structural reason. OmniVoice's
    # _generate_iterative runs a PER-ITEM Python loop inside every denoise step
    # (top-k, gumbel, masked_fill, copy_ per batch element), and materializes
    # float32 logits of [2B, 8, S, 1025]. Both scale with B and neither is
    # batched work, so the win flattens almost immediately:
    #
    #     batch  1 -> 61.2 ms/item      batch  8 -> 53.3 ms/item
    #     batch  2 -> 48.8 ms/item      batch 16 -> 47.8 ms/item
    #     batch  4 -> 49.1 ms/item      batch 24 -> 52.3 ms/item
    #
    # Everything batching has to offer is captured by batch 2-4; past that a
    # larger batch adds latency (a batch of 24 blocks the GPU for 1.26 s) and
    # buys nothing. So the cap is deliberately modest.
    max_batch: int = 8
    batch_timeout_ms: float = 8.0        # collection window before dispatch
    # Total estimated audio frames in one batch. A batch is atomic — nothing in
    # it returns until all of it does — so this bounds how long one request can
    # be made to wait by the company it keeps.
    max_batch_frames: int = 2400
    # How much length mismatch to tolerate within a batch. generate() pads every
    # item to the longest, and measurement says the padding is not worth paying:
    # batching a short chunk with a long one ran at 0.66x of doing them
    # separately. Keep this tight — it is a quality-of-service knob, not a
    # throughput one.
    length_bucket_ratio: float = 1.5
    # Requests beyond this wait in the queue rather than piling onto the GPU.
    max_active_requests: int = 256

    # --- Warmup ---
    warmup: bool = True
    warmup_batch: int = 4
    warmup_sentence: str = (
        "नमस्ते. मैं बजाज finance से बोल रही हूं, एक recorded line के माध्यम से. "
        "क्या मैं customer name से बात कर रही हूं?"
    )


class TextSettings(BaseModel):
    """Multilingual text preprocessing (flowtts/text).

    A port of github.com/Ajaj-Ali/text_preprocessor_for_TTS extended to all 22
    scheduled Indian languages, with OmniVoice control-tag protection and
    per-script normalization of code-mixed input.
    """

    enabled: bool = True
    sanitize: bool = True
    contractions: bool = True
    datetime: bool = True
    urls_emails: bool = True
    phone_numbers: bool = True
    otp_digit_splitting: bool = True
    numbers: bool = True
    abbreviations: bool = True
    symbols: bool = True
    code_mixed: bool = True
    lowercase: bool = False
    min_digit_run: int = 4
    latin_language: str = "en-IN"
    # Language assumed when a request declares none and the text has no script
    # to detect from (pure digits/punctuation).
    default_language: str | None = None


class VoiceSettings(BaseModel):
    """Voice-clone registry: precomputed npz prompts addressed by alias."""

    voices_dir: str = _VOICES_DIR
    # Alias used when a request omits voice_id (must exist in voices_dir).
    default_voice: str = "priya"
    # Language passed to OmniVoice when a request omits it (None = auto-detect).
    default_language: str | None = None
    # Reference clips longer than this are trimmed before encoding: prompt
    # tokens sit in front of every generated chunk, so a 30 s reference makes
    # every request in the batch pay for 30 s of prefix.
    max_reference_seconds: float = 20.0


class OutputSettings(BaseModel):
    """Audio output format for the WebSocket and HTTP streams.

    OmniVoice is natively 24 kHz. If sample_rate < native, chunks are resampled
    before encoding. The rate is always echoed in the chunk header / response
    so compliant clients adapt automatically.
    """

    sample_rate: int = OMNIVOICE_NATIVE_SR   # 24000 native; 16000/8000 to resample
    encoding: Literal["pcm_int16"] = "pcm_int16"
    # Default container for the REST API when the caller does not ask.
    default_format: Literal["wav", "pcm", "mp3", "opus"] = "wav"


class StreamingSettings(BaseModel):
    """Chunked streaming — the only way to stream a non-autoregressive model.

    Chunks are measured in characters and aligned to punctuation; see
    flowtts/synthesis/chunker.py for why the alignment matters more than the size.
    """

    enabled: bool = True

    # A chunk runs to the last sentence end within target + tolerance. Only when
    # 250 characters pass with no sentence ending does it fall back to a comma,
    # and only failing that to a word gap.
    target_chars: int = 200
    tolerance_chars: int = 50
    split_on_clause: bool = True
    # Shrinks the FIRST chunk only, trading one extra seam for a lower TTFB.
    # None keeps every chunk the same size, which is what sounds best.
    first_chunk_chars: int | None = None
    max_chunks: int = 128

    # --- Stitching (flowtts/processing/stitch.py) ---
    # The pause inserted after each kind of boundary, in milliseconds. A full
    # stop gets a real pause; a comma a shorter one; a word-gap cut none at all,
    # because the phrase genuinely continues and gets a crossfade instead.
    sentence_gap_ms: float = 260.0
    clause_gap_ms: float = 130.0
    crossfade_ms: float = 20.0
    edge_fade_ms: float = 8.0
    click_fade_ms: float = 3.0
    final_fade_ms: float = 12.0
    trim_silence: bool = True
    trim_keep_ms: float = 10.0
    # Every chunk is normalized to one level for the whole utterance. OmniVoice's
    # output level wanders ~3.4 dB between calls, which reads as the voice
    # changing character mid-sentence, so the clamp has to be wider than that.
    level_match: bool = True
    level_match_max_db: float = 6.0
    # Streaming chunks skip OmniVoice's own edge padding and fades: those insert
    # ~200 ms of silence per chunk, which is exactly what the stitcher would then
    # have to cut back out.
    disable_model_edge_processing: bool = True


class ServerSettings(BaseModel):
    """Listener configuration for the unified service entry point."""

    host: str = "0.0.0.0"
    ws_base_port: int = 8080     # WebSocket gateway (FlowTTS protocol)
    ws_ports: int = 1            # number of consecutive WS ports to bind
    http_port: int = 8000        # REST + OpenAI-compatible + WS at /ws
    ctrl_port: int = 8764        # health / ready / metrics / ports
    ws_idle_timeout_s: float = 300.0
    request_timeout_s: float = 120.0
    cors_origins: list[str] = ["*"]


class WebSocketSettings(BaseModel):
    """Legacy alias kept so existing deployments' env vars keep working."""

    host: str = "0.0.0.0"
    port: int = 8080


class Settings(BaseSettings):
    """Top-level FlowTTS/OmniVoice settings."""

    model_config = SettingsConfigDict(env_prefix="FLOWTTS_", env_nested_delimiter="__")

    ws: WebSocketSettings = WebSocketSettings()
    server: ServerSettings = ServerSettings()
    omnivoice: OmniVoiceSettings = OmniVoiceSettings()
    generation: GenerationDefaults = GenerationDefaults()
    text: TextSettings = TextSettings()
    voices: VoiceSettings = VoiceSettings()
    output: OutputSettings = OutputSettings()
    streaming: StreamingSettings = StreamingSettings()

    # Directory of pre-generated WAV files named by SHA256 of raw transcript.
    # Cache hits bypass the model entirely (a large win for repeated IVR prompts).
    wav_cache_dir: str | None = _WAV_CACHE_DIR
    wav_cache_enabled: bool = True

    # Redis queue / pubsub (secondary, Redis-backed multi-process path).
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

    api_keys: list[str] = Field(default_factory=list)   # empty = no auth


settings = Settings()


# ---------------------------------------------------------------------------
# Latency profiles
# ---------------------------------------------------------------------------
# Coherent groups of settings, because the knobs are not independent: halving
# num_step and leaving CFG on costs about as much as leaving num_step alone and
# turning CFG off, and picking one at random gives neither the latency nor the
# quality you wanted.
# Every profile keeps guidance_scale at 2.0. Turning it down buys nothing on
# this model (see GenerationDefaults.guidance_scale) and costs robustness, so
# num_step is the only latency dial that actually moves.
#
# Measured on an L40S shared with a training job, single request, 2.4 s of
# Hindi: num_step 4 -> 86 ms, 8 -> 190 ms, 16 -> 342 ms, 32 -> 695 ms.
PROFILES: dict[str, dict] = {
    # Lowest latency. 4 steps is the floor before artifacts start showing on
    # Indic consonant clusters; at 2 the output is audible but visibly less
    # stable run to run.
    "fast": {
        "generation": {"num_step": 4, "guidance_scale": 2.0},
        "omnivoice": {"max_batch": 8, "batch_timeout_ms": 6.0},
    },
    # The default: ~2x the compute of `fast` for clearly steadier prosody.
    "balanced": {
        "generation": {"num_step": 8, "guidance_scale": 2.0},
        "omnivoice": {"max_batch": 8, "batch_timeout_ms": 8.0},
    },
    # Upstream's own defaults. Use for offline generation and for building the
    # WAV cache, where latency does not matter and quality is the whole point.
    "quality": {
        "generation": {"num_step": 32, "guidance_scale": 2.0},
        "omnivoice": {"max_batch": 4, "batch_timeout_ms": 12.0},
    },
}


def apply_profile(name: str) -> dict:
    """Apply a latency profile in place; return the values it set.

    Explicit environment variables are NOT overridden — a profile is a starting
    point, and an operator who set FLOWTTS_GENERATION__NUM_STEP meant it.
    """
    import os

    profile = PROFILES.get(name)
    if profile is None:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")

    applied: dict = {}
    for section, values in profile.items():
        target = getattr(settings, section)
        for key, value in values.items():
            env_key = f"FLOWTTS_{section.upper()}__{key.upper()}"
            if env_key in os.environ:
                continue
            setattr(target, key, value)
            applied.setdefault(section, {})[key] = value
    return applied


def resolve_model_source() -> str:
    """Return the OmniVoice weights source to load.

    Prefers a local weights directory (settings.omnivoice.model_path, resolved
    relative to the repo root when not absolute); otherwise the HuggingFace repo id.
    """
    path = Path(settings.omnivoice.model_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return str(path) if path.is_dir() else settings.omnivoice.model_repo


def is_local_model() -> bool:
    """True if a local weights directory will be used (no HF download needed)."""
    path = Path(settings.omnivoice.model_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.is_dir()


def resolve_path(value: str | None) -> Path | None:
    """Resolve a possibly-relative configured path against the repo root."""
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else _REPO_ROOT / path
