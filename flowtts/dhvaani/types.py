"""Pipeline position: CONTRACTS — the dataclasses and protocols every DhVaani
module is written against.

Nothing here imports torch at module scope beyond type checking, so this file
stays importable from the API layer, the CLI tools and the tests without
touching CUDA.

Object lifecycle
----------------
    text ──normalizer──► normalised text
         ──chunker────► list[str] spans
         ──tokenizer──► SpanRequest (token ids + voice ref)
         ──scheduler──► SpanResult  (mel frames)
         ──vocoder────► AudioChunk  (PCM bytes)
         ──api────────► client
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import torch


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------
@dataclass
class VoicePrompt:
    """A voice, fully preprocessed and ready for conditioning.

    Everything expensive about a reference clip -- decoding, resampling,
    silence removal, RMS normalisation, mel extraction, transcript tokenisation
    -- is done once when the voice is created and cached here. The synthesis hot
    path only ever reads these fields.

    Attributes:
        voice_id: stable identifier used by clients.
        mel: ``(1, T_p, 100)`` prompt mel features, already multiplied by
            ``flow.feat_scale`` and cast to the engine dtype, resident on GPU.
        mel_frames: ``T_p``, the prompt's true frame count.
        token_ids: transcript token ids (character level, from tokens.txt).
        prompt_rms: RMS of the original clip, used to restore the output's
            loudness when the clip was quieter than ``flow.target_rms``.
        frames_per_token: ``mel_frames / len(token_ids)``. This *is* DhVaani's
            duration model -- generated speech inherits the prompt's rate -- so
            it is what the chunker uses to convert a target audio duration into
            a character budget.
    """

    voice_id: str
    mel: "torch.Tensor"
    mel_frames: int
    token_ids: list[int]
    prompt_rms: float
    frames_per_token: float

    # Metadata (persisted alongside the tensors)
    name: str = ""
    description: str = ""
    language: str = ""
    transcript: str = ""
    sample_rate: int = 24000
    duration_s: float = 0.0
    created_at: float = field(default_factory=time.time)
    source_filename: str = ""
    checksum: str = ""

    def chars_for_seconds(self, seconds: float, speed: float = 1.0) -> int:
        """How many characters render to roughly ``seconds`` of audio in this voice."""
        from flowtts.dhvaani.config import FRAME_RATE_HZ

        if self.frames_per_token <= 0:
            return 80
        frames = seconds * FRAME_RATE_HZ
        return max(1, int(frames * speed / self.frames_per_token))

    def to_metadata(self) -> dict:
        return {
            "voice_id": self.voice_id,
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "transcript": self.transcript,
            "sample_rate": self.sample_rate,
            "duration_s": round(self.duration_s, 3),
            "mel_frames": self.mel_frames,
            "n_tokens": len(self.token_ids),
            "frames_per_token": round(self.frames_per_token, 6),
            "prompt_rms": round(self.prompt_rms, 6),
            "created_at": self.created_at,
            "source_filename": self.source_filename,
            "checksum": self.checksum,
        }


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
@dataclass
class SynthParams:
    """Per-request generation knobs. Defaults come from ``dhv_settings.flow``."""

    num_step: int = 8
    guidance_scale: float = 1.0
    cfg_until_t: float = 1.0
    t_shift: float = 0.5
    speed: float = 1.0
    seed: int | None = None
    output_sample_rate: int = 24000

    @classmethod
    def from_settings(cls, s=None, **overrides) -> "SynthParams":
        from flowtts.dhvaani.config import dhv_settings

        s = s or dhv_settings
        base = cls(
            num_step=s.flow.num_step,
            guidance_scale=s.flow.guidance_scale,
            cfg_until_t=s.flow.cfg_until_t,
            t_shift=s.flow.t_shift,
            speed=s.flow.speed,
            seed=s.flow.seed,
            output_sample_rate=s.audio.output_sample_rate,
        )
        for k, v in overrides.items():
            if v is not None and hasattr(base, k):
                setattr(base, k, v)
        return base

    def uses_cfg(self) -> bool:
        return self.guidance_scale != 0.0


class SpanState(str, Enum):
    QUEUED = "queued"       # admitted, not yet in an arena slot
    FLOWING = "flowing"     # occupying a slot, running Euler steps
    VOCODING = "vocoding"   # mel done, waiting on / inside the vocoder
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SpanRequest:
    """One contiguous piece of text to render as a single flow trajectory.

    A client request becomes N spans (see ``text/chunker.py``). Spans of the
    same request share a ``request_id`` and are emitted to the client strictly
    in ``span_index`` order, even though the scheduler may finish them out of
    order.
    """

    request_id: str
    span_index: int
    n_spans: int
    text: str
    token_ids: list[int]
    voice: VoicePrompt
    params: SynthParams

    is_final: bool = False
    submitted_at: float = field(default_factory=time.perf_counter)

    # --- filled in by the scheduler ---
    state: SpanState = SpanState.QUEUED
    total_frames: int = 0        # prompt + generated, before bucketing
    gen_frames: int = 0          # generated only
    bucket: int = 0
    slot: int = -1
    step: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str | None = None

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class SpanResult:
    """Rendered audio for one span."""

    request_id: str
    span_index: int
    pcm: Any                    # np.ndarray float32, mono, at output_sample_rate
    sample_rate: int
    is_final: bool
    frames: int                 # mel frames generated
    flow_ms: float = 0.0
    vocode_ms: float = 0.0
    queue_ms: float = 0.0
    steps: int = 0
    error: str | None = None

    @property
    def duration_s(self) -> float:
        return 0.0 if self.pcm is None else len(self.pcm) / self.sample_rate


@dataclass
class AudioChunk:
    """What actually goes on the wire, after stitching and encoding."""

    request_id: str
    chunk_index: int
    audio: bytes
    sample_rate: int
    encoding: str               # "pcm_int16" | "pcm_float32"
    is_final: bool
    text: str = ""
    cache_hit: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class RequestMetrics:
    """Per-request timing, emitted once on completion."""

    request_id: str
    voice_id: str
    language: str
    n_chars: int
    n_spans: int
    normalize_ms: float = 0.0
    tokenize_ms: float = 0.0
    queue_ms: float = 0.0
    ttfb_ms: float = 0.0
    flow_ms: float = 0.0
    vocode_ms: float = 0.0
    total_ms: float = 0.0
    audio_s: float = 0.0
    steps_total: int = 0
    cache_hit: bool = False
    error: str | None = None

    @property
    def rtf(self) -> float:
        return self.total_ms / 1000.0 / self.audio_s if self.audio_s > 0 else 0.0


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------
class FmStepBackend(Protocol):
    """Executes one flow-decoder velocity evaluation.

    Implementations: ``backends/torch_backend.py`` (eager / compiled / CUDA
    graphs), ``backends/trt_backend.py`` (in-process TensorRT),
    ``backends/triton_backend.py`` (NVIDIA Triton Inference Server).

    The batch handed in may be heterogeneous in ``t`` -- that is the whole point
    of the continuous-batching scheduler -- so implementations must not assume a
    scalar timestep. Shapes are always a registered bucket width.
    """

    name: str

    def supports_bucket(self, batch: int, frames: int) -> bool:
        """Whether this backend can execute ``(batch, frames)`` right now."""
        ...

    def fm_step(
        self,
        x: "torch.Tensor",              # (B, T, 100) noisy features
        text_condition: "torch.Tensor",  # (B, T, 100)
        speech_condition: "torch.Tensor",  # (B, T, 100)
        t: "torch.Tensor",              # (B,) per-sample timestep
        padding_mask: "torch.Tensor",   # (B, T) bool, True = padded
    ) -> "torch.Tensor":
        """Return predicted velocity ``(B, T, 100)``."""
        ...

    def warmup(self, buckets: Sequence[int], batch_sizes: Sequence[int]) -> None:
        """Pre-build any graphs/contexts for the given shapes."""
        ...

    def close(self) -> None:
        ...


class TextEncoderBackend(Protocol):
    """Encodes token ids to the frame-rate text condition."""

    def encode(
        self, token_ids_batch: Sequence[Sequence[int]]
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return ``(embed (B, S, 100), tokens_lens (B,))``."""
        ...


class VocoderBackend(Protocol):
    """Mel -> waveform."""

    sample_rate: int

    def decode(self, mel: "torch.Tensor", lengths: Sequence[int]) -> list["torch.Tensor"]:
        """``mel`` is ``(B, 100, T)``; returns one 1-D waveform per item, each cut
        to ``lengths[i] * hop_length`` samples."""
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class DhvaaniError(Exception):
    """Base class for all DhVaani errors."""

    status_code = 500
    code = "internal_error"


class VoiceNotFound(DhvaaniError):
    status_code = 404
    code = "voice_not_found"


class VoiceAlreadyExists(DhvaaniError):
    status_code = 409
    code = "voice_exists"


class InvalidReferenceAudio(DhvaaniError):
    status_code = 400
    code = "invalid_reference_audio"


class TextTooLong(DhvaaniError):
    status_code = 413
    code = "text_too_long"


class QueueFull(DhvaaniError):
    status_code = 503
    code = "queue_full"


class EngineNotReady(DhvaaniError):
    status_code = 503
    code = "engine_not_ready"


class RequestCancelled(DhvaaniError):
    status_code = 499
    code = "cancelled"


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]
