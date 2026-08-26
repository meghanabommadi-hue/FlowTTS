"""Pipeline position: API CONTRACT — request/response schemas.

Role in pipeline:
  One schema shared by every transport (REST, OpenAI-compatible, WebSocket), so
  a parameter added here is available everywhere at once and documented once in
  the OpenAPI page.

  http_app / ws  → SynthesisRequest → GenParams → engine.synthesize(…)

Coverage: every field of OmniVoice's ``OmniVoiceGenerationConfig`` and every
argument of ``OmniVoice.generate()`` is reachable per request — the denoise
schedule, CFG, the two temperatures, the layer penalty, prompt/output
post-processing, edge padding and fades, OmniVoice's internal long-form
chunking, plus speed, duration, instruct and language. Anything left ``None``
falls through to ``settings.generation``, so a caller sends only what they want
to change.

The three synthesis modes OmniVoice supports map onto the fields like this:

    voice clone   voice_id, or reference_audio + reference_text
    voice design  instruct   ("Female, Elderly, British Accent")
    auto voice    neither
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

AudioFormat = Literal["wav", "pcm", "mp3", "opus"]

# The generation-config fields a request may override, and their bounds. Kept
# here rather than inline so validation, the OpenAPI docs and GenParams.build()
# all read the same list.
GENERATION_FIELDS = (
    "num_step", "guidance_scale", "t_shift", "layer_penalty_factor",
    "position_temperature", "class_temperature", "denoise",
    "preprocess_prompt", "postprocess_output", "pad_duration", "fade_duration",
    "audio_chunk_duration", "audio_chunk_threshold",
)


class GenerationOverrides(BaseModel):
    """Per-request overrides for OmniVoice's generation config.

    Every field defaults to ``None``, meaning "use the server default".
    """

    num_step: Optional[int] = Field(
        None, ge=1, le=128,
        description="Iterative denoising steps. The dominant latency knob — cost "
                    "scales about linearly. 4 is fast, 8 balanced, 32 upstream's default.",
    )
    guidance_scale: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="Classifier-free guidance strength. NOT a speed knob on this "
                    "model: the cond+uncond batch is built either way, so 0 costs "
                    "the same as 2.0 while risking near-silent output on short "
                    "chunks. Leave at 2.0 unless you are deliberately experimenting.",
    )
    t_shift: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Timestep shift; smaller emphasises low-SNR steps.",
    )
    layer_penalty_factor: Optional[float] = Field(
        None, ge=0.0, le=50.0,
        description="Penalty encouraging earlier codebook layers to unmask first.",
    )
    position_temperature: Optional[float] = Field(
        None, ge=0.0, le=20.0,
        description="Temperature for which positions unmask next. 0 is deterministic.",
    )
    class_temperature: Optional[float] = Field(
        None, ge=0.0, le=20.0,
        description="Temperature for token values. 0 is greedy.",
    )
    denoise: Optional[bool] = Field(
        None, description="Prepend the <|denoise|> token (cleaner output on noisy references)."
    )
    preprocess_prompt: Optional[bool] = Field(
        None, description="Trim silence from, and punctuate, the reference prompt."
    )
    postprocess_output: Optional[bool] = Field(
        None, description="Run OmniVoice's silence removal on the generated audio."
    )
    pad_duration: Optional[float] = Field(
        None, ge=0.0, le=2.0,
        description="Silence padded to each side, in seconds. Forced to 0 for "
                    "streaming chunks, where it would become a gap at every seam.",
    )
    fade_duration: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Fade-in/out length in seconds."
    )
    audio_chunk_duration: Optional[float] = Field(
        None, ge=1.0, le=60.0,
        description="OmniVoice's own long-form chunk length. This server chunks "
                    "upstream of it, so raising this avoids double-chunking.",
    )
    audio_chunk_threshold: Optional[float] = Field(
        None, ge=1.0, le=120.0,
        description="Estimated duration above which OmniVoice chunks internally.",
    )

    def as_overrides(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class NormalizerOverrides(BaseModel):
    """Per-request control of the multilingual text preprocessor."""

    enabled: Optional[bool] = Field(None, description="Master switch for normalization.")
    numbers: Optional[bool] = None
    datetime: Optional[bool] = None
    urls_emails: Optional[bool] = None
    phone_numbers: Optional[bool] = None
    otp_digit_splitting: Optional[bool] = None
    abbreviations: Optional[bool] = None
    symbols: Optional[bool] = None
    contractions: Optional[bool] = None
    code_mixed: Optional[bool] = Field(
        None, description="Normalize each script run of code-mixed text in its own language."
    )
    lowercase: Optional[bool] = None
    min_digit_run: Optional[int] = Field(
        None, ge=2, le=12,
        description="Bare digit runs at least this long are read digit by digit.",
    )
    latin_language: Optional[str] = Field(
        None, description="Language for Latin runs inside an Indic sentence."
    )

    def as_overrides(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class SynthesisRequest(BaseModel):
    """A synthesis request. Only ``text`` is required."""

    text: str = Field(..., min_length=1, max_length=20000,
                      description="Text to speak. OmniVoice inline control tags "
                                  "are preserved: [laughter], [B EY1 S] (ARPAbet), "
                                  "[dissatisfaction-hnn].")

    # --- Voice selection: one of these three modes ---
    voice_id: Optional[str] = Field(
        None, description="Alias of a cloned voice (see GET /v1/voices). Voice-clone mode."
    )
    reference_audio: Optional[str] = Field(
        None, description="Base64 WAV/MP3 of a reference clip, for a one-shot clone "
                          "without registering a voice. Requires reference_text.",
    )
    reference_text: Optional[str] = Field(
        None, description="Transcript of reference_audio. Required with it — there is no ASR."
    )
    instruct: Optional[str] = Field(
        None, max_length=500,
        description="Voice-design instruction, e.g. 'Female, Elderly, British Accent'. "
                    "Used when no reference voice is given.",
    )

    # --- Delivery ---
    language: Optional[str] = Field(
        None, description="Language name or code ('hi', 'hindi', 'ta', 'en-IN'). "
                          "Omit to detect from the text's script.",
    )
    speed: Optional[float] = Field(
        None, gt=0.1, le=3.0, description="Speaking rate. >1 faster, <1 slower."
    )
    duration: Optional[float] = Field(
        None, gt=0.0, le=300.0,
        description="Force the output to exactly this many seconds. Overrides speed.",
    )
    sample_rate: Optional[int] = Field(
        None, description="Output rate. Native is 24000; 16000/8000 resample."
    )
    format: Optional[AudioFormat] = Field(None, description="Container for the response body.")
    stream: bool = Field(False, description="Stream chunks as they are generated (low TTFB).")

    # --- Tuning ---
    generation: Optional[GenerationOverrides] = None
    normalizer: Optional[NormalizerOverrides] = None
    normalize: Optional[bool] = Field(
        None, description="Shorthand for normalizer.enabled."
    )
    chunked: Optional[bool] = Field(
        None, description="Split long text and stitch. Default on; false forces "
                          "one generate() call for the whole text.",
    )

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    def generation_overrides(self) -> dict:
        """Everything that shapes GenParams, flattened."""
        overrides = self.generation.as_overrides() if self.generation else {}
        if self.speed is not None:
            overrides["speed"] = self.speed
        if self.duration is not None:
            overrides["duration"] = self.duration
        return overrides

    def normalizer_overrides(self) -> dict:
        overrides = self.normalizer.as_overrides() if self.normalizer else {}
        if self.normalize is not None:
            overrides["enabled"] = self.normalize
        return overrides


class SpeechRequest(BaseModel):
    """OpenAI ``/v1/audio/speech``-compatible request.

    Accepts the OpenAI field names so existing SDK clients work unchanged, and
    additionally accepts this server's own fields, so a caller can reach the
    OmniVoice knobs without leaving the compatible endpoint.
    """

    model: str = Field("omnivoice", description="Ignored; present for OpenAI compatibility.")
    input: str = Field(..., min_length=1, max_length=20000, description="Text to speak.")
    voice: Optional[str] = Field(None, description="Voice alias (OpenAI's name for voice_id).")
    response_format: Optional[AudioFormat] = Field(None, description="wav | pcm | mp3 | opus")
    speed: Optional[float] = Field(None, gt=0.1, le=3.0)
    stream: bool = False

    # FlowTTS extensions
    language: Optional[str] = None
    instruct: Optional[str] = None
    sample_rate: Optional[int] = None
    generation: Optional[GenerationOverrides] = None
    normalizer: Optional[NormalizerOverrides] = None

    def to_synthesis_request(self) -> SynthesisRequest:
        return SynthesisRequest(
            text=self.input,
            voice_id=self.voice,
            language=self.language,
            instruct=self.instruct,
            speed=self.speed,
            sample_rate=self.sample_rate,
            format=self.response_format,
            stream=self.stream,
            generation=self.generation,
            normalizer=self.normalizer,
        )


class VoiceCloneRequest(BaseModel):
    """JSON body for ``POST /v1/voices`` (the multipart form takes the same fields)."""

    voice_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    reference_text: str = Field(..., min_length=1,
                                description="Transcript of the clip. Required — there is no ASR.")
    audio_base64: Optional[str] = Field(None, description="Base64 of a WAV/MP3 reference clip.")
    language: Optional[str] = Field(None, description="Preferred synthesis language for this voice.")
    overwrite: bool = Field(False, description="Replace an existing voice with this id.")


class VoiceInfo(BaseModel):
    voice_id: str
    language: Optional[str] = None
    reference_frames: int
    ref_text: str
    sample_rate: Optional[int] = None
    is_default: bool = False


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]
    default_voice: Optional[str] = None


class SynthesisMetadata(BaseModel):
    """What the server did, echoed back on non-streaming responses."""

    sample_rate: int
    format: AudioFormat
    duration_seconds: float
    chunks: int
    language: Optional[str] = None
    voice_id: Optional[str] = None
    normalized_text: Optional[str] = None
    ttfb_ms: Optional[int] = None
    total_ms: int
    real_time_factor: Optional[float] = None
    cache_hit: bool = False


class SynthesisResponse(BaseModel):
    """JSON (base64) response from ``POST /v1/tts`` when audio is not streamed."""

    audio_base64: str
    metadata: SynthesisMetadata


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
