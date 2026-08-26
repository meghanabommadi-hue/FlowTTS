"""Pipeline position: TEXT PREPROCESSING — orchestrator (pure stdlib + optional backends).

Role in pipeline:
  The single entry point every request path uses before text reaches the model:

      ws / rest → normalize_for_tts(text, language) → (clean_text, language)
                → chunker.split_for_streaming(clean_text, …)
                → engine.synthesize(chunk, …)

Stage order, ported from github.com/Ajaj-Ali/text_preprocessor_for_TTS
(`indic_tts_normalizer.pipeline`):

    sanitize → native-digit→ASCII → contractions → datetime → URLs/emails →
    phone/OTP digit-splitting → numbers → abbreviations → symbols → collapse

Two things wrap that order here, both required by this model:

  1. **Control-tag protection.** OmniVoice's inline syntax — ``[laughter]``,
     ``[dissatisfaction-hnn]``, ``[B EY1 S]`` — is parked before stage one and
     restored after the last, so normalization can never eat it.
  2. **Per-script segmentation.** Code-mixed input is split into script runs and
     each run is normalized in the language that script implies (see
     script_detect.split_by_script). Normalizing "आपका balance ₹2,500 है" as one
     Hindi string speaks the Latin run wrong, and as one English string speaks
     the Devanagari wrong.

Casing is preserved by default (``lowercase=False``). Upstream lowercases
everything; for a TTS front-end that loses the acronym/proper-noun signal the
model uses, and every stage here already matches case-insensitively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from flowtts.text.abbreviations import expand_abbreviations
from flowtts.text.contact_info import (
    expand_phone_numbers,
    expand_urls_and_emails,
    split_long_digit_runs,
)
from flowtts.text.contractions import expand_contractions
from flowtts.text.datetime_expand import expand_datetime
from flowtts.text.languages import resolve_language, to_ascii_digits
from flowtts.text.numbers import expand_numbers
from flowtts.text.sanitize import (
    collapse_whitespace,
    extract_tags,
    light_sanitize,
    restore_tags,
    sanitize,
)
from flowtts.text.script_detect import detect_language, split_by_script
from flowtts.text.symbols import expand_symbols

logger = logging.getLogger(__name__)

_LATIN_LANGUAGES = {"en", "en-IN"}


@dataclass
class NormalizerConfig:
    """Toggle individual normalization stages. All default on except lowercase."""

    enabled: bool = True            # master switch — off means light cleanup only
    sanitize: bool = True
    contractions: bool = True
    datetime: bool = True
    urls_emails: bool = True
    phone_numbers: bool = True
    otp_digit_splitting: bool = True
    numbers: bool = True
    abbreviations: bool = True
    symbols: bool = True
    code_mixed: bool = True         # normalize each script run in its own language
    lowercase: bool = False         # upstream default is True; see module docstring
    # Bare digit runs at least this long are read digit-by-digit (OTPs, PINs,
    # account numbers). Comma-grouped amounts take the cardinal path regardless.
    min_digit_run: int = 4
    # Language used for Latin runs inside an Indic sentence.
    latin_language: str = "en-IN"
    extra_abbreviations: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict | None) -> "NormalizerConfig":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _normalize_segment(text: str, lang: str, cfg: NormalizerConfig) -> str:
    """Run the stage chain over one single-script run."""
    if cfg.lowercase:
        text = text.lower()

    if cfg.contractions and lang in _LATIN_LANGUAGES:
        text = expand_contractions(text)
    if cfg.datetime:
        text = expand_datetime(text, lang)
    if cfg.urls_emails:
        text = expand_urls_and_emails(text)
    if cfg.phone_numbers:
        text = expand_phone_numbers(text, lang)
    if cfg.otp_digit_splitting:
        text = split_long_digit_runs(text, lang, min_digits=cfg.min_digit_run)
    if cfg.numbers:
        text = expand_numbers(text, lang)
    if cfg.abbreviations:
        text = expand_abbreviations(text, lang)
    if cfg.symbols:
        text = expand_symbols(text, lang)
    return text


def normalize_text(
    text: str,
    language: str | None = None,
    config: Optional[NormalizerConfig] = None,
) -> str:
    """Normalize *text* for TTS in *language*; never raises on bad input.

    A failing stage degrades to the partially-normalized text rather than
    dropping the utterance: a slightly awkward number reading is a far better
    outcome for a live call than an error.
    """
    if not text:
        return ""

    cfg = config or NormalizerConfig()
    if not cfg.enabled:
        return light_sanitize(text)

    lang = resolve_language(language or detect_language(text))

    # Park OmniVoice control tags before anything can rewrite them.
    parked, tags = extract_tags(light_sanitize(text))

    try:
        if cfg.sanitize:
            parked = sanitize(parked)
        parked = to_ascii_digits(parked)

        if cfg.code_mixed:
            segments = split_by_script(parked, lang, latin_language=cfg.latin_language)
            parked = "".join(
                _normalize_segment(seg.text, seg.language, cfg) for seg in segments
            )
        else:
            parked = _normalize_segment(parked, lang, cfg)
    except Exception:  # noqa: BLE001 — normalization must never fail a request
        logger.warning("normalization failed for lang=%s; using partial result",
                       lang, exc_info=True)

    return restore_tags(collapse_whitespace(parked), tags)


def normalize_for_tts(
    text: str,
    language: str | None = None,
    config: Optional[NormalizerConfig] = None,
) -> tuple[str, str | None]:
    """Normalize *text* and resolve the language to send to the model.

    Returns ``(normalized_text, language)`` where ``language`` is the caller's
    value if they gave one, or the script-detected language if they did not.
    ``None`` language means "let OmniVoice decide" and is preserved only when
    detection finds no script at all.
    """
    if not text:
        return "", language

    resolved = language or detect_language(text, default="")
    normalized = normalize_text(text, resolved or None, config)
    return normalized, (resolved or None)


def preprocess_text(
    text: str,
    language: str | None = None,
    max_chunk_len: int = 200,
    min_chunk_len: int = 50,
    config: Optional[NormalizerConfig] = None,
) -> list[str]:
    """Normalize *text* and split it into TTS-friendly chunks.

    Kept for API parity with the upstream package. Streaming synthesis uses
    ``flowtts.synthesis.chunker.split_for_streaming`` instead, which is
    duration-aware and deliberately emits a short first chunk for low TTFB.
    """
    from flowtts.synthesis.chunker import chunk_text  # local: avoids a cycle

    if max_chunk_len < min_chunk_len:
        raise ValueError(
            f"max_chunk_len ({max_chunk_len}) must be >= min_chunk_len ({min_chunk_len})"
        )
    return chunk_text(
        normalize_text(text, language, config),
        max_chunk_len=max_chunk_len,
        min_chunk_len=min_chunk_len,
    )
