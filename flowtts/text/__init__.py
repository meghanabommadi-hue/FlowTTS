"""Multilingual text preprocessing for OmniVoice TTS.

A port of github.com/Ajaj-Ali/text_preprocessor_for_TTS (`indic-tts-normalizer`)
adapted to this server: all 22 scheduled Indian languages instead of 9, OmniVoice
control-tag protection, code-mixed (Hinglish) per-script normalization, and
total number backends that degrade instead of raising.

    from flowtts.text import normalize_for_tts
    clean, lang = normalize_for_tts("आपका balance ₹2,500 है।", "hi")
"""

from flowtts.text.abbreviations import register_abbreviation
from flowtts.text.languages import (
    SUPPORTED_LANGUAGES,
    get_profile,
    is_known_language,
    omnivoice_lang,
    register_language_alias,
    resolve_language,
)
from flowtts.text.pipeline import (
    NormalizerConfig,
    normalize_for_tts,
    normalize_text,
    preprocess_text,
)
from flowtts.text.sanitize import light_sanitize
from flowtts.text.script_detect import detect_language, is_code_mixed, split_by_script
from flowtts.text.symbols import register_symbol

__all__ = [
    "NormalizerConfig",
    "SUPPORTED_LANGUAGES",
    "detect_language",
    "get_profile",
    "is_code_mixed",
    "is_known_language",
    "light_sanitize",
    "normalize_for_tts",
    "normalize_text",
    "omnivoice_lang",
    "preprocess_text",
    "register_abbreviation",
    "register_language_alias",
    "register_symbol",
    "resolve_language",
    "split_by_script",
]
