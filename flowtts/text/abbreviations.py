"""Pipeline position: TEXT PREPROCESSING — abbreviation expansion, per language.

Ported from github.com/Ajaj-Ali/text_preprocessor_for_TTS
(`indic_tts_normalizer.abbreviations`), with the starter dictionary widened to
the abbreviations that actually dominate Indian voice-bot traffic (EMI, KYC,
OTP, UPI, NEFT, IFSC, …) and to Devanagari honorifics.

Acronyms that must be *spelled out* rather than read as a word get an explicit
spaced expansion ("KYC" → "K Y C"): OmniVoice will happily pronounce "kyc" as a
syllable otherwise, which is wrong in every Indian language.

Use :func:`register_abbreviation` to extend at runtime rather than forking.
"""

from __future__ import annotations

import re

from flowtts.text.languages import resolve_language

# Written-form abbreviations that end in a period.
_EN_ABBREVIATIONS = {
    "dr.": "doctor",
    "mr.": "mister",
    "mrs.": "misses",
    "ms.": "miss",
    "prof.": "professor",
    "etc.": "etcetera",
    "e.g.": "for example",
    "i.e.": "that is",
    "approx.": "approximately",
    "vs.": "versus",
    "no.": "number",
    "govt.": "government",
    "dept.": "department",
    "fig.": "figure",
    "rs.": "rupees",
    "inr": "rupees",
    "a/c": "account",
    "ltd.": "limited",
    "pvt.": "private",
    "st.": "saint",
}

# Financial / telecom acronyms read letter by letter. These are the ones a
# collections or support bot says dozens of times per call.
_EN_ACRONYMS = {
    "emi": "E M I",
    "otp": "O T P",
    "kyc": "K Y C",
    "upi": "U P I",
    "neft": "N E F T",
    "rtgs": "R T G S",
    "imps": "I M P S",
    "ifsc": "I F S C",
    "pan": "P A N",
    "cvv": "C V V",
    "atm": "A T M",
    "nach": "N A C H",
    "ecs": "E C S",
    "sms": "S M S",
    "id": "I D",
    "url": "U R L",
    "faq": "F A Q",
    "gst": "G S T",
    "tds": "T D S",
    "nbfc": "N B F C",
    "sip": "S I P",
    "nav": "N A V",
    "roi": "R O I",
}

_HI_ABBREVIATIONS = {
    "डॉ.": "डॉक्टर",
    "श्री.": "श्री",
    "कृ.": "कृपया",
    "रु.": "रुपये",
    "प्रो.": "प्रोफेसर",
}

ABBREVIATIONS: dict[str, dict[str, str]] = {
    "en": {**_EN_ABBREVIATIONS, **_EN_ACRONYMS},
    "en-IN": {**_EN_ABBREVIATIONS, **_EN_ACRONYMS},
    "hi": dict(_HI_ABBREVIATIONS),
    "mr": dict(_HI_ABBREVIATIONS),
    "ne": dict(_HI_ABBREVIATIONS),
}

_pattern_cache: dict[str, re.Pattern | None] = {}


def _rebuild_pattern(lang: str) -> None:
    table = ABBREVIATIONS.get(lang)
    if not table:
        _pattern_cache[lang] = None
        return
    escaped = sorted((re.escape(k) for k in table), key=len, reverse=True)
    # (?<!\w) / (?!\w) rather than \b: several keys end in "." where \b does not
    # match the way you would expect.
    _pattern_cache[lang] = re.compile(
        r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)", re.IGNORECASE
    )


def register_abbreviation(lang: str, abbreviation: str, expansion: str) -> None:
    """Add or override an abbreviation for a language at runtime."""
    canonical = resolve_language(lang)
    ABBREVIATIONS.setdefault(canonical, {})[abbreviation.lower()] = expansion
    _rebuild_pattern(canonical)


def expand_abbreviations(text: str, lang: str) -> str:
    canonical = resolve_language(lang)
    if canonical not in _pattern_cache:
        _rebuild_pattern(canonical)
    pattern = _pattern_cache.get(canonical)
    if pattern is None:
        return text
    table = ABBREVIATIONS[canonical]
    # .lower() is a no-op for caseless scripts and normalizes English keys
    # regardless of how the input was written.
    return pattern.sub(lambda m: table.get(m.group(0).lower(), m.group(0)), text)
