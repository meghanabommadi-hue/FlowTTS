"""Unit tests for flowtts.text — the multilingual TTS text preprocessor.

No GPU, no model, no network. Number-backend assertions are written so they pass
whether or not num2words / indic-num2words are installed: the fallback chain is
part of the contract, so both rungs are checked for the property that matters
(no bare digits survive) rather than for one exact wording.
"""

from __future__ import annotations

import pytest

from flowtts.text import (
    NormalizerConfig,
    detect_language,
    is_code_mixed,
    normalize_for_tts,
    normalize_text,
    omnivoice_lang,
    resolve_language,
    split_by_script,
)
from flowtts.text.sanitize import extract_tags, restore_tags

NBSP = " "


def clean(text: str) -> str:
    """Normalizer output as the model receives it (NBSP joins undone)."""
    return text.replace(NBSP, " ")


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("given,expected", [
    ("hi", "hi"), ("Hindi", "hi"), ("hin", "hi"), ("HI", "hi"),
    ("en-IN", "en-IN"), ("en_in", "en-IN"), ("indian english", "en-IN"),
    ("odia", "or"), ("ory", "or"), ("oriya", "or"),
    ("bangla", "bn"), ("panjabi", "pa"), ("meitei", "mni"),
    ("nepali", "ne"), ("konkani", "kok"), ("santali", "sat"),
    (None, "en"), ("klingon", "en"),
])
def test_resolve_language(given, expected):
    assert resolve_language(given) == expected


def test_omnivoice_code_translation():
    """OmniVoice keys several Indic languages by ISO 639-3, not 639-1."""
    assert omnivoice_lang("or") == "ory"
    assert omnivoice_lang("ne") == "npi"
    assert omnivoice_lang("kok") == "knn"
    assert omnivoice_lang("doi") == "dgo"
    assert omnivoice_lang("hi") == "hi"


def test_unknown_language_passes_through():
    """OmniVoice covers 600+ languages; do not clobber a code we lack tables for."""
    assert omnivoice_lang("swh") == "swh"
    assert omnivoice_lang(None) is None


# ---------------------------------------------------------------------------
# Control-tag protection — the OmniVoice-specific requirement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "[laughter] You really got me.",
    "He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.",
    "[dissatisfaction-hnn] not happy. [surprise-oh] really?",
])
def test_control_tags_survive_normalization(text):
    out = clean(normalize_text(text, "en"))
    for tag in ("[laughter]", "[B EY1 S]", "[B AE1 S]",
                "[dissatisfaction-hnn]", "[surprise-oh]"):
        if tag in text:
            assert tag in out, f"{tag} was destroyed: {out!r}"


def test_tag_extraction_roundtrip():
    text = "a [one] b [two] c"
    parked, tags = extract_tags(text)
    assert tags == ["[one]", "[two]"]
    assert restore_tags(parked, tags) == text


def test_ordinary_parentheses_are_still_removed():
    """Only OmniVoice's square-bracket syntax is protected, not all brackets."""
    out = normalize_text("call me (soon) please", "en")
    assert "(" not in out and ")" not in out


# ---------------------------------------------------------------------------
# Numbers, currency, dates, contact info
# ---------------------------------------------------------------------------
def _no_bare_digits(text: str) -> bool:
    return not any(ch.isdigit() for ch in text)


@pytest.mark.parametrize("lang,text", [
    ("hi", "आपका बकाया ₹2,500 है।"),
    ("bn", "আপনার ব্যালেন্স ৯,৫০০ টাকা।"),
    ("ta", "உங்கள் கணக்கில் ₹1,250 உள்ளது."),
    ("te", "మీ ఖాతాలో ₹1,250 ఉంది."),
    ("kn", "ನಿಮ್ಮ ಖಾತೆಯಲ್ಲಿ ₹1,250 ಇದೆ."),
    ("ml", "നിങ്ങളുടെ അക്കൗണ്ടിൽ ₹1,250 ഉണ്ട്."),
    ("gu", "તમારા ખાતામાં ₹1,250 છે."),
    ("pa", "ਤੁਹਾਡਾ ਬਕਾਇਆ ₹5,000 ਹੈ।"),
    ("or", "ଆପଣଙ୍କ ବାକି ₹1,200 ଅଛି।"),
    ("mr", "तुमच्या खात्यात ₹1,250 आहेत."),
    ("as", "আপোনাৰ বেলেঞ্চ ₹1,500 আছে।"),
    ("ne", "तपाईंको बाँकी ₹2,000 छ।"),
    ("ur", "آپ کا بیلنس ₹2,500 ہے۔"),
    ("sat", "ᱟᱢᱟᱜ ᱵᱮᱞᱮᱱᱥ ₹1,200 ᱢᱮᱱᱟᱜᱼᱟ।"),
    ("en-IN", "Your balance is Rs. 1,250.50."),
])
def test_every_indic_language_speaks_its_numbers(lang, text):
    """No language may leave a bare numeral for the model to guess at."""
    out = clean(normalize_text(text, lang))
    assert _no_bare_digits(out), f"{lang}: digits survived — {out!r}"
    assert out.strip(), f"{lang}: normalization emptied the text"


def test_native_digits_are_converted():
    out = clean(normalize_text("मेरे पास १२३ रुपये हैं।", "hi"))
    assert _no_bare_digits(out)
    assert "१" not in out


def test_numeral_words_are_bound_with_nbsp():
    """The chunker must not be able to split one number across two chunks."""
    raw = normalize_text("₹2,500", "hi")
    assert NBSP in raw, f"expanded numeral was not bound: {raw!r}"
    assert " " not in raw.strip(), f"numeral words joined with a breakable space: {raw!r}"


def test_otp_is_read_digit_by_digit():
    out = clean(normalize_text("Your OTP is 4821.", "en"))
    for word in ("four", "eight", "two", "one"):
        assert word in out.lower(), out


def test_grouped_amount_is_read_as_a_cardinal_not_digits():
    """1,250 is money; 4821 is a code. The comma is the signal."""
    out = clean(normalize_text("Rs. 1,250", "en-IN")).lower()
    pytest.importorskip("num2words")
    assert "thousand" in out, out


def test_indian_mobile_is_grouped_five_and_five():
    out = clean(normalize_text("call 9876543210 now", "en"))
    assert "," in out, f"no pause between the two digit groups: {out!r}"
    assert _no_bare_digits(out)


def test_url_and_email_are_spelled_out():
    out = clean(normalize_text("mail ravi123@example.co.in or visit abc4.com", "en"))
    assert " at " in out and " dot " in out
    assert _no_bare_digits(out), out
    # A name's digits are read one at a time, never as a quantity.
    assert "one hundred" not in out.lower()


def test_time_keeps_the_sentence_boundary():
    """The meridiem regex must not swallow the sentence's final period."""
    out = clean(normalize_text("Due at 10:05 pm. Thanks.", "en"))
    assert out.count(".") >= 2, out


def test_dates_are_day_first():
    pytest.importorskip("num2words")
    out = clean(normalize_text("on 15/04/2026", "en")).lower()
    assert "fifteenth" in out and "april" in out, out


# ---------------------------------------------------------------------------
# Code-mixed handling
# ---------------------------------------------------------------------------
def test_code_mixed_detection():
    assert is_code_mixed("आपका balance है")
    assert not is_code_mixed("आपका बकाया है")
    assert not is_code_mixed("your balance")


def test_code_mixed_prefers_the_indic_language():
    """Hinglish is Hindi with English loanwords, even when Latin has more characters."""
    assert detect_language("आपका balance ₹2,500 है") == "hi"


def test_script_segmentation_splits_runs():
    segments = split_by_script("आपका balance है", "hi")
    languages = [s.language for s in segments]
    assert "hi" in languages and "en-IN" in languages


def test_standalone_numbers_follow_the_sentence_language():
    """"OTP 4821" inside a Tamil sentence must not be read in English."""
    out = clean(normalize_text("உங்கள் OTP 4821", "ta"))
    assert "four" not in out.lower(), out
    assert _no_bare_digits(out)


def test_digits_glued_to_latin_letters_stay_with_them():
    """abc123.com must not be torn apart by number handling."""
    out = clean(normalize_text("visit abc123.com", "hi"))
    assert "dot com" in out, out


# ---------------------------------------------------------------------------
# Config and robustness
# ---------------------------------------------------------------------------
def test_disabled_normalizer_only_strips_control_characters():
    text = "आपका बकाया ₹2,500 है।\x00"
    out = normalize_text(text, "hi", NormalizerConfig(enabled=False))
    assert "₹2,500" in out
    assert "\x00" not in out


def test_stage_toggles_are_honoured():
    cfg = NormalizerConfig(numbers=False, otp_digit_splitting=False, datetime=False)
    out = normalize_text("I owe 250 rupees", "en", cfg)
    assert "250" in out


def test_casing_is_preserved_by_default():
    assert "NEFT" in normalize_text("Use NEFT today", "en") or \
           "N E F T" in normalize_text("Use NEFT today", "en")
    assert "Ravi" in normalize_text("Ravi called", "en")


def test_empty_and_whitespace_input():
    assert normalize_text("", "hi") == ""
    assert normalize_text("   ", "hi") == ""
    assert normalize_for_tts("", "hi") == ("", "hi")


def test_normalization_never_raises_on_pathological_input():
    for text in ("₹" * 200, "..." * 100, "12:99:99 45/45/4545", "\U0001f600" * 50,
                 "a" * 5000, "[unclosed", "unopened]"):
        assert isinstance(normalize_text(text, "hi"), str)


def test_language_is_reported_back():
    _, lang = normalize_for_tts("வணக்கம்", None)
    assert lang == "ta"
