"""Tests for text normalisation across DhVaani's 27 languages.

Two tiers exist: 14 languages `indic_tts_normalizer` has profiles for get full
number/date/abbreviation expansion; the other 13 get a local partial pass. The
tests pin that the partial tier DEGRADES rather than raising, since a
normalisation gap must never fail a synthesis request.
"""

from __future__ import annotations

import pytest

from flowtts.dhvaani.config import DhvaaniSettings
from flowtts.dhvaani.text import lang as langmod
from flowtts.dhvaani.text.normalizer import TextNormalizer


@pytest.fixture(scope="module")
def norm():
    return TextNormalizer()


def test_all_27_languages_registered():
    assert len(langmod.DHVAANI_LANGUAGES) == 27
    for code in ("as", "bn", "brx", "doi", "en", "gu", "hi", "hne", "bho", "kn",
                 "ks", "kok", "mai", "mag", "ml", "mni", "mr", "ne", "or", "pa",
                 "raj", "sa", "sat", "sd", "ta", "te", "ur"):
        assert code in langmod.DHVAANI_LANGUAGES


def test_language_tiers_split_14_13():
    full = [x for x in langmod.DHVAANI_LANGUAGES.values() if x.tier == "full"]
    partial = [x for x in langmod.DHVAANI_LANGUAGES.values() if x.tier == "partial"]
    assert len(full) == 14 and len(partial) == 13


@pytest.mark.parametrize(
    "text,expected",
    [
        ("नमस्ते, आपका EMI बकाया है", "hi"),
        ("வணக்கம், நன்றி", "ta"),
        ("നമസ്കാരം", "ml"),
        ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "pa"),
        ("ଓଡ଼ିଆ ଭାଷା", "or"),
        ("Hello there", "en"),
        ("اردو زبان", "ur"),
        ("ಕನ್ನಡ", "kn"),
        ("తెలుగు", "te"),
        ("ગુજરાતી", "gu"),
    ],
)
def test_script_detection(text, expected):
    assert langmod.detect_language(text) == expected


def test_code_switched_text_detects_indic_not_latin():
    """Indic traffic routinely embeds English ('EMI', 'OTP'). Latin must not
    outvote the Indic script or normalisation is routed to the wrong language."""
    assert langmod.detect_language("आपका EMI OTP UPI NEFT है") == "hi"


def test_language_aliases_resolve():
    assert langmod.resolve("HIN") == "hi"
    assert langmod.resolve("en-IN") == "en"
    assert langmod.resolve("mni-Mtei") == "mni"
    assert langmod.resolve(None, "తెలుగు") == "te"
    assert langmod.resolve("nonsense", "") == "hi"


def test_arabic_script_maps_both_digit_sets():
    """Urdu uses Extended Arabic-Indic (U+06F0); Arabic proper uses U+0660.
    Real text mixes them."""
    table = langmod.native_digit_table("ur")
    assert table[0x06F1] == "1"
    assert table[0x0661] == "1"


def test_devanagari_digits_map():
    table = langmod.native_digit_table("hi")
    assert "".join(table[0x0966 + d] for d in range(10)) == "0123456789"


# --- normalisation behaviour (needs the library) ---------------------------
def _requires_lib(norm):
    if not norm.available:
        pytest.skip("indic_tts_normalizer is not installed")


def test_hindi_native_digits_expand(norm):
    _requires_lib(norm)
    out = norm.normalize("मेरे पास १२३ रुपये हैं।", "hi")
    assert "१२३" not in out and "123" not in out  # spelled out, not left as digits


def test_english_currency_and_time(norm):
    _requires_lib(norm)
    out = norm.normalize("Call me at 9:30am, I will pay 1250 rupees.", "en")
    assert "9:30" not in out
    assert "1250" not in out


def test_otp_digits_split(norm):
    _requires_lib(norm)
    out = norm.normalize("Your OTP is 483920", "en")
    assert "483920" not in out


def test_case_preserved_by_default(norm):
    """The library lowercases unconditionally; DhVaani's vocab has both ASCII
    cases, so we bypass that."""
    _requires_lib(norm)
    assert "ABC" in norm.normalize("Pay ABC now", "en")


def test_lowercase_flag_is_honoured():
    s = DhvaaniSettings()
    s.text.lowercase = True
    n = TextNormalizer(s)
    if not n.available:
        pytest.skip("indic_tts_normalizer is not installed")
    assert "ABC" not in n.normalize("Pay ABC now", "en")


@pytest.mark.parametrize("code", ["or", "as", "ne", "sa", "kok", "brx", "doi", "raj",
                                  "ks", "sd", "mni", "sat", "ur"])
def test_partial_tier_degrades_without_raising(norm, code):
    """The 13 languages with no normaliser profile must return usable text."""
    out = norm.normalize("123 test", code)
    assert isinstance(out, str) and out


def test_partial_tier_still_maps_native_digits(norm):
    out = norm.normalize("ମୋ ପାଖରେ ୧୨୩ ଅଛି", "or")
    assert "123" in out          # mapped to ASCII
    assert "୧୨୩" not in out


def test_partial_tier_expands_rupee_sign(norm):
    out = norm.normalize("100 ₹", "or")
    assert "₹" not in out


def test_control_and_emoji_stripped(norm):
    out = norm.normalize("hello﻿world\U0001f600", "en")
    assert "﻿" not in out and "\U0001f600" not in out


def test_zwj_preserved(norm):
    """ZWJ/ZWNJ are meaningful inside Indic conjuncts -- stripping them changes
    pronunciation."""
    out = norm.normalize("क्‍ष", "hi")
    assert "‍" in out


def test_cache_hits(norm):
    text = "कैश टेस्ट वाक्य एक दो तीन"
    before = norm.cache_stats()["hits"]
    norm.normalize(text, "hi")
    norm.normalize(text, "hi")
    assert norm.cache_stats()["hits"] > before


def test_empty_input(norm):
    assert norm.normalize("", "hi") == ""


def test_describe_all_shape():
    rows = langmod.describe_all()
    assert len(rows) == 27
    assert set(rows[0]) == {"code", "name", "native_name", "script", "normalization"}
