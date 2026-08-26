"""Pipeline position: TEXT PREPROCESSING — numeral → words (pure stdlib + optional backends).

Role in pipeline:
  Turns every numeric token into speakable words in the segment's language:
  integers, decimals, negatives, percentages, currency, and English ordinals.

Ported from github.com/Ajaj-Ali/text_preprocessor_for_TTS
(`indic_tts_normalizer.numbers`), with the backend lookup made total. Upstream
imports ``num2words`` and ``num_to_words`` at module import time and calls
whichever the profile names — so a language the installed backend does not
cover raises, and a box without those wheels cannot import the package at all.

Here the backends are optional and tried in order, per language:

    indic_num2words → num2words → digit-by-digit in the target script → digits

The last two rungs matter for the languages no cardinal backend covers
(Santali, Manipuri, Bodo, Sanskrit, Kashmiri, Sindhi, Assamese, Tulu). Reading
"2500" digit-by-digit in Santali is worse prosody than a proper cardinal but is
still correct speech; leaving the numeral raw makes the model either spell it in
English or hallucinate, which is much worse.
"""

from __future__ import annotations

import logging
import re

from flowtts.text.languages import LanguageProfile, get_profile, resolve_language

logger = logging.getLogger(__name__)

_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)

# One numeric token: optional currency sign, optional minus, an integer with
# either Indian (2-digit) or Western (3-digit) comma grouping, an optional
# fractional part, and an optional trailing percent sign.
_NUMBER_TOKEN_RE = re.compile(
    r"(?P<currency>[₹$€£¥])?"
    r"(?P<sign>-)?"
    r"(?P<int>\d{1,3}(?:,\d{2,3})+|\d+)"
    r"(?:\.(?P<decimal>\d+))?"
    r"(?P<percent>%)?"
)

# Above this many digits a cardinal reading stops being useful speech
# ("nine hundred eighty seven billion…"); read it digit by digit instead.
_MAX_CARDINAL_DIGITS = 15

# The words of a single expanded numeral are joined with a non-breaking space
# instead of a plain one. "2,500" becomes four Devanagari words that mean one
# number, and a chunk boundary landing between "दो" and "हज़ार" is audible as two
# separate numbers. The chunker treats NBSP as non-splittable; the synthesizer
# converts it back to a plain space before the text reaches the model, so
# nothing unusual is ever tokenized.
NBSP = "\u00a0"


def _join(pieces) -> str:
    return NBSP.join(p for p in pieces if p)

_BACKENDS: dict[str, object] = {}


def _num2words():
    if "num2words" not in _BACKENDS:
        try:
            from num2words import num2words as fn
            _BACKENDS["num2words"] = fn
        except Exception:  # noqa: BLE001 — optional dependency
            logger.info("num2words not installed; English numbers fall back to digits")
            _BACKENDS["num2words"] = None
    return _BACKENDS["num2words"]


def _indic_num2words():
    if "indic" not in _BACKENDS:
        try:
            from num_to_words import num_to_word as fn
            _BACKENDS["indic"] = fn
        except Exception:  # noqa: BLE001 — optional dependency
            logger.info("indic-num2words not installed; Indic numbers fall back to digits")
            _BACKENDS["indic"] = None
    return _BACKENDS["indic"]


def say_digits(n: str, profile: LanguageProfile) -> str:
    """Read a run of digits one at a time in *profile*'s language."""
    words = profile.digit_words
    if not words:
        return _join(n)
    return _join(words[int(d)] for d in n if d.isdigit())


def say_integer(n: int, profile: LanguageProfile) -> str:
    """Speak one non-negative integer, falling back through the backend chain."""
    text = str(n)
    if len(text) > _MAX_CARDINAL_DIGITS:
        return say_digits(text, profile)

    if profile.number_backend == "indic_num2words":
        fn = _indic_num2words()
        if fn is not None:
            try:
                return _join(
                    fn(n, lang=profile.number_lang, separator=" ", combiner=" ").split()
                )
            except Exception:  # noqa: BLE001 — backend may not cover this language
                logger.debug("indic-num2words failed for lang=%s", profile.number_lang)

    if profile.number_backend in ("num2words", "indic_num2words"):
        fn = _num2words()
        if fn is not None:
            try:
                return _join(fn(n, lang=profile.number_lang).split())
            except Exception:  # noqa: BLE001 — num2words raises on unknown lang
                logger.debug("num2words failed for lang=%s", profile.number_lang)

    return say_digits(text, profile)


def say_ordinal(n: int, profile: LanguageProfile) -> str:
    """Speak an ordinal ("21st"). Only num2words languages have real ordinals."""
    fn = _num2words()
    if fn is not None:
        try:
            return _join(fn(n, lang=profile.number_lang, to="ordinal").split())
        except Exception:  # noqa: BLE001
            pass
    return say_integer(n, profile)


def _expand_match(match: re.Match, profile: LanguageProfile) -> str:
    words = profile.words
    int_part = match.group("int").replace(",", "")
    decimal_part = match.group("decimal")
    sign = match.group("sign")
    currency = match.group("currency")
    percent = match.group("percent")

    pieces: list[str] = []
    if sign:
        pieces.append(words.get("minus", "minus"))
    pieces.append(say_integer(int(int_part), profile))
    if decimal_part:
        pieces.append(words.get("point", "point"))
        pieces.append(say_digits(decimal_part, profile))
    if currency:
        pieces.append(words.get("currency", {}).get(currency, currency))
    if percent:
        pieces.append(words.get("percent", "percent"))
    return _join(pieces)


def _dedupe_currency(text: str, profile: LanguageProfile) -> str:
    """Collapse a currency word we just emitted next to one already in the text.

    "\u20b91,500 \u099f\u0995\u09be" is written the way people write it, with both the sign
    and the word. Expanding the sign then yields "... \u099f\u0995\u09be \u099f\u0995\u09be".
    """
    for word in set(profile.words.get("currency", {}).values()):
        pattern = re.compile(rf"({re.escape(word)})([\s ]+\1)+")
        text = pattern.sub(r"\1", text)
    return text


def expand_numbers(text: str, lang: str) -> str:
    """Expand every numeric token in *text* (decimals, negatives, %, currency)."""
    profile = get_profile(lang)

    if profile.number_backend == "num2words" and _num2words() is not None:
        text = _ORDINAL_RE.sub(lambda m: say_ordinal(int(m.group(1)), profile), text)

    text = _NUMBER_TOKEN_RE.sub(lambda m: _expand_match(m, profile), text)
    return _dedupe_currency(text, profile)


def words_for(lang: str) -> dict:
    """The point/minus/percent/currency wording table for *lang*."""
    return get_profile(resolve_language(lang)).words
