"""Pipeline position: TEXT PREPROCESSING — URLs, emails, phone numbers, OTPs.

Ported from github.com/Ajaj-Ali/text_preprocessor_for_TTS
(`indic_tts_normalizer.contact_info`), with two changes driven by Indian
voice-bot traffic:

  • the digit-run threshold is configurable and defaults to 4, not 6. Upstream
    reads 6+ digit runs digit-by-digit and leaves a bare 4-digit OTP to be read
    as a cardinal ("one thousand two hundred thirty four") — which is wrong for
    the single most common thing an Indian IVR ever reads aloud. Comma-grouped
    amounts ("1,234") still take the cardinal path, because numbers.py sees them
    first with their commas intact.
  • Indian mobile numbers are grouped 5+5 when spoken, with a short pause
    between groups, instead of ten flat digits.

URLs and emails are spelled out in English regardless of target language —
that is how they are spoken in real multilingual input.
"""

from __future__ import annotations

import re

from flowtts.text.languages import get_profile
from flowtts.text.numbers import say_digits

_URL_RE = re.compile(
    r"\b(?:https?://)?(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9-]*(?:\.[a-zA-Z0-9-]+)*"
    r"\.(?:com|in|org|net|co|io|gov|edu|info|biz|me|app|dev|ai|xyz|[a-zA-Z]{2})"
    r"(?:/[^\s]*)?\b"
)
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")

# An Indian mobile number: optional +91/0 prefix, then 10 digits starting 6-9,
# tolerating the spaces and dashes people actually type.
_INDIAN_MOBILE_RE = re.compile(
    r"(?<!\d)(?:\+?91[\s-]?|0)?([6-9]\d{9})(?!\d)"
)

_URL_CHAR_WORDS = [
    (".", " dot "),
    ("/", " slash "),
    ("-", " dash "),
    ("_", " underscore "),
    (":", " colon "),
]

_EN_DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six",
                   "seven", "eight", "nine")
_DIGITS_RE = re.compile(r"\d+")


def _spell_digits_en(text: str) -> str:
    """Read digits inside a URL/email one at a time, always in English.

    Done here rather than left to numbers.py for two reasons: "abc123.com" is
    a name, not a quantity, so "one hundred and twenty-three" is wrong; and
    leaving the digits glued to letters makes the later cardinal pass produce
    "abcone hundred and twenty-three".
    """
    return _DIGITS_RE.sub(
        lambda m: " " + " ".join(_EN_DIGIT_WORDS[int(d)] for d in m.group(0)) + " ",
        text,
    )


def _spell_url(url: str) -> str:
    spelled = re.sub(r"^https?://", "", url)
    spelled = re.sub(r"^www\.", "", spelled)
    for char, word in _URL_CHAR_WORDS:
        spelled = spelled.replace(char, word)
    return " ".join(_spell_digits_en(spelled).split())


def _spell_email(email: str) -> str:
    local, _, domain = email.partition("@")
    for char, word in _URL_CHAR_WORDS:
        local = local.replace(char, word)
        domain = domain.replace(char, word)
    local = " ".join(_spell_digits_en(local).split())
    domain = " ".join(_spell_digits_en(domain).split())
    return f"{local} at {domain}"


def expand_urls_and_emails(text: str) -> str:
    text = _EMAIL_RE.sub(lambda m: _spell_email(m.group(0)), text)
    return _URL_RE.sub(lambda m: _spell_url(m.group(0)), text)


def expand_phone_numbers(text: str, lang: str) -> str:
    """Read Indian mobile numbers as two five-digit groups with a pause."""
    profile = get_profile(lang)

    def _sub(m: re.Match) -> str:
        digits = m.group(1)
        return f"{say_digits(digits[:5], profile)}, {say_digits(digits[5:], profile)}"

    return _INDIAN_MOBILE_RE.sub(_sub, text)


def split_long_digit_runs(text: str, lang: str, *, min_digits: int = 4) -> str:
    """Read runs of *min_digits* or more digits one digit at a time.

    Bare runs that long are codes — OTPs, PINs, account and reference numbers —
    not quantities. Real quantities reach this stage comma-grouped and are
    consumed by numbers.py before it.
    """
    profile = get_profile(lang)
    pattern = re.compile(rf"(?<!\d)\d{{{min_digits},}}(?!\d)")
    return pattern.sub(lambda m: say_digits(m.group(0), profile), text)
