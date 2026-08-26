"""Pipeline position: TEXT PREPROCESSING — structural cleanup (pure stdlib).

Role in pipeline:
  First stage of the normalizer. Strips HTML, control characters, emoji and
  formatting noise so later stages see plain speakable text.

Ported from github.com/Ajaj-Ali/text_preprocessor_for_TTS
(`indic_tts_normalizer.sanitize`), with one behavioural change that matters for
this model: upstream deletes every bracket character, which would destroy
OmniVoice's inline control syntax — ``[laughter]``, ``[dissatisfaction-hnn]``,
``[B EY1 S]`` (ARPAbet pronunciation). Those tags are what "expose all
parameters" means at the text level, so this module protects them: they are
lifted out before cleanup and restored afterwards, byte-for-byte.
"""

from __future__ import annotations

import re

# OmniVoice's inline control syntax: anything in square brackets that does not
# itself contain a bracket. Emotion tokens, non-verbal tags and ARPAbet
# pronunciations all take this form and are passed to the model verbatim.
BRACKET_TAG_RE = re.compile(r"\[[^\[\]]*\]")

# Sentinel used while a tag is parked. Chosen to survive every later stage:
# no digits (numbers.py), no letters (abbreviations.py), no symbols table entry,
# and no whitespace (collapse_whitespace).
_TAG_SENTINEL = ""

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_CHAR_RE = re.compile(r"[\r\n\t\x0b\x0c]")
# C0/C1 control characters that are not the whitespace handled above. Leaving
# these in silently corrupts the model's tokenizer.
_UNPRINTABLE_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]")
_BULLET_RE = re.compile(r"[•‣◦▪▫·∙]")
_BRACKET_RE = re.compile(r"[()\[\]{}<>]")

# Broad but not exhaustive coverage of the common emoji / pictograph blocks.
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "\U0000fe0f"
    "\U0001f000-\U0001f0ff"
    "]+",
    flags=re.UNICODE,
)

# A punctuation character repeated, with or without whitespace between the
# repeats: "!!!" -> "!", "!  !  !" -> "!".
_REPEATED_PUNCT_RE = re.compile(r"([!?.,;:\-_=+*\\|@#$%^&~`/])(?:\s*\1)+")
_MULTI_SLASH_RE = re.compile(r"/{3,}")
# Every whitespace character EXCEPT the non-breaking space, which numbers.py
# uses to bind the words of one numeral together for the chunker.
_WHITESPACE_RE = re.compile(r"[^\S\u00a0]+")


def extract_tags(text: str) -> tuple[str, list[str]]:
    """Replace every ``[...]`` control tag with a sentinel; return (text, tags)."""
    tags: list[str] = []

    def _park(m: re.Match) -> str:
        tags.append(m.group(0))
        return _TAG_SENTINEL

    return BRACKET_TAG_RE.sub(_park, text), tags


def restore_tags(text: str, tags: list[str]) -> str:
    """Put the tags extracted by :func:`extract_tags` back, in order."""
    if not tags:
        return text
    out: list[str] = []
    it = iter(tags)
    for piece in text.split(_TAG_SENTINEL):
        out.append(piece)
        nxt = next(it, None)
        if nxt is not None:
            out.append(nxt)
    return "".join(out)


def strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub(" ", text)


def strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub(" ", text)


def normalize_bullets_and_brackets(text: str) -> str:
    text = _BULLET_RE.sub(" ", text)
    return _BRACKET_RE.sub(" ", text)


def collapse_repeated_punctuation(text: str) -> str:
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    return _MULTI_SLASH_RE.sub("//", text)


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def sanitize(text: str, *, keep_brackets: bool = False) -> str:
    """Strip HTML, emoji and formatting noise; normalize whitespace.

    ``keep_brackets=True`` leaves bracket characters alone — used when the
    caller has already parked OmniVoice control tags and only ordinary
    parentheses remain to clean.
    """
    text = strip_html(text)
    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = _UNPRINTABLE_RE.sub("", text)
    text = strip_emoji(text)
    text = _BULLET_RE.sub(" ", text) if keep_brackets else normalize_bullets_and_brackets(text)
    text = collapse_repeated_punctuation(text)
    return collapse_whitespace(text)


def light_sanitize(text: str) -> str:
    """Minimal, multilingual-safe cleanup: control characters out, nothing else.

    This is what runs when normalization is disabled. OmniVoice is a 600+
    language model with its own subword tokenizer, so dropping "unusual" scripts
    here silently empties Bengali/Assamese/Tamil input and produces garbage
    audio. Every language's letters, marks, digits and punctuation pass through
    untouched.
    """
    if not text:
        return ""
    text = _UNPRINTABLE_RE.sub("", text)
    return text.strip()
