"""Pipeline position: TEXT PREPROCESSING — script detection + code-mixed splitting.

Role in pipeline:
  Real call-centre input is code-mixed ("आपका balance ₹2,500 है"). Normalizing
  that whole string as Hindi speaks "2,500" in Devanagari words but leaves the
  Latin run alone; normalizing it as English does the reverse. Neither is right.

  This module segments text into maximal runs of one script, so the pipeline can
  normalize each run in the language that script implies, then stitch the result
  back together in the original order.

      "आपका balance ₹2,500 है"
        → [Segment("आपका ", "hi"), Segment("balance ", "en-IN"),
           Segment("₹2,500 ", "hi"), Segment("है", "hi")]

  Script-neutral characters (digits, punctuation, whitespace, currency signs)
  carry no script of their own, so they attach to the run they are adjacent to —
  otherwise every number would split its sentence into three fragments.

Pure stdlib: no GPU, no model, unit-testable anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from flowtts.text.languages import SCRIPT_DEFAULT_LANGUAGE, get_profile, resolve_language

# Unicode ranges per script we care about. Ordered by how specific they are;
# the first hit wins.
_SCRIPT_RANGES: list[tuple[str, tuple[int, int]]] = [
    ("deva", (0x0900, 0x097F)),
    ("deva", (0xA8E0, 0xA8FF)),   # Devanagari Extended
    ("beng", (0x0980, 0x09FF)),
    ("guru", (0x0A00, 0x0A7F)),
    ("gujr", (0x0A80, 0x0AFF)),
    ("orya", (0x0B00, 0x0B7F)),
    ("taml", (0x0B80, 0x0BFF)),
    ("telu", (0x0C00, 0x0C7F)),
    ("knda", (0x0C80, 0x0CFF)),
    ("mlym", (0x0D00, 0x0D7F)),
    ("sinh", (0x0D80, 0x0DFF)),
    ("arab", (0x0600, 0x06FF)),
    ("arab", (0x0750, 0x077F)),
    ("arab", (0xFB50, 0xFDFF)),
    ("arab", (0xFE70, 0xFEFF)),
    ("olck", (0x1C50, 0x1C7F)),
    ("mtei", (0xABC0, 0xABFF)),
    ("latn", (0x0041, 0x005A)),
    ("latn", (0x0061, 0x007A)),
    ("latn", (0x00C0, 0x024F)),
    ("cjk",  (0x4E00, 0x9FFF)),
    ("hang", (0xAC00, 0xD7AF)),
    ("kana", (0x3040, 0x30FF)),
    ("cyrl", (0x0400, 0x04FF)),
    ("grek", (0x0370, 0x03FF)),
    ("thai", (0x0E00, 0x0E7F)),
]

# Characters that belong to no script and must not break a run: whitespace,
# ASCII digits, ASCII punctuation, general punctuation, currency signs, the
# zero-width joiners Indic scripts use internally, and the danda (U+0964/0965) --
# which lives in the Devanagari block but is shared punctuation across Indic
# scripts, so treating it as Devanagari would split every Tamil sentence.
_NEUTRAL_RE = re.compile(
    "["
    r"\s\d"
    r"!-/:-@\[-`{-~"          # ASCII punctuation
    "\u00a0-\u00bf"           # Latin-1 spaces, punctuation and signs
    "\u2000-\u206f"           # general punctuation (dashes, quotes, ellipsis)
    "\u20a0-\u20cf"           # currency symbols
    "\u2190-\u2bff"           # arrows, math operators, misc symbols
    "\u200b-\u200d\ufeff"     # ZWSP / ZWNJ / ZWJ / BOM
    "\u0964\u0965"            # danda, double danda
    "]"
)


# A number as written: digits plus the separators that stay inside one token.
_DIGIT_RUN_RE = re.compile(r"[₹$€£¥]?\d[\d,.\-:]*\d%?|[₹$€£¥]?\d%?")


def _is_latin_letter(ch: str) -> bool:
    return bool(ch) and script_of_char(ch) == "latn"


def script_of_char(ch: str) -> Optional[str]:
    """Return the script key for *ch*, or None if it is script-neutral."""
    if _NEUTRAL_RE.match(ch):
        return None
    cp = ord(ch)
    for name, (lo, hi) in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    return None


def dominant_script(text: str) -> Optional[str]:
    """Return the script most characters of *text* belong to, or None."""
    counts: dict[str, int] = {}
    for ch in text:
        s = script_of_char(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


def detect_language(text: str, default: str = "en") -> str:
    """Best-effort language for *text* from its script mix.

    This resolves a *script*, not a language — Devanagari could be Hindi,
    Marathi, Nepali or Sanskrit. It returns the highest-prior language for that
    script (see SCRIPT_DEFAULT_LANGUAGE) and is only ever used when the caller
    supplied no language at all.

    A non-Latin script wins over Latin whenever both are present, even if Latin
    has more characters: code-mixed input like "आपका balance ₹2,500 है" is Hindi
    speech carrying English loanwords, and synthesizing it as English mangles
    the Devanagari.
    """
    counts: dict[str, int] = {}
    for ch in text:
        s = script_of_char(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return resolve_language(default)
    non_latin = {k: v for k, v in counts.items() if k != "latn"}
    script = max(non_latin or counts, key=lambda k: (non_latin or counts)[k])
    return SCRIPT_DEFAULT_LANGUAGE.get(script, resolve_language(default))


@dataclass
class Segment:
    """One maximal single-script run of the input, with the language to use."""

    text: str
    language: str
    script: Optional[str]


def split_by_script(
    text: str,
    base_language: str,
    *,
    latin_language: str | None = None,
) -> list[Segment]:
    """Split *text* into script runs, tagging each with the language to normalize it in.

    ``base_language`` is the request's declared language and wins for any run
    written in that language's own script. Latin runs inside an Indic sentence
    are normalized as ``latin_language`` (default: Indian English, so "2,500"
    inside an English run reads with lakh/crore grouping like the rest of the
    utterance). Runs in a third script fall back to that script's default
    language.

    Neutral characters attach to the preceding run — or to the following one at
    the very start of the string — so numbers and punctuation never split a
    sentence.
    """
    if not text:
        return []

    base_language = resolve_language(base_language)
    base_script = get_profile(base_language).script or "latn"
    latin_language = resolve_language(latin_language or "en-IN")

    def _language_for(script: Optional[str]) -> str:
        if script is None or script == base_script:
            return base_language
        if script == "latn":
            return latin_language
        return SCRIPT_DEFAULT_LANGUAGE.get(script, base_language)

    # 1. Label every character with its script (None for neutral).
    labels = [script_of_char(ch) for ch in text]

    # 2. Give neutral characters the script of the nearest scripted character to
    #    their left; leading neutrals inherit from the right instead.
    resolved: list[Optional[str]] = [None] * len(labels)
    last: Optional[str] = None
    for i, s in enumerate(labels):
        if s is not None:
            last = s
        resolved[i] = last
    if last is None:
        # Entire string is script-neutral (pure digits/punctuation).
        return [Segment(text, base_language, None)]
    first_scripted = next((s for s in labels if s is not None), None)
    for i, s in enumerate(labels):
        if resolved[i] is None:
            resolved[i] = first_scripted
        else:
            break

    # 3. Give free-standing numbers the sentence's own language. Digits carry no
    #    script, so step 2 hands them to whichever run precedes them — which
    #    makes "உங்கள் OTP 4821" read the code in English inside a Tamil
    #    sentence. Digits glued to Latin letters ("abc123.com", "v2", "3rd") are
    #    left alone, or URL and version expansion would be torn apart.
    for m in _DIGIT_RUN_RE.finditer(text):
        before = text[m.start() - 1] if m.start() else ""
        after = text[m.end()] if m.end() < len(text) else ""
        if _is_latin_letter(before) or _is_latin_letter(after):
            continue
        for i in range(m.start(), m.end()):
            resolved[i] = base_script

    # 4. Merge adjacent identical labels into segments.
    segments: list[Segment] = []
    start = 0
    for i in range(1, len(text) + 1):
        if i == len(text) or resolved[i] != resolved[start]:
            script = resolved[start]
            segments.append(Segment(text[start:i], _language_for(script), script))
            start = i
    return segments


def is_code_mixed(text: str) -> bool:
    """True if *text* contains characters from more than one script."""
    seen = set()
    for ch in text:
        s = script_of_char(ch)
        if s:
            seen.add(s)
            if len(seen) > 1:
                return True
    return False
