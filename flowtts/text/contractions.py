"""Pipeline position: TEXT PREPROCESSING — English contraction expansion.

Ported verbatim in behaviour from github.com/Ajaj-Ali/text_preprocessor_for_TTS
(`indic_tts_normalizer.contractions`). Applied only to English/Indian-English
segments — including the Latin runs inside a code-mixed Hindi sentence, where
"I'll" is just as likely to appear as in a pure English one.

Deliberately conservative: only unambiguous, high-frequency contractions. The
expansion preserves the input's casing pattern so "Don't" does not become
"do not" mid-sentence when the rest of the pipeline is case-preserving.
"""

from __future__ import annotations

import re

CONTRACTIONS = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "here's": "here is",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mightn't": "might not",
    "might've": "might have",
    "mustn't": "must not",
    "must've": "must have",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "what're": "what are",
    "when's": "when is",
    "where's": "where is",
    "who's": "who is",
    "who'll": "who will",
    "won't": "will not",
    "wouldn't": "would not",
    "would've": "would have",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# Both the ASCII apostrophe and U+2019 (what most editors and phone keyboards
# actually produce) must match, or half of real traffic slips through.
_APOSTROPHES = "['’]"


def _key_pattern(key: str) -> str:
    return re.escape(key).replace("'", _APOSTROPHES)


_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(_key_pattern(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _match_case(source: str, expansion: str) -> str:
    """Give *expansion* the casing pattern of *source* ("Don't" → "Do not")."""
    if source.isupper() and len(source) > 1:
        return expansion.upper()
    if source[:1].isupper():
        return expansion[:1].upper() + expansion[1:]
    return expansion


def expand_contractions(text: str) -> str:
    """Expand English contractions, preserving the original casing pattern."""

    def _sub(m: re.Match) -> str:
        raw = m.group(0)
        key = raw.lower().replace("’", "'")
        expansion = CONTRACTIONS.get(key)
        return _match_case(raw, expansion) if expansion else raw

    return _CONTRACTION_RE.sub(_sub, text)
