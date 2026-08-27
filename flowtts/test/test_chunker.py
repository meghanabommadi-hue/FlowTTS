"""Unit tests for flowtts.synthesis.chunker — smart streaming chunking.

The properties under test are the ones that decide whether the stitched result
sounds like one person: chunks stay inside the size budget, sentences are never
cut in half, a comma is only used once no sentence end fits, nothing lands
inside an atomic span, and no text is lost.

Pure stdlib — no GPU.
"""

from __future__ import annotations

import re

import pytest

from flowtts.synthesis.chunker import (
    CLAUSE,
    END,
    SENTENCE,
    WORD,
    dominant_chars_per_second,
    estimate_duration,
    split_for_streaming,
)
from flowtts.text import normalize_text

NBSP = " "

HINDI = ("नमस्ते. मैं बजाज finance से वाणी बोल रही हूं, एक recorded line के माध्यम से. "
         "क्या मैं customer name से बात कर रही हूं? कृपया आज ही payment करें।")
ENGLISH = ("Hello, this is Dr. Sharma calling from Apollo Hospital. Your appointment "
           "is confirmed for tomorrow at ten thirty. Please arrive fifteen minutes early.")


# One sentence, no terminator until the very end, with commas inside it — the
# only shape for which the comma fallback is the right answer. Must exceed
# target + tolerance or it is legitimately left whole.
LONG_SENTENCE = (
    "आपकी loan application approve हो गई है और पचास हज़ार रुपये सीधे आपके bank account "
    "में transfer कर दिए जाएंगे, जिसमें दो से तीन कार्य दिवस लग सकते हैं और उसके बाद "
    "आपको एक confirmation message भी प्राप्त होगा, जिसमें पूरी जानकारी दी जाएगी और "
    "उसके साथ आपके loan की सारी शर्तें भी विस्तार से बताई जाएंगी, ताकि आपको किसी "
    "प्रकार की कोई असुविधा न हो और आप समय पर अपनी किस्तें जमा कर सकें"
)
PARAGRAPH = " ".join([ENGLISH, HINDI, ENGLISH, HINDI])


def _rejoin(chunks) -> str:
    """All chunk text with whitespace removed, to prove nothing was dropped."""
    return re.sub(r"\s+", "", "".join(c.text for c in chunks))


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------
def test_empty_input():
    assert split_for_streaming("") == []
    assert split_for_streaming("   ") == []


def test_short_input_is_one_chunk():
    chunks = split_for_streaming("Hello.")
    assert [c.text for c in chunks] == ["Hello."]
    assert chunks[0].boundary == END


@pytest.mark.parametrize("text", [HINDI, ENGLISH, LONG_SENTENCE])
def test_no_text_is_lost(text):
    assert _rejoin(split_for_streaming(text)) == re.sub(r"\s+", "", text)


@pytest.mark.parametrize("text", [HINDI, ENGLISH])
def test_no_empty_chunks(text):
    assert all(c.text.strip() for c in split_for_streaming(text))


def test_last_chunk_is_always_END():
    for text in (HINDI, ENGLISH, LONG_SENTENCE, "Hi."):
        assert split_for_streaming(text)[-1].boundary == END


# ---------------------------------------------------------------------------
# The size budget: at most target + tolerance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [HINDI, ENGLISH, LONG_SENTENCE, PARAGRAPH])
def test_chunks_respect_the_budget(text):
    """Only an unbreakable run may exceed target + tolerance."""
    for chunk in split_for_streaming(text, target_chars=200, tolerance_chars=50):
        assert len(chunk) <= 250 or " " not in chunk.text, \
            f"{len(chunk)} chars: {chunk.text[:80]!r}"


def test_ordinary_utterances_are_a_single_chunk():
    """Under 250 characters there is no reason to cut at all — and every cut is
    a seam the listener can hear."""
    assert len(split_for_streaming(HINDI)) == 1
    assert len(split_for_streaming(ENGLISH)) == 1


def test_budget_is_configurable():
    assert len(split_for_streaming(PARAGRAPH, target_chars=80, tolerance_chars=20)) > \
           len(split_for_streaming(PARAGRAPH, target_chars=200, tolerance_chars=50))


# ---------------------------------------------------------------------------
# Boundary priority: sentence, then comma, then word gap
# ---------------------------------------------------------------------------
def test_sentences_are_packed_whole():
    """A sentence shorter than the budget is never cut in half."""
    text = " ".join(["This is sentence number %d and it is of a moderate length." % i
                     for i in range(8)])
    chunks = split_for_streaming(text, target_chars=200, tolerance_chars=50)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.boundary == SENTENCE
        assert chunk.text.rstrip().endswith("."), chunk.text


def test_comma_is_only_used_when_no_sentence_end_fits():
    """A comma is a breath inside a thought — a last resort, not a first choice."""
    # Sentences available well inside the budget: never a comma.
    packed = split_for_streaming(
        "Alpha, beta, gamma. Delta, epsilon, zeta. Eta, theta, iota. Kappa, lambda, mu.",
        target_chars=40, tolerance_chars=10)
    assert all(c.boundary in (SENTENCE, END) for c in packed), \
        [(c.text, c.boundary) for c in packed]

    # One long sentence with commas and no terminator: now the comma is used.
    chunks = split_for_streaming(LONG_SENTENCE, target_chars=200, tolerance_chars=50)
    assert len(chunks) > 1
    assert chunks[0].boundary == CLAUSE, [(c.text[:40], c.boundary) for c in chunks]


def test_commas_can_be_disabled_entirely():
    chunks = split_for_streaming(LONG_SENTENCE, target_chars=200, tolerance_chars=50,
                                 split_on_clause=False)
    assert all(c.boundary != CLAUSE for c in chunks)


def test_word_gap_only_when_there_is_no_punctuation_at_all():
    text = " ".join(["word"] * 200)
    chunks = split_for_streaming(text, target_chars=100, tolerance_chars=20)
    assert len(chunks) > 1
    assert all(c.boundary in (WORD, END) for c in chunks)


def test_abbreviation_period_is_not_a_sentence_end():
    chunks = split_for_streaming("Please meet Dr. Sharma at the clinic today.",
                                 target_chars=20, tolerance_chars=5)
    assert not any(c.text.rstrip().endswith("Dr.") for c in chunks), chunks


@pytest.mark.parametrize("atomic,text", [
    ("[laughter]", "Well [laughter] that was something else entirely, was it not."),
    ("[B EY1 S]", "He plays the [B EY1 S] guitar in the band every single weekend."),
    ("12:30", "The meeting starts at 12:30 sharp tomorrow in the main conference room."),
    ("3.14159", "The value of pi is 3.14159 which most people never need to recall."),
])
def test_atomic_spans_are_never_split(atomic, text):
    chunks = split_for_streaming(text, target_chars=20, tolerance_chars=5)
    assert any(atomic in c.text for c in chunks), \
        f"{atomic!r} was split across chunks: {[c.text for c in chunks]}"


def test_expanded_numerals_are_never_split():
    """The NBSP the normalizer writes inside a numeral must hold across chunking."""
    text = normalize_text("आपका बकाया ₹2,500 है, कृपया आज ही payment करें।", "hi")
    chunks = split_for_streaming(text, target_chars=20, tolerance_chars=5)
    for word in (w for w in text.split(" ") if NBSP in w):
        assert any(word in c.text for c in chunks), \
            f"numeral {word!r} was split: {[c.text for c in chunks]}"


def test_first_chunk_can_be_shrunk_for_ttfb():
    chunks = split_for_streaming(PARAGRAPH, target_chars=200, first_chunk_chars=60)
    assert len(chunks[0]) < len(chunks[1])


# ---------------------------------------------------------------------------
# Duration model
# ---------------------------------------------------------------------------
def test_indic_script_is_slower_per_character_than_latin():
    """Equal character counts are not equal audio; the batcher's estimate needs that."""
    assert dominant_chars_per_second("नमस्ते दुनिया") < dominant_chars_per_second("hello world")


def test_estimate_duration_scales_with_length():
    assert estimate_duration("hello there " * 10) > estimate_duration("hello there") * 8


def test_empty_duration_is_zero():
    assert estimate_duration("") == 0.0


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "x" * 400,                        # one unbreakable token
    "word " * 500,                     # no punctuation at all
    "..........",                      # punctuation only
    "क" * 300,
    "a,b,c,d,e,f,g,h" * 30,
    "Hi. Ok. Yes. No. Sure.",          # many tiny sentences
])
def test_pathological_input_still_chunks(text):
    chunks = split_for_streaming(text)
    assert chunks
    assert all(c.text.strip() for c in chunks)
    assert _rejoin(chunks) == re.sub(r"\s+", "", text)
