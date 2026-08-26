"""Unit tests for flowtts.synthesis.chunker — smart streaming chunking.

The properties under test are the ones that decide latency and whether the
stitched result sounds like one utterance: the first chunk is short, no chunk
lands inside something atomic, chunk text is preserved, and the boundary chosen
is the best-quality one available inside the budget.

Pure stdlib — no GPU.
"""

from __future__ import annotations

import re

import pytest

from flowtts.synthesis.chunker import (
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


def _rejoin(chunks: list[str]) -> str:
    """Chunk text minus the punctuation the chunker is allowed to append."""
    return re.sub(r"[\s,]+", "", "".join(chunks))


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------
def test_empty_input():
    assert split_for_streaming("") == []
    assert split_for_streaming("   ") == []


def test_short_input_is_one_chunk():
    assert split_for_streaming("Hello.") == ["Hello."]


@pytest.mark.parametrize("text", [HINDI, ENGLISH])
def test_no_text_is_lost(text):
    chunks = split_for_streaming(text)
    assert _rejoin(chunks) == re.sub(r"[\s,]+", "", text)


@pytest.mark.parametrize("text", [HINDI, ENGLISH])
def test_no_empty_chunks(text):
    assert all(chunk.strip() for chunk in split_for_streaming(text))


# ---------------------------------------------------------------------------
# The latency lever
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [HINDI, ENGLISH])
def test_first_chunk_is_short(text):
    """TTFB is the cost of chunk 0 alone, so chunk 0 must stay small."""
    chunks = split_for_streaming(text)
    assert len(chunks) > 1
    assert estimate_duration(chunks[0]) <= 2.8, chunks[0]


def test_first_chunk_may_stretch_to_reach_a_real_boundary():
    """A word-gap cut sounds broken; a slightly longer natural one does not."""
    text = "[laughter] You really got me. I did not see that coming at all."
    first = split_for_streaming(text)[0]
    assert first.endswith("."), first
    assert "[laughter]" in first


def test_later_chunks_are_larger_than_the_first():
    chunks = split_for_streaming(HINDI + " " + HINDI)
    assert len(chunks) >= 3
    assert estimate_duration(chunks[1]) >= estimate_duration(chunks[0])


def test_chunk_count_stays_bounded():
    chunks = split_for_streaming(" ".join(["This is a sentence."] * 200))
    assert 1 < len(chunks) <= 128


# ---------------------------------------------------------------------------
# Boundary quality
# ---------------------------------------------------------------------------
def test_sentence_boundaries_are_preferred():
    text = "One two three four five. Six seven eight nine ten. Eleven twelve."
    chunks = split_for_streaming(text, first_chunk_seconds=1.6, chunk_seconds=2.5)
    assert all(c.rstrip().endswith((".", ",")) for c in chunks), chunks


def test_abbreviation_period_is_not_a_sentence_end():
    chunks = split_for_streaming("Please meet Dr. Sharma at the clinic today.",
                                 first_chunk_seconds=1.0)
    assert not any(c.rstrip().endswith("Dr.") for c in chunks), chunks


@pytest.mark.parametrize("atomic,text", [
    ("[laughter]", "Well [laughter] that was something else entirely, was it not."),
    ("[B EY1 S]", "He plays the [B EY1 S] guitar in the band every single weekend."),
    ("12:30", "The meeting starts at 12:30 sharp tomorrow in the main conference room."),
    ("3.14159", "The value of pi is 3.14159 which most people never need to recall."),
])
def test_atomic_spans_are_never_split(atomic, text):
    chunks = split_for_streaming(text, first_chunk_seconds=0.6, chunk_seconds=1.2)
    assert any(atomic in chunk for chunk in chunks), \
        f"{atomic!r} was split across chunks: {chunks}"


def test_expanded_numerals_are_never_split():
    """The NBSP the normalizer writes inside a numeral must hold across chunking."""
    text = normalize_text("आपका बकाया ₹2,500 है, कृपया आज ही payment करें।", "hi")
    chunks = split_for_streaming(text, first_chunk_seconds=0.7, chunk_seconds=1.2)
    for chunk in chunks:
        # A chunk either contains a whole NBSP-joined group or none of it.
        assert not chunk.startswith(NBSP) and not chunk.endswith(NBSP), chunk
    numeral = [w for w in text.split(" ") if NBSP in w]
    for word in numeral:
        assert any(word in chunk for chunk in chunks), \
            f"numeral {word!r} was split: {chunks}"


def test_non_final_chunks_are_punctuated():
    """An unterminated chunk makes OmniVoice trail off at the seam."""
    chunks = split_for_streaming(HINDI)
    for chunk in chunks[:-1]:
        assert chunk[-1] in ".?!।॥…,;:—–", chunk


def test_terminate_chunks_can_be_turned_off():
    chunks = split_for_streaming("one two three four five six seven eight nine ten",
                                 first_chunk_seconds=0.4, chunk_seconds=0.6,
                                 terminate_chunks=False)
    assert not any(c.endswith(",") for c in chunks)


# ---------------------------------------------------------------------------
# Duration model
# ---------------------------------------------------------------------------
def test_indic_script_is_slower_per_character_than_latin():
    """Equal character counts are not equal audio; the budget must know that."""
    assert dominant_chars_per_second("नमस्ते दुनिया") < dominant_chars_per_second("hello world")


def test_estimate_duration_scales_with_length():
    short = estimate_duration("hello there")
    long = estimate_duration("hello there " * 10)
    assert long > short * 8


def test_empty_duration_is_zero():
    assert estimate_duration("") == 0.0


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "x" * 400,                       # one unbreakable token
    "word " * 500,                    # no punctuation at all
    "..........",                     # punctuation only
    "क" * 300,
    "a,b,c,d,e,f,g,h" * 30,
])
def test_pathological_input_still_chunks(text):
    chunks = split_for_streaming(text)
    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert _rejoin(chunks) == re.sub(r"[\s,]+", "", text)


# ---------------------------------------------------------------------------
# The minimum-length floor — below ~1 s of audio the model returns silence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    HINDI, ENGLISH,
    "ᱡᱚᱦᱟᱨ, ᱟᱢᱟᱜ ᱵᱮᱞᱮᱱᱥ ᱢᱤᱫ ᱦᱟᱡᱟᱨ ᱵᱟᱨ ᱥᱟᱭ ᱴᱟᱠᱟ ᱢᱮᱱᱟᱜᱼᱟ।",
    "नमस्ते. मैं वाणी हूं. आपका बकाया है. धन्यवाद.",
    "Hi. Ok. Yes. No. Sure. Right. Done. Fine.",
])
def test_no_chunk_is_below_the_floor(text):
    """Every emitted chunk must clear min_chunk_seconds, not just the last one."""
    chunks = split_for_streaming(text, min_chunk_seconds=1.0)
    if len(chunks) < 2:
        return
    for chunk in chunks:
        assert estimate_duration(chunk) >= 0.9, \
            f"chunk below the floor would risk silent output: {chunk!r} ({chunks})"


def test_many_tiny_sentences_are_coalesced():
    text = "Hi. Ok. Yes. No. Sure. Right. Done. Fine. Good. Great."
    chunks = split_for_streaming(text, min_chunk_seconds=1.0)
    assert len(chunks) < 10, chunks
    assert _rejoin(chunks) == re.sub(r"[\s,]+", "", text)
