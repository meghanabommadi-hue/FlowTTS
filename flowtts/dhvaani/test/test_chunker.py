"""Tests for the streaming chunk schedule.

The chunker is what makes streaming possible on a non-autoregressive model, so
the properties that matter are: the first span is short (time-to-first-byte),
later spans grow (prompt amortisation), no span can exceed the largest arena
bucket (the scheduler rejects those), and no input shape produces one giant
span.
"""

from __future__ import annotations

import pytest

from flowtts.dhvaani.config import dhv_settings
from flowtts.dhvaani.text.chunker import SmartChunker, add_punctuation
from flowtts.dhvaani.types import VoicePrompt


def make_voice(prompt_seconds: float = 2.0, n_tokens: int = 60) -> VoicePrompt:
    frames = int(prompt_seconds * 93.75)
    return VoicePrompt(
        voice_id="test", mel=None, mel_frames=frames,
        token_ids=list(range(n_tokens)), prompt_rms=0.1,
        frames_per_token=frames / n_tokens,
    )


HINDI = (
    "नमस्ते, मैं बजाज फाइनेंस से बोल रही हूं। आपकी ईएमआई दो हज़ार पांच सौ रुपये बकाया है। "
    "कृपया आज ही भुगतान करें, अन्यथा विलंब शुल्क लग सकता है। "
    "आप हमारे मोबाइल ऐप से, या यूपीआई, एनईएफटी और आईएमपीएस के माध्यम से भुगतान कर सकते हैं।"
)
NO_PUNCT = (
    "this is a long stretch of text with absolutely no punctuation anywhere in it "
    "at all which forces the chunker to fall back to whitespace wrapping repeatedly"
)


def test_short_text_is_one_span():
    ch = SmartChunker()
    spans = ch.split("नमस्ते, आपकी कैसे मदद करूं?", make_voice())
    assert len(spans) == 1
    assert spans[0].is_final


def test_long_text_ramps_span_length():
    ch = SmartChunker()
    spans = ch.split(HINDI, make_voice())
    assert len(spans) > 1
    # First span short for TTFB, later spans longer to amortise the prompt.
    assert spans[0].est_seconds <= dhv_settings.chunk.first_chunk_seconds * 1.35
    assert max(s.est_seconds for s in spans[1:]) > spans[0].est_seconds


def test_only_last_span_is_final():
    ch = SmartChunker()
    spans = ch.split(HINDI, make_voice())
    assert [s.is_final for s in spans] == [False] * (len(spans) - 1) + [True]
    assert [s.index for s in spans] == list(range(len(spans)))


def test_text_without_punctuation_still_splits():
    """A single unpunctuated paragraph is one 'sentence'; without special
    handling it would sail through as one span and destroy TTFB."""
    ch = SmartChunker()
    spans = ch.split(NO_PUNCT, make_voice())
    assert len(spans) > 1
    assert spans[0].est_seconds <= dhv_settings.chunk.first_chunk_seconds * 1.35


def test_single_enormous_token_is_hard_wrapped():
    ch = SmartChunker()
    spans = ch.split("a" * 400, make_voice())
    assert len(spans) > 1
    assert all(len(s.text) < 400 for s in spans)


@pytest.mark.parametrize("prompt_seconds", [1.0, 2.0, 3.0])
def test_no_span_exceeds_largest_bucket(prompt_seconds):
    """The scheduler fails a span that needs more frames than the biggest
    arena bucket, so the chunker must never emit one."""
    ch = SmartChunker()
    voice = make_voice(prompt_seconds)
    max_bucket = dhv_settings.buckets.buckets[-1]
    for text in (HINDI, NO_PUNCT, "a" * 2000, "आपका भुगतान बकाया है। " * 40):
        for span in ch.split(text, voice):
            frames = int(len(span.text) * voice.frames_per_token) + voice.mel_frames
            assert frames <= max_bucket, (text[:20], len(span.text), frames)


def test_empty_and_whitespace():
    ch = SmartChunker()
    assert ch.split("", make_voice()) == []
    assert ch.split("   \n ", make_voice()) == []


def test_add_punctuation():
    assert add_punctuation("hello") == "hello."
    assert add_punctuation("नमस्ते।") == "नमस्ते।"
    assert add_punctuation("what?") == "what?"
    assert add_punctuation("  spaced  ") == "spaced."
    assert add_punctuation("") == ""


def test_every_span_ends_punctuated():
    """ZipVoice was trained on punctuated text; a span without a terminator
    ends abruptly."""
    ch = SmartChunker()
    from flowtts.dhvaani.text.chunker import _CLAUSES, _TERMINATORS

    for span in ch.split(HINDI, make_voice()):
        assert span.text[-1] in _TERMINATORS + _CLAUSES


def test_faster_speed_packs_more_text_per_span():
    ch = SmartChunker()
    v = make_voice()
    slow = ch.split(HINDI, v, speed=1.0)
    fast = ch.split(HINDI, v, speed=1.5)
    # At higher speed the same seconds hold more characters, so fewer spans.
    assert len(fast) <= len(slow)
