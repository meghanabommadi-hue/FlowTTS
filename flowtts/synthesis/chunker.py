"""Pipeline position: SMART CHUNKING — normalized text → streaming chunks (pure stdlib).

Role in pipeline:
  OmniVoice is non-autoregressive: one ``generate()`` produces a whole utterance
  and cannot emit audio token-by-token. The only way to stream it is to cut the
  text up, generate each piece, and send each piece's PCM as it is ready. Where
  those cuts land decides whether the result sounds like one person speaking or
  like several clips edited together.

      normalize_for_tts(text) → split_for_streaming(text) → [Chunk, Chunk, …]
                              → engine.synthesize(c.text)  (dispatched together)
                              → stitch.StreamStitcher      → PCM frames

**Sentences are the unit.** Each chunk is generated independently, conditioned
only on the voice prompt — no prosodic state carries across a boundary. A cut
inside a sentence therefore produces two fragments that were each given
sentence-shaped intonation, and no amount of crossfading hides that. A cut
*between* sentences lands where a speaker would pause anyway.

Chunks are built by packing whole sentences up to ``target_chars`` (200),
allowing ``tolerance_chars`` (±50) of slack so a sentence that would just
overshoot is kept whole rather than split.

Inside that window the split point is chosen by a strict priority:

    1. a sentence terminator  . ? ! । ॥ 。 ？ ！
    2. failing that, a clause mark  , ; : —
    3. failing that, a bare word gap

The ordering matters more than the sizes. A comma is a breath *inside* a
thought, so cutting there hands the two halves independently-generated
intonation that does not join up — the same artifact as any other mid-sentence
cut. It is therefore a last resort, reached only when 250 characters have gone
by with no sentence ending in them. Within whichever kind wins, the *last*
candidate is taken, so as many whole sentences are packed together as fit and
the number of seams stays as low as possible.

Every chunk records **why** the split after it happened — sentence end, clause
mark, or word gap — because the stitcher needs to know. A sentence boundary
wants a real pause; a word-gap cut wants a crossfade and no gap at all. Without
that distinction the stitcher either runs sentences together or drops silence
into the middle of a phrase, and both are audible.

Splits never land inside an OmniVoice control tag (``[laughter]``,
``[B EY1 S]``), a decimal or time, an abbreviation like "Dr.", or a numeral the
normalizer bound together with a non-breaking space.

Pure stdlib — no torch, no GPU. See flowtts/test/test_chunker.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Boundary kinds — what caused the split AFTER a chunk
# ---------------------------------------------------------------------------
SENTENCE = "sentence"   # . ? ! । ॥ — a speaker would pause here
CLAUSE = "clause"       # , ; : —    — a shorter breath
WORD = "word"           # no punctuation available; a cut we were forced into
END = "end"             # the last chunk; nothing follows

_RANK = {SENTENCE: 0, CLAUSE: 1, WORD: 2}
_KIND = {0: SENTENCE, 1: CLAUSE, 2: WORD}


@dataclass
class Chunk:
    """One unit of text to synthesize, and how it joins to the next."""

    text: str
    boundary: str = END   # SENTENCE / CLAUSE / WORD / END
    index: int = 0

    def __str__(self) -> str:      # a Chunk can stand in for its text
        return self.text

    def __len__(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Speaking-rate model (used for reporting estimated durations and for batching)
# ---------------------------------------------------------------------------
# Characters of *normalized* text per second of synthesized speech. Indic
# abugidas pack more phonemes per character than Latin script, so equal
# character counts are not equal amounts of audio.
_CHARS_PER_SECOND = {
    "deva": 12.0, "beng": 12.0, "guru": 12.0, "gujr": 12.0, "orya": 12.0,
    "taml": 13.0, "telu": 13.0, "knda": 13.0, "mlym": 13.0,
    "arab": 12.0, "olck": 12.0, "mtei": 12.0,
    "latn": 15.0, "cjk": 5.0, "kana": 7.0,
}
_DEFAULT_CHARS_PER_SECOND = 14.0

# Sentence terminators across the scripts this server serves.
_TERMINATORS = ".?!।॥。？！…"
_CLAUSE_CHARS = ",;:—–、，；："

# A boundary is the punctuation, any closing quote/bracket, and the space after
# it. NBSP is excluded from the whitespace class: the normalizer uses it to bind
# the words of one numeral, and a split there would read as two numbers.
_SENTENCE_BOUNDARY = re.compile(rf"[{re.escape(_TERMINATORS)}]['\"”’)\]]*[^\S ]+")
_CLAUSE_BOUNDARY = re.compile(rf"[{re.escape(_CLAUSE_CHARS)}]['\"”’)\]]*[^\S ]+")
_WORD_BOUNDARY = re.compile(r"[^\S ]+")

# Spans a split must never land inside.
_ATOMIC = re.compile(
    r"\[[^\[\]]*\]"                              # OmniVoice tag: [laughter], [B EY1 S]
    r"|\d+(?:[.,:]\d+)+"                          # 3.14, 12:30, 1,250
    r"|(?:[A-Za-z]\.){2,}"                        # i.e., e.g., U.S.A.
    r"|(?:Dr|Mr|Mrs|Ms|Prof|St|Rs|No|vs|etc)\."   # common abbreviations
)

# A period closing an abbreviation is not a sentence end.
_ABBREV_TAIL = re.compile(
    r"(?:^|[\s(])(?:Dr|Mr|Mrs|Ms|Prof|St|Rs|No|vs|etc|[A-Za-z])\.$", re.IGNORECASE
)

# An abbreviation together with the space after it. Suppressing the split at the
# end of this span stops a chunk ending on "Dr." and leaving "Sharma" to open
# the next one.
_ABBREV_SPAN = re.compile(
    r"(?:Dr|Mr|Mrs|Ms|Prof|St|Rs|No|vs|etc)\.\s+", re.IGNORECASE
)


def dominant_chars_per_second(text: str) -> float:
    """Speaking rate for *text*, from whichever script most of it is in."""
    from flowtts.text.script_detect import dominant_script

    return _CHARS_PER_SECOND.get(dominant_script(text) or "", _DEFAULT_CHARS_PER_SECOND)


def estimate_duration(text: str, chars_per_second: float | None = None) -> float:
    """Estimated seconds of speech for *text*."""
    if not text:
        return 0.0
    return len(text) / (chars_per_second or dominant_chars_per_second(text))


# ---------------------------------------------------------------------------
# Split points
# ---------------------------------------------------------------------------
def _split_points(text: str, split_on_clause: bool = True) -> list[tuple[int, int]]:
    """Legal split offsets as (offset, rank), sorted by offset.

    ``offset`` is where the next chunk starts, so ``text[:offset]`` keeps its own
    punctuation. Offsets inside an atomic span are dropped.

    Clause marks are collected but rank below sentence ends, so they are only
    ever chosen when no sentence ends inside the budget. ``split_on_clause=False``
    drops them entirely, leaving a word gap as the fallback instead.
    """
    atomic = [(m.start(), m.end()) for m in _ATOMIC.finditer(text)]
    # For an abbreviation the END of the span is forbidden too: that offset is
    # the gap between "Dr." and the name it belongs to.
    glued = [(m.start(), m.end()) for m in _ABBREV_SPAN.finditer(text)]

    def _inside(i: int) -> bool:
        return (any(s < i < e for s, e in atomic)
                or any(s < i <= e for s, e in glued))

    points: dict[int, int] = {}
    for m in _SENTENCE_BOUNDARY.finditer(text):
        if _ABBREV_TAIL.search(text[:m.end()].rstrip()):
            continue
        if not _inside(m.end()):
            points[m.end()] = _RANK[SENTENCE]
    if split_on_clause:
        for m in _CLAUSE_BOUNDARY.finditer(text):
            if not _inside(m.end()):
                points.setdefault(m.end(), _RANK[CLAUSE])
    for m in _WORD_BOUNDARY.finditer(text):
        if not _inside(m.end()):
            points.setdefault(m.end(), _RANK[WORD])

    return sorted(points.items())


def split_for_streaming(
    text: str,
    *,
    target_chars: int = 200,
    tolerance_chars: int = 50,
    first_chunk_chars: int | None = None,
    split_on_clause: bool = True,
    max_chunks: int = 128,
) -> list[Chunk]:
    """Split *text* into chunks of about ``target_chars``, aligned to punctuation.

    Within ``target_chars + tolerance_chars`` (250 by default) the split goes to
    the last sentence end; only if 250 characters pass with no sentence ending
    does it fall back to the last comma, and only failing that to a word gap.
    So ordinary prose comes out as whole sentences packed together, and a
    sentence shorter than the budget is never cut in half.

    ``split_on_clause=False`` removes the comma fallback entirely, leaving a
    word gap as the last resort instead.

    ``first_chunk_chars`` shrinks the first chunk only, trading one extra
    boundary for a lower time-to-first-byte. ``None`` gives uniform chunks, which
    is what sounds best: every boundary is a place the audio can betray the seam.

    Returns ``[]`` for empty input, otherwise at least one Chunk. The last
    chunk's boundary is always ``END``.
    """
    text = (text or "").strip()
    if not text:
        return []

    points = _split_points(text, split_on_clause=split_on_clause)
    chunks: list[Chunk] = []
    start = 0
    n = len(text)

    while start < n and len(chunks) < max_chunks - 1:
        budget = (first_chunk_chars if (first_chunk_chars and not chunks)
                  else target_chars)
        limit = start + budget + tolerance_chars
        if limit >= n:
            break   # everything left fits in one final chunk

        window = [(off, rank) for off, rank in points if start < off <= limit]
        if window:
            # Best KIND first, then the LAST candidate of that kind. Sentence
            # ends outrank commas, so a comma is only ever used once 250
            # characters have gone by without one; taking the last packs as many
            # whole sentences together as fit.
            best_rank = min(rank for _, rank in window)
            split_at = max(off for off, rank in window if rank == best_rank)
            rank = best_rank
        else:
            # One unbreakable run longer than the whole window — no punctuation
            # and no spaces. Overshoot to the next legal point rather than
            # cutting inside a word.
            after = [(off, rank) for off, rank in points if off > limit]
            if not after:
                break
            split_at, rank = after[0]

        piece = text[start:split_at].strip()
        if piece:
            chunks.append(Chunk(text=piece, boundary=_KIND[rank], index=len(chunks)))
        start = split_at

    tail = text[start:].strip()
    if tail:
        chunks.append(Chunk(text=tail, boundary=END, index=len(chunks)))
    if chunks:
        chunks[-1].boundary = END
    return chunks


def chunk_text(text: str, max_chunk_len: int = 200, min_chunk_len: int = 50) -> list[str]:
    """Character-budget chunking returning plain strings.

    Kept for parity with the upstream ``indic_tts_normalizer.chunker`` API and
    used by ``flowtts.text.preprocess_text``.
    """
    text = (text or "").strip()
    if not text:
        return []
    return [c.text for c in split_for_streaming(
        text, target_chars=max_chunk_len, tolerance_chars=max(0, min_chunk_len // 2)
    )]
