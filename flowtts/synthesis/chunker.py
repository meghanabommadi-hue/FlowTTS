"""Pipeline position: SMART CHUNKING — normalized text → streaming chunks (pure stdlib).

Role in pipeline:
  OmniVoice is non-autoregressive: one ``generate()`` produces a whole utterance
  and cannot emit audio token-by-token. The only way to stream it is to cut the
  text up, generate each piece, and send each piece's PCM the moment it is ready.
  Where those cuts land decides both the time-to-first-byte and whether the
  result sounds like one sentence or four.

      normalize_for_tts(text) → split_for_streaming(text) → [c0, c1, c2, …]
                              → engine.synthesize(c_i)  (all dispatched at once)
                              → stitch.StreamStitcher   → PCM frames

Three ideas do the work:

**Progressive sizing, with a floor.** Chunk 0 targets ~1.2 s of audio, chunk 1
~3 s, and later chunks the full cap. TTFB is set by chunk 0 alone, so it should
be as small as prosody allows; once the client is playing, larger chunks give
better prosody and fewer boundaries. As long as the engine's real-time factor
stays under 1 the stream never starves, because chunk *i+1* generates while
chunk *i* plays.

The floor (``min_chunk_seconds``) is not cosmetic: below roughly 1 s of target
audio OmniVoice starts returning outright silence, reproducibly, in Hindi and
Santali alike — see :func:`_merge_short`. Chunks under the floor are merged
rather than sent.

**Duration, not characters.** A 60-character Devanagari sentence and a
60-character English one are not the same amount of speech, and a chunk budget
in characters silently means a different TTFB per language. Chunks are budgeted
in estimated seconds of audio, from a per-script speaking rate.

**Boundary quality over budget fill.** Within a chunk's budget, the best
*kind* of boundary always wins: a sentence end, else a clause mark, else a bare
word gap. Ending a chunk early at a full stop costs a little throughput; ending
it late in the middle of "two thousand and | twenty-six" is audible in every
single stream. Splits can never land inside an OmniVoice control tag
(``[laughter]``, ``[B EY1 S]``), a decimal or a time, or an abbreviation like
"Dr.". Later chunks get a generous 10 s budget precisely so that most sentences
never need splitting at all — only the first chunk trades prosody for latency.

Pure stdlib — no torch, no GPU. See flowtts/test/test_chunker.py.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Speaking-rate model
# ---------------------------------------------------------------------------
# Characters of *normalized* text per second of synthesized speech. Indic
# abugidas pack more phonemes per character than Latin script, so equal
# character counts are not equal amounts of audio. These are deliberately
# conservative (they over-estimate duration slightly), which makes the first
# chunk a little short rather than a little long — the safe direction for TTFB.
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

_SENTENCE_BOUNDARY = re.compile(rf"[{re.escape(_TERMINATORS)}]['\"”’)\]]*[^\S\u00a0]+")
_CLAUSE_BOUNDARY = re.compile(rf"[{re.escape(_CLAUSE_CHARS)}]['\"”’)\]]*[^\S\u00a0]+")
# Any whitespace run EXCEPT one containing a non-breaking space: the
# normalizer binds the words of a single numeral with NBSP so a chunk
# boundary can never land between "दो" and "हज़ार".
_WORD_BOUNDARY = re.compile(r"[^\S\u00a0]+")

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

# An abbreviation together with the space after it. Suppressing the split point
# at the end of this span is what stops a chunk ending on "Dr." and leaving
# "Sharma" to open the next one — the sentence-boundary rule already ignores the
# period, but the plain word gap after it is still a legal split otherwise.
_ABBREV_SPAN = re.compile(
    r"(?:Dr|Mr|Mrs|Ms|Prof|St|Rs|No|vs|etc)\.\s+", re.IGNORECASE
)

# Boundary quality, best first. A lower rank always wins over budget fill.
_RANK_SENTENCE, _RANK_CLAUSE, _RANK_WORD = 0, 1, 2


def dominant_chars_per_second(text: str) -> float:
    """Speaking rate for *text*, from whichever script most of it is in."""
    from flowtts.text.script_detect import dominant_script

    return _CHARS_PER_SECOND.get(dominant_script(text) or "", _DEFAULT_CHARS_PER_SECOND)


def estimate_duration(text: str, chars_per_second: float | None = None) -> float:
    """Estimated seconds of speech for *text*."""
    if not text:
        return 0.0
    return len(text) / (chars_per_second or dominant_chars_per_second(text))


def _split_points(text: str) -> list[tuple[int, int]]:
    """Legal split offsets in *text* as (offset, rank), sorted by offset.

    ``offset`` is the index at which the next chunk starts, so ``text[:offset]``
    keeps its own punctuation. Offsets inside an atomic span are dropped.
    """
    atomic = [(m.start(), m.end()) for m in _ATOMIC.finditer(text)]
    # For an abbreviation the END of the span is forbidden too, not just its
    # interior: that offset is the gap between "Dr." and the name it belongs to.
    glued = [(m.start(), m.end()) for m in _ABBREV_SPAN.finditer(text)]

    def _inside(i: int) -> bool:
        return (any(s < i < e for s, e in atomic)
                or any(s < i <= e for s, e in glued))

    points: dict[int, int] = {}

    for m in _SENTENCE_BOUNDARY.finditer(text):
        # "Dr. Sharma" / "e.g. this": the period belongs to the abbreviation.
        if _ABBREV_TAIL.search(text[:m.end()].rstrip()):
            continue
        if not _inside(m.end()):
            points[m.end()] = _RANK_SENTENCE
    for m in _CLAUSE_BOUNDARY.finditer(text):
        if not _inside(m.end()):
            points.setdefault(m.end(), _RANK_CLAUSE)
    for m in _WORD_BOUNDARY.finditer(text):
        if not _inside(m.end()):
            points.setdefault(m.end(), _RANK_WORD)

    return sorted(points.items())


def _ensure_terminated(chunk: str) -> str:
    """Give a chunk end punctuation if it has none.

    OmniVoice reads an unterminated chunk as a trailing-off fragment, so a
    stream cut at a bare word gap gets an audible pitch drift at every boundary.
    A comma is appended rather than a period: a mid-utterance chunk *is* a
    continuation, and the comma contour is the one that stitches cleanly.
    """
    if not chunk:
        return chunk
    return chunk if chunk[-1] in _TERMINATORS or chunk[-1] in _CLAUSE_CHARS else chunk + ","


def _merge_short(chunks: list[str], min_chars: int) -> list[str]:
    """Fold every chunk below ``min_chars`` into a neighbour.

    Not cosmetic. OmniVoice becomes unreliable on very short targets: a chunk of
    ~30 audio frames comes back as pure silence a good fraction of the time,
    measured across Hindi and Santali alike, while the same sentence at ~125
    frames is stable every run. The model simply has too little to condition on.
    A sliver is also an audible click at the seam even when it does speak.

    Merges forward where possible so the merged text still reads in order, and
    backward for a trailing sliver.
    """
    if len(chunks) < 2:
        return chunks

    merged: list[str] = []
    carry = ""
    for chunk in chunks:
        candidate = f"{carry} {chunk}".strip() if carry else chunk
        if len(candidate) < min_chars:
            carry = candidate          # still too short — keep accumulating
            continue
        merged.append(candidate)
        carry = ""
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}".strip()
        else:
            merged.append(carry)
    return merged


def split_for_streaming(
    text: str,
    *,
    first_chunk_seconds: float = 1.2,
    first_chunk_max_seconds: float = 2.6,
    second_chunk_seconds: float = 3.0,
    chunk_seconds: float = 10.0,
    min_chunk_seconds: float = 1.0,
    max_chunks: int = 128,
    terminate_chunks: bool = True,
) -> list[str]:
    """Split *text* into progressively larger streaming chunks.

    Budgets are estimated seconds of audio (see :func:`estimate_duration`). The
    first chunk is small so time-to-first-byte is small; later chunks grow to
    ``chunk_seconds`` for prosody. ``first_chunk_max_seconds`` is how far the
    first chunk may run over its budget to reach a sentence or clause boundary
    instead of stopping at a bare word gap. Returns ``[]`` for empty input and
    otherwise at least one chunk.
    """
    text = (text or "").strip()
    if not text:
        return []

    rate = dominant_chars_per_second(text)
    points = _split_points(text)
    budgets = [max(1, int(first_chunk_seconds * rate)),
               max(1, int(second_chunk_seconds * rate))]
    full_budget = max(1, int(chunk_seconds * rate))
    min_chars = max(1, int(min_chunk_seconds * rate))

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n and len(chunks) < max_chunks - 1:
        budget = budgets[len(chunks)] if len(chunks) < len(budgets) else full_budget
        target = start + budget
        if target >= n:
            break

        # Boundary quality beats budget fill, always. Cutting a chunk short at a
        # full stop costs a little throughput; cutting it long in the middle of
        # "two thousand and | twenty-six" is audible in every single stream.
        # Candidates whose chunk clears the floor once stripped. Measuring the
        # stripped length matters: an offset one space past the floor yields a
        # piece one character under it, which then gets merged into the next
        # chunk wholesale and undoes the progressive sizing.
        def _clears_floor(off: int) -> bool:
            return len(text[start:off].strip()) >= min_chars

        candidates = [(off, rank) for off, rank in points
                      if start < off <= target and _clears_floor(off)]

        # The first chunk is allowed to run over budget to reach a real
        # boundary. "[laughter] You," costs less latency than the sentence
        # "[laughter] You really got me." but sounds broken, and the extra
        # ~0.7 s of audio is generated in roughly the same wall-clock time.
        if len(chunks) == 0 and not any(r <= _RANK_CLAUSE for _, r in candidates):
            stretch = start + max(1, int(first_chunk_max_seconds * rate))
            natural = [(off, rank) for off, rank in points
                       if start < off <= stretch and rank <= _RANK_CLAUSE
                       and _clears_floor(off)]
            if natural:
                candidates = natural

        if not candidates:
            # Nothing inside the budget clears the floor — take the first offset
            # that does, even beyond the budget. A chunk slightly over budget is
            # far better than one the model may return as silence.
            beyond = [(off, rank) for off, rank in points
                      if off > target and _clears_floor(off)]
            candidates = beyond[:1]

        if candidates:
            best_rank = min(rank for _, rank in candidates)
            at_rank = [off for off, rank in candidates if rank == best_rank]
            # The first chunk takes the EARLIEST natural break it can — that is
            # the whole TTFB lever, and "नमस्ते." or "Hello," is a complete
            # prosodic unit. Every later chunk takes the latest, to fill the
            # budget and keep the boundary count down.
            first_natural = len(chunks) == 0 and best_rank <= _RANK_CLAUSE
            split_at = min(at_rank) if first_natural else max(at_rank)
        else:
            # Nothing legal inside the budget (one very long token): overshoot to
            # the next legal point rather than cutting a word in half.
            after = [off for off, _ in points if off > target]
            if not after:
                break
            split_at = after[0]

        piece = text[start:split_at].strip()
        if piece:
            chunks.append(piece)
        start = split_at

    tail = text[start:].strip()
    if tail:
        chunks.append(tail)

    # Safety net for input where no split can clear the floor at all (a page of
    # two-word sentences); the loop above already respects it otherwise.
    chunks = _merge_short(chunks, min_chars)

    if terminate_chunks and len(chunks) > 1:
        chunks = [_ensure_terminated(c) for c in chunks[:-1]] + [chunks[-1]]
    return chunks


def chunk_text(text: str, max_chunk_len: int = 200, min_chunk_len: int = 50) -> list[str]:
    """Character-budget chunking, for callers that think in characters.

    Kept for parity with the upstream ``indic_tts_normalizer.chunker`` API and
    used by ``flowtts.text.preprocess_text``. Streaming synthesis uses
    :func:`split_for_streaming` instead.
    """
    text = (text or "").strip()
    if not text:
        return []
    rate = dominant_chars_per_second(text)
    return split_for_streaming(
        text,
        first_chunk_seconds=max_chunk_len / rate,
        second_chunk_seconds=max_chunk_len / rate,
        chunk_seconds=max_chunk_len / rate,
        min_chunk_seconds=min_chunk_len / rate,
        terminate_chunks=False,
    )
