"""Pipeline position: STREAMING TEXT SPLITTER (pure Python — no GPU/torch).

Role in pipeline:
  OmniVoice is non-autoregressive: it cannot emit audio token-by-token. To stream
  we split the input text into chunks, synthesize each chunk (batched with other
  requests), and stream its PCM as soon as it is ready. A deliberately SHORT first
  chunk minimizes time-to-first-byte; later chunks are larger for smoother prosody.

  server.py → split_for_streaming(text, ...) → [chunk0, chunk1, ...]
            → engine.synthesize(chunk_i) per chunk → PCM → WebSocket

This module is intentionally dependency-free so it can be unit-tested on any box
(no CUDA, no torch). See flowtts/test/test_text_chunker.py.
"""

from __future__ import annotations

import re

# Strong sentence terminators: Devanagari danda (।), double danda (॥), and ASCII . ? !
# Kept ATTACHED to the preceding sentence when splitting.
_STRONG_BOUNDARY = re.compile(r"(?<=[।॥\.\?\!])\s+")
# Soft (clause) boundaries used only when a single sentence exceeds the cap.
_SOFT_BOUNDARY = re.compile(r"(?<=[,;:—])\s+")


def _split_keep(text: str, pattern: re.Pattern) -> list[str]:
    """Split on *pattern* boundaries, returning non-empty stripped units."""
    return [u.strip() for u in pattern.split(text) if u and u.strip()]


def _hard_wrap(unit: str, cap: int) -> list[str]:
    """Last resort: split an over-long unit on soft boundaries, then whitespace,
    never exceeding *cap* characters per piece (except an unbreakable token)."""
    if len(unit) <= cap:
        return [unit]

    pieces: list[str] = []
    for sub in _split_keep(unit, _SOFT_BOUNDARY) or [unit]:
        if len(sub) <= cap:
            pieces.append(sub)
            continue
        # Still too long — pack whole words up to the cap.
        cur = ""
        for word in sub.split(" "):
            candidate = f"{cur} {word}".strip()
            if cur and len(candidate) > cap:
                pieces.append(cur)
                cur = word
            else:
                cur = candidate
        if cur:
            pieces.append(cur)
    return pieces


def split_for_streaming(
    text: str,
    *,
    first_chunk_max_chars: int = 60,
    chunk_max_chars: int = 160,
    min_chunk_chars: int = 12,
) -> list[str]:
    """Split *text* into streaming chunks.

    - The first chunk is capped at ``first_chunk_max_chars`` (low TTFB).
    - Remaining chunks are packed greedily up to ``chunk_max_chars``.
    - Chunks shorter than ``min_chunk_chars`` are merged into a neighbour so we
      never synthesize a sliver (except when the whole input is that short).
    - Chunk boundaries prefer sentence terminators; over-long sentences fall back
      to clause/word wrapping.

    Returns a list with at least one element (empty input → ``[]``).
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1. Break into sentence-level units, hard-wrapping any that exceed chunk_max_chars.
    units: list[str] = []
    for sentence in _split_keep(text, _STRONG_BOUNDARY) or [text]:
        units.extend(_hard_wrap(sentence, chunk_max_chars))

    # 2. Greedily pack units into chunks; first chunk uses the smaller cap.
    chunks: list[str] = []
    cur = ""
    cap = first_chunk_max_chars
    for unit in units:
        candidate = f"{cur} {unit}".strip()
        if cur and len(candidate) > cap:
            chunks.append(cur)
            cur = unit
            cap = chunk_max_chars  # subsequent chunks may be larger
        else:
            cur = candidate
    if cur:
        chunks.append(cur)

    # 3. Merge a too-short trailing chunk back into its predecessor.
    if len(chunks) >= 2 and len(chunks[-1]) < min_chunk_chars:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return chunks


def normalize_text(text: str) -> str:
    """Keep only English (ASCII) and Hindi (Devanagari U+0900–U+097F); drop the rest.

    Mirrors the normalization the previous model applied so client behaviour is
    unchanged. Removes emoji, Arabic/Urdu, CJK, etc. that OmniVoice's tokenizer
    would otherwise mishandle for this Hindi/English deployment.
    """
    return re.sub(r"[^\x00-\x7Fऀ-ॿ]", "", text or "").strip()
