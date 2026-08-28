#!/usr/bin/env python3
"""Cut a long recording into TTS-sized chunks that carry GROUND-TRUTH text.

The corpus ships human transcripts for whole recordings, some of them minutes
long. Training drops anything over 30s, which is why Hausa capped at ~60h. This
module turns those recordings into usable 0.8-30s chunks.

Whisper (NaijaVox) is used ONLY to locate words in time. The text written out is
always the human transcript, projected onto those timings by aligning the ASR
hypothesis to the ground truth. ASR errors therefore shift boundaries slightly;
they never corrupt the training text.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def norm(w: str) -> str:
    """Fold a token for matching only - never for output."""
    w = unicodedata.normalize("NFKC", w).lower()
    return _PUNCT.sub("", w).strip()


@dataclass
class Chunk:
    start: float
    end: float
    text: str
    n_words: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def project_gt_onto_words(asr_words: List[dict], gt_text: str) -> List[dict]:
    """Give every ground-truth word a (start,end) by aligning it to the ASR.

    Returns [{w, start, end, matched}] over GT words, in GT order. Unmatched GT
    words are interpolated between their nearest matched neighbours, so a run of
    ASR errors degrades timing locally instead of dropping text.
    """
    gt_words = gt_text.split()
    if not gt_words:
        return []
    if not asr_words:
        return []

    a_norm = [norm(w["w"]) for w in asr_words]
    g_norm = [norm(w) for w in gt_words]

    out: List[dict] = [{"w": w, "start": None, "end": None, "matched": False}
                       for w in gt_words]
    sm = difflib.SequenceMatcher(a=g_norm, b=a_norm, autojunk=False)
    for gi, ai, size in sm.get_matching_blocks():
        for k in range(size):
            g, a = gi + k, ai + k
            if g < len(out) and a < len(asr_words):
                out[g]["start"] = float(asr_words[a]["start"])
                out[g]["end"] = float(asr_words[a]["end"])
                out[g]["matched"] = True

    # interpolate the gaps
    anchors = [i for i, o in enumerate(out) if o["matched"]]
    if not anchors:
        return []
    first, last = anchors[0], anchors[-1]
    for i in range(first):                      # before the first anchor
        out[i]["start"] = out[first]["start"]
        out[i]["end"] = out[first]["start"]
    for i in range(last + 1, len(out)):         # after the last
        out[i]["start"] = out[last]["end"]
        out[i]["end"] = out[last]["end"]
    for a, b in zip(anchors, anchors[1:]):
        if b - a <= 1:
            continue
        t0, t1 = out[a]["end"], out[b]["start"]
        span = max(t1 - t0, 0.0)
        n = b - a
        for j in range(1, n):
            out[a + j]["start"] = t0 + span * (j - 1) / n
            out[a + j]["end"] = t0 + span * j / n
    return out


def chunk_words(words: List[dict], min_sec: float = 0.8, max_sec: float = 30.0,
                pause_sec: float = 0.35, target_sec: float = 12.0) -> List[Chunk]:
    """Group timed words into chunks, preferring to break at pauses.

    A chunk is closed when the next word would push it past max_sec, or when it
    is already past target_sec and a real pause follows. Chunks shorter than
    min_sec are dropped rather than padded - a 0.3s fragment is not a training
    example.
    """
    chunks: List[Chunk] = []
    cur: List[dict] = []

    def flush():
        if not cur:
            return
        start = cur[0]["start"]
        end = cur[-1]["end"]
        text = " ".join(w["w"] for w in cur).strip()
        if end - start >= min_sec and text:
            chunks.append(Chunk(start, end, text, len(cur)))

    for i, w in enumerate(words):
        if w["start"] is None:
            continue
        if not cur:
            cur = [w]
            continue
        span = w["end"] - cur[0]["start"]
        gap = w["start"] - cur[-1]["end"]
        if span > max_sec:
            flush(); cur = [w]; continue
        cur.append(w)
        cur_span = cur[-1]["end"] - cur[0]["start"]
        nxt = words[i + 1] if i + 1 < len(words) else None
        nxt_gap = (nxt["start"] - w["end"]) if nxt and nxt["start"] is not None else 999
        if cur_span >= target_sec and nxt_gap >= pause_sec:
            flush(); cur = []
    flush()
    return chunks


def chunk_recording(asr_words: List[dict], gt_text: str, total_duration: float,
                    min_sec: float = 0.8, max_sec: float = 30.0,
                    pause_sec: float = 0.35, target_sec: float = 12.0,
                    min_coverage: float = 0.5) -> Optional[List[Chunk]]:
    """Full path: project GT onto ASR timings, then cut.

    Returns None when the alignment is too poor to trust - better to skip a
    recording than to emit chunks whose audio and text disagree.
    """
    timed = project_gt_onto_words(asr_words, gt_text)
    if not timed:
        return None
    matched = sum(1 for t in timed if t["matched"])
    if matched / max(len(timed), 1) < min_coverage:
        return None
    chunks = chunk_words(timed, min_sec, max_sec, pause_sec, target_sec)
    chunks = [c for c in chunks if 0 <= c.start < c.end <= total_duration + 0.5]
    return chunks or None
