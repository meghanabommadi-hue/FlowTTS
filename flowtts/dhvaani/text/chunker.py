"""Pipeline position: SMART CHUNKING — one utterance -> a schedule of spans.

Role in pipeline:
  Sits between normalisation and tokenisation. Decides how the text is carved up
  into the independent flow trajectories the scheduler will render.

Why chunking is not optional here
---------------------------------
DhVaani is non-autoregressive: `model.sample()` renders an ENTIRE span in one
shot, after `num_step` Euler iterations over every frame of it. There is no
partial output to stream. So for a 6-second sentence, time-to-first-audio equals
time-to-ALL-audio unless the text is split.

Splitting turns that into a pipeline: span 1 is rendered and sent while span 2
is still in the ODE, and so on. Time-to-first-byte becomes a function of the
FIRST span's length only.

Why the spans are not all the same size
---------------------------------------
Every span is conditioned on the voice prompt, and the prompt's frames are part
of the flow decoder's sequence. A 2-second prompt is 187 frames that get
recomputed for every span. So a span costs:

    (prompt_frames + span_frames) x num_step x (2 if CFG)

Short spans are latency-optimal but throughput-terrible: at a 1-second span with
a 2-second prompt, 65% of the GPU work is spent re-rendering the prompt. Long
spans are the reverse.

The schedule below resolves that by ramping: a short first span to get audio
moving, then progressively longer spans once the client's buffer is filling
faster than it drains (which it is, at RTF well below 1). Concretely, with the
default 1.2s / 2.5s / 4.5s schedule and a 2s prompt, prompt overhead falls from
62% on the first span to 31% by the third.

Boundary selection
------------------
Spans are rendered independently, so a boundary mid-phrase produces an audible
seam no crossfade can hide. Breaks are therefore taken at sentence terminators
first, then clause punctuation, then whitespace, and only as a last resort
mid-token. Each span also gets terminal punctuation appended, matching upstream
`add_punctuation()`: ZipVoice was trained on punctuated text and a span without
it ends abruptly.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

import structlog

from flowtts.dhvaani.config import dhv_settings
from flowtts.dhvaani.types import VoicePrompt

logger = structlog.get_logger(__name__)

# Sentence terminators across every script DhVaani supports:
#   ASCII . ! ?
#   U+0964 DEVANAGARI DANDA, U+0965 DOUBLE DANDA (Devanagari, Bengali, Odia,
#          Gurmukhi, Gujarati and others reuse these)
#   U+061F ARABIC QUESTION MARK, U+06D4 ARABIC FULL STOP (Urdu, Sindhi, Kashmiri)
#   U+3002 IDEOGRAPHIC FULL STOP, U+FF01/U+FF1F fullwidth (occasionally seen in
#          copy-pasted text)
_TERMINATORS = ".!?" + "।॥؟۔。！？"
# Clause-level breaks: ASCII , ; : plus U+060C ARABIC COMMA and U+061B ARABIC
# SEMICOLON, and the Devanagari abbreviation sign is NOT a break.
_CLAUSES = ",;:" + "،؛，"

_SENT_RE = re.compile("(?<=[" + re.escape(_TERMINATORS) + r"])\s+")
_CLAUSE_RE = re.compile("(?<=[" + re.escape(_CLAUSES) + r"])\s+")


@dataclass
class Span:
    """One independently rendered piece of an utterance."""

    text: str
    index: int
    is_final: bool
    est_seconds: float


class SmartChunker:
    """Splits normalised text into a latency-aware span schedule."""

    def __init__(self, settings=None):
        self._s = settings or dhv_settings

    # -- budgets -------------------------------------------------------------
    def _target_seconds(self, index: int) -> float:
        c = self._s.chunk
        if index == 0:
            return c.first_chunk_seconds
        if index == 1:
            return c.second_chunk_seconds
        return c.steady_chunk_seconds

    def _budget_chars(self, index: int, voice: VoicePrompt, speed: float) -> int:
        target = min(self._target_seconds(index), self._s.chunk.max_span_seconds)
        return max(1, voice.chars_for_seconds(target, speed))

    def _est_seconds(self, text: str, voice: VoicePrompt, speed: float) -> float:
        from flowtts.dhvaani.config import FRAME_RATE_HZ

        if voice.frames_per_token <= 0:
            return len(text) / 15.0
        frames = len(text) * voice.frames_per_token / max(speed, 1e-6)
        return frames / FRAME_RATE_HZ

    # -- splitting -----------------------------------------------------------
    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [p for p in _SENT_RE.split(text) if p.strip()]

    @staticmethod
    def _split_clauses(text: str) -> list[str]:
        return [p for p in _CLAUSE_RE.split(text) if p.strip()]

    @staticmethod
    def _hard_wrap(text: str, limit: int) -> list[str]:
        """Last resort: break on whitespace, then mid-token if a single token is
        longer than the limit (a URL, or an unsegmented script)."""
        out: list[str] = []
        cur = ""
        for word in text.split(" "):
            while len(word) > limit:
                if cur:
                    out.append(cur)
                    cur = ""
                out.append(word[:limit])
                word = word[limit:]
            cand = (cur + " " + word).strip() if cur else word
            if len(cand) > limit and cur:
                out.append(cur)
                cur = word
            else:
                cur = cand
        if cur:
            out.append(cur)
        return out

    def _atoms(self, text: str, max_len: int) -> list[str]:
        """Break text into the smallest pieces we are willing to join at."""
        atoms: list[str] = []
        for sent in self._split_sentences(text):
            if len(sent) <= max_len:
                atoms.append(sent)
                continue
            for clause in self._split_clauses(sent):
                if len(clause) <= max_len:
                    atoms.append(clause)
                else:
                    atoms.extend(self._hard_wrap(clause, max_len))
        return atoms

    def _max_span_seconds(self, voice: VoicePrompt) -> float:
        """Longest span we may emit, clamped by the largest arena bucket.

        The scheduler cannot render a span whose (prompt + generated) frames
        exceed the biggest bucket, and it fails such a span loudly rather than
        truncating it. Deriving the cap here from the same setting keeps the two
        from drifting apart when someone tunes one and not the other.
        """
        from flowtts.dhvaani.config import FRAME_RATE_HZ

        budget_frames = self._s.buckets.buckets[-1] - voice.mel_frames
        # Leave a little headroom: predict_feature_lens rounds up, and a span
        # that lands exactly on the boundary would be rejected.
        usable = max(0.5, (budget_frames * 0.95) / FRAME_RATE_HZ)
        return min(self._s.chunk.max_span_seconds, usable)

    # -- public --------------------------------------------------------------
    def split(self, text: str, voice: VoicePrompt, speed: float = 1.0) -> list[Span]:
        text = (text or "").strip()
        if not text:
            return []

        c = self._s.chunk
        total_est = self._est_seconds(text, voice, speed)

        # One span is always better than two when the whole thing is short: no
        # seam, no repeated prompt cost, and TTFB is already fine.
        if not c.enabled or total_est <= c.single_span_max_seconds:
            return [Span(add_punctuation(text), 0, True, total_est)]

        # The largest span we will ever emit bounds the atom size.
        max_span_s = self._max_span_seconds(voice)
        hard_cap = max(1, voice.chars_for_seconds(max_span_s, speed))
        atoms = self._atoms(text, hard_cap)
        if not atoms:
            return [Span(add_punctuation(text), 0, True, total_est)]

        min_chars = max(1, voice.chars_for_seconds(c.min_chunk_seconds, speed))

        # Pack atoms into spans, growing the budget as the schedule ramps.
        #
        # Oversized atoms are wrapped against the budget IN FORCE AT THE TIME
        # they are reached, not a fixed one. That distinction is what keeps
        # time-to-first-byte low for text with no sentence punctuation: such
        # text is a single huge atom, and wrapping it to the steady budget
        # would make the very first span a 4-second render.
        pending = deque(atoms)
        spans: list[str] = []
        cur = ""
        budget = self._budget_chars(0, voice, speed)

        while pending:
            atom = pending.popleft()
            if not cur and len(atom) > budget:
                pieces = self._hard_wrap(atom, budget)
                if len(pieces) > 1:
                    pending.extendleft(reversed(pieces))
                    continue
            cand = (cur + " " + atom).strip() if cur else atom
            if not cur:
                cur = cand
                continue
            if len(cand) <= budget:
                cur = cand
                continue
            # Emitting `cur` now. If it is below the floor, keep growing instead:
            # a 0.3 s span sounds clipped and wastes a whole prompt render.
            if len(cur) < min_chars and len(cand) <= hard_cap:
                cur = cand
                continue
            spans.append(cur)
            cur = ""
            budget = self._budget_chars(len(spans), voice, speed)
            pending.appendleft(atom)
        if cur:
            spans.append(cur)

        # A trailing runt is merged backwards when the result still fits, for the
        # same reason.
        if len(spans) > 1 and len(spans[-1]) < min_chars:
            merged = spans[-2] + " " + spans[-1]
            if len(merged) <= hard_cap:
                spans = spans[:-2] + [merged]

        out: list[Span] = []
        for i, s in enumerate(spans):
            out.append(
                Span(
                    text=add_punctuation(s),
                    index=i,
                    is_final=(i == len(spans) - 1),
                    est_seconds=self._est_seconds(s, voice, speed),
                )
            )
        return out


def add_punctuation(text: str) -> str:
    """Append a full stop when the text does not already end in punctuation.

    Mirrors `zipvoice.utils.infer.add_punctuation`. ZipVoice was trained on
    punctuated text; a span without a terminator gets a truncated-sounding tail.
    """
    text = text.strip()
    if not text:
        return text
    if text[-1] in _TERMINATORS or text[-1] in _CLAUSES:
        return text
    return text + "."
