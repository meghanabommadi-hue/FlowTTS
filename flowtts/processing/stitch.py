"""Pipeline position: AUDIO STITCHING — per-chunk waveforms → one seamless stream.

Role in pipeline:
  The chunker cuts text so each piece can be synthesized and sent early; this
  module puts the pieces back together so the listener hears one person, not a
  playlist.

      engine.synthesize(chunk_i) → StreamStitcher.push(wav, boundary) → PCM → client

Three problems have to be solved at every seam, and each is separately audible.

**Level.** Each chunk is generated independently, and OmniVoice's output level
wanders between calls: measured on one sentence split three ways, chunk RMS
ranged 0.058–0.086, a 3.4 dB swing, while the same text generated whole held
0.080–0.086. Matching only across the overlap — and clamping that match to a
couple of dB — leaves most of the jump in place, and it reads as the voice
changing character mid-utterance. So every chunk is normalized to one target
level for the whole utterance, measured over *voiced* frames so a chunk that
happens to carry more silence is not pushed louder to compensate.

**Pacing.** A sentence boundary is not the same join as a word-gap cut. Trimming
both sides to 20 ms and crossfading — the same treatment for every seam — runs
consecutive sentences together with no breath, which is exactly the "stitched
back together" quality. Each chunk therefore carries the reason it was cut, and
gets the gap that reason deserves: a real pause after a full stop, a shorter one
after a comma, and no gap at all after a word-gap cut, where a crossfade is
right because the phrase genuinely continues.

**Clicks.** Each chunk decodes with its own small DC offset; butting two
different offsets together is a step discontinuity. Removed per chunk, after
trimming rather than before — subtracting a whole-clip mean lifts the silent
regions off zero, and a lifted floor sits above the trim gate.

Pure NumPy — no torch, no GPU. See flowtts/test/test_stitch.py.
"""

from __future__ import annotations

import numpy as np

from flowtts.synthesis.chunker import CLAUSE, END, SENTENCE, WORD

# Below this RMS a 10 ms window counts as silence for edge trimming. -50 dBFS,
# matching the threshold OmniVoice's own remove_silence() uses.
_SILENCE_RMS = 10 ** (-50 / 20)
_TRIM_WINDOW_MS = 10.0

# How long a pause each kind of boundary deserves, in milliseconds. These are
# ordinary speech-timing values: a full stop is roughly a quarter-second of
# silence, a comma about half that, and a phrase that simply continued gets
# nothing at all.
DEFAULT_GAPS_MS = {
    SENTENCE: 260.0,
    CLAUSE: 130.0,
    WORD: 0.0,
    END: 0.0,
}


def remove_dc(wav: np.ndarray) -> np.ndarray:
    """Subtract the mean so chunks butt together without a step discontinuity."""
    if wav.size == 0:
        return wav
    return wav - wav.mean()


def frame_rms(wav: np.ndarray, sample_rate: int, window_ms: float = _TRIM_WINDOW_MS):
    """Per-window RMS, for gating and level measurement."""
    win = max(1, int(sample_rate * window_ms / 1000))
    n_win = wav.size // win
    if n_win < 1:
        return np.array([np.sqrt((wav.astype(np.float64) ** 2).mean())]) if wav.size \
            else np.zeros(0)
    frames = wav[: n_win * win].reshape(n_win, win)
    return np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))


def voiced_rms(wav: np.ndarray, sample_rate: int) -> float:
    """RMS over the *voiced* part of a clip only.

    A plain whole-clip RMS is diluted by however much silence the clip happens
    to carry, so normalizing on it makes a pause-heavy chunk louder than a dense
    one — which is the opposite of matching them. Frames below a fraction of the
    clip's own loudest frame are treated as silence and excluded.
    """
    if wav.size == 0:
        return 0.0
    rms = frame_rms(wav, sample_rate)
    if rms.size == 0:
        return 0.0
    loudest = rms.max()
    if loudest <= 0:
        return 0.0
    voiced = rms[rms > loudest * 0.1]
    return float(voiced.mean()) if voiced.size else float(loudest)


def trim_silence(
    wav: np.ndarray,
    sample_rate: int,
    *,
    keep_ms: float = 10.0,
    threshold: float = _SILENCE_RMS,
) -> np.ndarray:
    """Trim leading/trailing silence, keeping ``keep_ms`` of it as breathing room.

    A windowed RMS gate rather than a per-sample one: a per-sample threshold
    trips on the zero crossings inside ordinary speech and eats real audio.
    """
    if wav.size == 0:
        return wav

    win = max(1, int(sample_rate * _TRIM_WINDOW_MS / 1000))
    rms = frame_rms(wav, sample_rate)
    if rms.size < 2:
        return wav

    loud = rms > threshold
    if not loud.any():
        # Silent end to end — keep a token amount so the stream has continuity
        # rather than a hard zero-length gap.
        return wav[: max(1, int(sample_rate * keep_ms / 1000))]

    keep = max(0, int(sample_rate * keep_ms / 1000))
    first = max(0, int(np.argmax(loud)) * win - keep)
    last = min(wav.size, (rms.size - int(np.argmax(loud[::-1]))) * win + keep)
    return wav[first:last]


def equal_power_crossfade(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Blend ``tail`` into ``head`` over their common length at constant power.

    Adjacent speech chunks are uncorrelated, so they sum in power rather than
    amplitude; the sin/cos pair holds RMS constant across the overlap where a
    linear ramp dips about 3 dB in the middle.
    """
    n = min(len(tail), len(head))
    if n == 0:
        return np.concatenate([tail, head])
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    return tail[-n:] * np.cos(t) + head[:n] * np.sin(t)


def fade(wav: np.ndarray, n: int, *, direction: str) -> np.ndarray:
    """Apply a raised-cosine fade of ``n`` samples at one end of ``wav``."""
    if n <= 0 or wav.size == 0:
        return wav
    n = min(n, wav.size)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))
    out = wav.copy()
    if direction == "in":
        out[:n] *= ramp
    else:
        out[-n:] *= ramp[::-1]
    return out


class StreamStitcher:
    """Joins successive chunk waveforms into one continuous utterance.

    Usage::

        st = StreamStitcher(sample_rate=24000)
        for chunk, wav in zip(chunks, waves):
            audio = st.push(wav, boundary=chunk.boundary,
                            is_final=(chunk is chunks[-1]))
            if audio.size:
                send(pcm_int16(audio))

    ``push`` returns the audio that is safe to send now. Only a word-gap
    boundary holds anything back (one crossfade overlap); a punctuation boundary
    emits immediately, so most utterances stream with no added latency at all.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        overlap_ms: float = 20.0,
        edge_fade_ms: float = 8.0,
        click_fade_ms: float = 3.0,
        final_fade_ms: float = 12.0,
        trim: bool = True,
        trim_keep_ms: float = 10.0,
        level_match: bool = True,
        max_gain_db: float = 6.0,
        gaps_ms: dict | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.overlap = max(0, int(sample_rate * overlap_ms / 1000))
        self.edge_fade = max(0, int(sample_rate * edge_fade_ms / 1000))
        self.click_fade = max(0, int(sample_rate * click_fade_ms / 1000))
        self.final_fade = max(0, int(sample_rate * final_fade_ms / 1000))
        self.trim = trim
        self.trim_keep_ms = trim_keep_ms
        self.level_match = level_match
        self.max_gain = 10 ** (max_gain_db / 20)
        self.gaps_ms = {**DEFAULT_GAPS_MS, **(gaps_ms or {})}

        self._held = np.zeros(0, dtype=np.float32)   # tail awaiting a crossfade
        self._pending_gap = 0                        # silence owed before the next chunk
        self._target_rms: float | None = None        # the utterance's level
        self._first = True
        self._done = False

    # ------------------------------------------------------------------ helpers
    def _gap_samples(self, boundary: str) -> int:
        return int(self.sample_rate * self.gaps_ms.get(boundary, 0.0) / 1000)

    def _prepare(self, wav: np.ndarray) -> np.ndarray:
        """Trim, de-DC, and bring the chunk to the utterance's level."""
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            return wav

        # Trim before removing DC, not after: subtracting a whole-clip mean
        # lifts the silent regions off zero, and a lifted floor sits above the
        # -50 dBFS gate, so the trim would then find no silence at all.
        if self.trim:
            wav = trim_silence(wav, self.sample_rate, keep_ms=self.trim_keep_ms)
        wav = remove_dc(wav)

        if not self.level_match or wav.size == 0:
            return wav

        level = voiced_rms(wav, self.sample_rate)
        if level < 1e-6:
            return wav
        if self._target_rms is None:
            # The first chunk sets the level for the whole utterance, so the
            # opening is never re-gained and the rest is matched to it.
            self._target_rms = level
            return wav

        gain = float(np.clip(self._target_rms / level, 1 / self.max_gain, self.max_gain))
        if abs(gain - 1.0) < 1e-3:
            return wav
        out = wav * gain
        peak = float(np.abs(out).max())
        if peak > 0.99:              # never let the match clip
            out *= 0.99 / peak
        return out

    # ------------------------------------------------------------------ public
    def push(
        self,
        wav: np.ndarray,
        *,
        boundary: str = END,
        is_final: bool = False,
    ) -> np.ndarray:
        """Add one chunk and return the audio ready to send.

        ``boundary`` is what caused the split *after* this chunk — see
        flowtts.synthesis.chunker. It decides how the next chunk is joined on.
        """
        if self._done:
            return np.zeros(0, dtype=np.float32)

        wav = self._prepare(wav)
        if wav.size == 0:
            return self.flush() if is_final else np.zeros(0, dtype=np.float32)

        parts: list[np.ndarray] = []

        if self._first:
            wav = fade(wav, self.click_fade, direction="in")
            self._first = False
        elif self._held.size:
            # The previous chunk was cut at a word gap: the phrase continues, so
            # blend rather than pause.
            fused = equal_power_crossfade(self._held, wav)
            parts.append(fused)
            wav = wav[len(fused):]
            self._held = np.zeros(0, dtype=np.float32)
        else:
            # The previous chunk ended on punctuation: give the listener the
            # pause a speaker would have taken.
            if self._pending_gap:
                parts.append(np.zeros(self._pending_gap, dtype=np.float32))
            wav = fade(wav, self.click_fade, direction="in")
        self._pending_gap = 0

        if is_final or boundary == END:
            self._done = True
            parts.append(fade(wav, self.final_fade, direction="out"))
        elif boundary == WORD:
            # Hold one overlap back for the next chunk to fade into.
            if wav.size > self.overlap and self.overlap:
                self._held = wav[-self.overlap:].copy()
                parts.append(wav[: -self.overlap])
            else:
                self._held = wav
        else:
            parts.append(fade(wav, self.edge_fade, direction="out"))
            self._pending_gap = self._gap_samples(boundary)

        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """Emit whatever is still held, faded out. Idempotent."""
        if self._done:
            return np.zeros(0, dtype=np.float32)
        self._done = True
        held, self._held = self._held, np.zeros(0, dtype=np.float32)
        return fade(held, self.final_fade, direction="out")


def stitch_all(
    waves: list[np.ndarray],
    sample_rate: int,
    boundaries: list[str] | None = None,
    **kwargs,
) -> np.ndarray:
    """Stitch a complete list of chunk waveforms into one array (non-streaming).

    ``boundaries[i]`` is the boundary *after* waves[i]; defaults to treating
    every join as a sentence end, which is the safe assumption when the caller
    has not tracked them.
    """
    if not waves:
        return np.zeros(0, dtype=np.float32)
    if boundaries is None:
        boundaries = [SENTENCE] * (len(waves) - 1) + [END]

    stitcher = StreamStitcher(sample_rate, **kwargs)
    parts = [
        stitcher.push(wav, boundary=boundaries[i], is_final=(i == len(waves) - 1))
        for i, wav in enumerate(waves)
    ]
    parts.append(stitcher.flush())
    parts = [p for p in parts if p.size]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
