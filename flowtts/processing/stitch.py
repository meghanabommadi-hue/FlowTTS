"""Pipeline position: AUDIO STITCHING — per-chunk waveforms → one seamless stream.

Role in pipeline:
  The chunker cuts text so each piece can be synthesized and sent early; this
  module puts the pieces back together so the listener hears one utterance
  rather than four clips played back to back.

      engine.synthesize(chunk_i) → StreamStitcher.push(wav) → PCM bytes → client

Four things have to happen at every boundary, and skipping any of them is
audible:

  **Silence trim.** OmniVoice leaves ~100 ms of lead-in and trail-out silence on
  every generated clip. Concatenated, that is a ~200 ms hole at each seam — the
  single most obvious streaming artifact. Trimmed here with a fast numpy RMS
  gate rather than OmniVoice's pydub path, which is far too slow per chunk.

  **DC removal.** Each chunk decodes with its own small DC offset. Butting two
  different offsets together is a step discontinuity, which is a click.

  **Equal-power crossfade.** A linear crossfade dips ~3 dB in the middle because
  two uncorrelated signals sum in power, not amplitude. The sin/cos pair used
  here holds power constant across the overlap.

  **Level match.** Chunk-to-chunk loudness drifts a little. The tail of the
  outgoing chunk and the head of the incoming one are matched, but by at most
  ``max_gain_db`` so a genuinely quiet passage is not pumped up.

The crossfade needs the *next* chunk to exist, so each push holds back one
overlap's worth of samples and emits it fused with the following chunk. That
costs ``overlap_ms`` of extra latency once (20 ms by default) and nothing after.

Pure NumPy — no torch, no GPU. See flowtts/test/test_stitch.py.
"""

from __future__ import annotations

import numpy as np

# Below this RMS a 10 ms window counts as silence for edge trimming. -50 dBFS,
# matching the threshold OmniVoice's own remove_silence() uses.
_SILENCE_RMS = 10 ** (-50 / 20)
_TRIM_WINDOW_MS = 10.0


def remove_dc(wav: np.ndarray) -> np.ndarray:
    """Subtract the mean so chunks butt together without a step discontinuity."""
    if wav.size == 0:
        return wav
    return wav - wav.mean()


def trim_silence(
    wav: np.ndarray,
    sample_rate: int,
    *,
    keep_ms: float = 20.0,
    threshold: float = _SILENCE_RMS,
) -> np.ndarray:
    """Trim leading/trailing silence, keeping ``keep_ms`` of it as breathing room.

    A windowed RMS gate rather than a per-sample one: a per-sample threshold
    trips on the zero crossings inside ordinary speech and eats real audio.
    """
    if wav.size == 0:
        return wav

    win = max(1, int(sample_rate * _TRIM_WINDOW_MS / 1000))
    n_win = wav.size // win
    if n_win < 2:
        return wav

    frames = wav[: n_win * win].reshape(n_win, win)
    loud = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) > threshold
    if not loud.any():
        # Chunk is silent end to end — keep a token amount so the stream has
        # continuity rather than a hard zero-length gap.
        return wav[: max(1, int(sample_rate * keep_ms / 1000))]

    keep = max(0, int(sample_rate * keep_ms / 1000))
    first = max(0, int(np.argmax(loud)) * win - keep)
    last = min(wav.size, (n_win - int(np.argmax(loud[::-1]))) * win + keep)
    return wav[first:last]


def equal_power_crossfade(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Blend ``tail`` into ``head`` over their common length at constant power."""
    n = min(len(tail), len(head))
    if n == 0:
        return np.concatenate([tail, head])
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    fade_out, fade_in = np.cos(t), np.sin(t)
    return tail[-n:] * fade_out + head[:n] * fade_in


def match_level(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    max_gain_db: float = 1.5,
) -> float:
    """Gain that brings ``target``'s RMS to ``reference``'s, clamped to ±max_gain_db."""
    ref_rms = float(np.sqrt((reference.astype(np.float64) ** 2).mean())) if reference.size else 0.0
    tgt_rms = float(np.sqrt((target.astype(np.float64) ** 2).mean())) if target.size else 0.0
    if ref_rms < 1e-6 or tgt_rms < 1e-6:
        return 1.0
    limit = 10 ** (max_gain_db / 20)
    return float(np.clip(ref_rms / tgt_rms, 1 / limit, limit))


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
    """Joins successive chunk waveforms into one continuous stream.

    Usage::

        st = StreamStitcher(sample_rate=24000)
        for wav, is_final in chunks:
            audio = st.push(wav, is_final=is_final)
            if audio.size:
                send(pcm_int16(audio))

    ``push`` returns the audio that is safe to send now — everything except the
    overlap tail being held for the next chunk. The final push flushes that tail
    with a fade-out, so no explicit flush call is required.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        overlap_ms: float = 20.0,
        edge_fade_ms: float = 4.0,
        final_fade_ms: float = 12.0,
        trim: bool = True,
        trim_keep_ms: float = 20.0,
        level_match: bool = True,
        max_gain_db: float = 1.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.overlap = max(0, int(sample_rate * overlap_ms / 1000))
        self.edge_fade = max(0, int(sample_rate * edge_fade_ms / 1000))
        self.final_fade = max(0, int(sample_rate * final_fade_ms / 1000))
        self.trim = trim
        self.trim_keep_ms = trim_keep_ms
        self.level_match = level_match
        self.max_gain_db = max_gain_db

        self._held = np.zeros(0, dtype=np.float32)   # overlap tail awaiting a partner
        self._first = True
        self._done = False

    # ------------------------------------------------------------------ helpers
    def _prepare(self, wav: np.ndarray) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            return wav
        # Trim before removing DC, not after: subtracting a whole-clip mean
        # lifts the silent regions off zero, and a lifted floor sits above the
        # -50 dBFS gate, so the trim then finds no silence to remove at all.
        if self.trim:
            wav = trim_silence(wav, self.sample_rate, keep_ms=self.trim_keep_ms)
        return remove_dc(wav)

    # ------------------------------------------------------------------ public
    def push(self, wav: np.ndarray, *, is_final: bool = False) -> np.ndarray:
        """Add one chunk; return the audio ready to send."""
        if self._done:
            return np.zeros(0, dtype=np.float32)

        wav = self._prepare(wav)

        if wav.size == 0:
            # Empty chunk (all silence, or a text chunk that produced nothing):
            # keep the held tail so the next real chunk still fuses with it.
            return self.flush() if is_final else np.zeros(0, dtype=np.float32)

        if self._first:
            # Nothing to fuse with — just soften the very start of the stream.
            out = fade(wav, self.edge_fade, direction="in")
            self._first = False
        else:
            if self.level_match:
                gain = match_level(self._held, wav[: self.overlap or wav.size],
                                   max_gain_db=self.max_gain_db)
                if gain != 1.0:
                    wav = wav * gain
            fused = equal_power_crossfade(self._held, wav)
            # `fused` covers the first len(fused) samples of `wav`; the held tail
            # is fully consumed by it.
            out = np.concatenate([fused, wav[len(fused):]])
            self._held = np.zeros(0, dtype=np.float32)

        if is_final:
            self._done = True
            return fade(out, self.final_fade, direction="out")

        # Hold back one overlap for the next chunk to fade into.
        if self.overlap and out.size > self.overlap:
            self._held = out[-self.overlap:].copy()
            return out[: -self.overlap]
        # Chunk shorter than the overlap: hold all of it rather than emit a
        # fragment we would then have to crossfade against nothing.
        self._held = out
        return np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """Emit whatever is still held, faded out. Idempotent."""
        if self._done:
            return np.zeros(0, dtype=np.float32)
        self._done = True
        held, self._held = self._held, np.zeros(0, dtype=np.float32)
        return fade(held, self.final_fade, direction="out")


def stitch_all(
    chunks: list[np.ndarray],
    sample_rate: int,
    **kwargs,
) -> np.ndarray:
    """Stitch a complete list of chunk waveforms into one array (non-streaming)."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    stitcher = StreamStitcher(sample_rate, **kwargs)
    parts = [
        stitcher.push(wav, is_final=(i == len(chunks) - 1))
        for i, wav in enumerate(chunks)
    ]
    parts.append(stitcher.flush())
    return np.concatenate([p for p in parts if p.size]) if any(p.size for p in parts) \
        else np.zeros(0, dtype=np.float32)
