"""Pipeline position: SPAN STITCHING — independent spans -> one seamless stream.

Role in pipeline:
  Last DSP stage before bytes go on the wire. Joins the spans the chunker
  created back into audio that sounds like one continuous utterance.

The problem
-----------
Spans are rendered by INDEPENDENT flow trajectories. Each starts from its own
Gaussian noise and each ends up with its own DC offset and phase. Concatenating
them produces a step discontinuity at every join -- an audible click.

`flowtts/server.py` (the older MiraTTS path) handles this by fading IN the head
of each chunk. That works there because consecutive chunks are decoded from a
single autoregressive token stream with overlapping context, so they are already
phase-coherent; the fade only suppresses a codec transient. Here the spans share
no context at all, and a fade-in alone leaves a clearly audible seam: the tail of
span N still ends abruptly at full amplitude.

The fix: true overlap-add
-------------------------
Hold back the last `crossfade` samples of every non-final span. When the next
span arrives, blend that held tail against the new span's head with
complementary linear ramps and emit the blend as the join. Nothing is repeated,
nothing is dropped, and the amplitude envelope is continuous across the seam.

This costs no perceived latency. The held tail is emitted the moment the next
span lands, and spans are produced far faster than real time (RTF well below 1),
so the client's playback cursor is always well behind the held region.
"""

from __future__ import annotations

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def _rms_windows(x: np.ndarray, win: int) -> np.ndarray:
    n = x.size // win
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12)


class SpanStitcher:
    """Streaming overlap-add across the spans of ONE request."""

    def __init__(
        self,
        sample_rate: int,
        crossfade_s: float,
        final_fade_s: float = 0.02,
        trim_edges: bool = True,
        threshold_db: float = -45.0,
    ):
        self.sample_rate = sample_rate
        self.hold = max(0, int(crossfade_s * sample_rate))
        self.final_fade = max(0, int(final_fade_s * sample_rate))
        self.trim_edges = trim_edges
        self.threshold = 10.0 ** (threshold_db / 20.0)
        self._held = np.zeros(0, dtype=np.float32)
        self._spans = 0
        self._emitted = 0

    # -- helpers -------------------------------------------------------------
    def _trim(self, x: np.ndarray) -> np.ndarray:
        """Cheap numpy edge-silence trim.

        Deliberately NOT pydub: upstream's `remove_silence` runs a full
        split-on-silence pass costing 100+ ms, which is more than the flow
        decoder spends on a short span. A windowed RMS scan gets the same
        practical result for edges in microseconds.
        """
        if not self.trim_edges or x.size == 0:
            return x
        win = max(1, self.sample_rate // 100)  # 10 ms
        rms = _rms_windows(x, win)
        if rms.size == 0:
            return x
        loud = np.nonzero(rms > self.threshold)[0]
        if loud.size == 0:
            # An entirely quiet span is usually a model artefact on punctuation;
            # emitting a little silence is better than emitting nothing, which
            # would desynchronise a client counting samples.
            return x[: min(x.size, win * 5)]
        pad = 3  # keep 30 ms so speech onsets are not clipped
        lo = max(0, loud[0] - pad) * win
        hi = min(rms.size, loud[-1] + 1 + pad) * win
        return x[lo:hi]

    @staticmethod
    def _ramp(n: int, up: bool) -> np.ndarray:
        if up:
            return np.linspace(0.0, 1.0, n, dtype=np.float32)
        return np.linspace(1.0, 0.0, n, dtype=np.float32)

    # -- public --------------------------------------------------------------
    def push(self, pcm: np.ndarray, is_final: bool) -> np.ndarray:
        """Return the PCM that is safe to emit now."""
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        pcm = self._trim(pcm)
        self._spans += 1

        if pcm.size == 0:
            if is_final:
                return self.flush()
            return np.zeros(0, dtype=np.float32)

        if self._held.size:
            k = min(self._held.size, pcm.size)
            if pcm.size < self._held.size:
                # Degenerate: this span is shorter than the crossfade itself.
                # Emit the held tail intact rather than discarding samples; the
                # seam is slightly worse but the stream stays sample-exact.
                merged = np.concatenate([self._held, pcm])
            else:
                head = self._held[:k] * self._ramp(k, up=False) + pcm[:k] * self._ramp(k, up=True)
                merged = np.concatenate([head, pcm[k:]])
            self._held = np.zeros(0, dtype=np.float32)
        else:
            merged = pcm

        if is_final:
            if self.final_fade and merged.size > self.final_fade:
                merged = merged.copy()
                merged[-self.final_fade:] *= self._ramp(self.final_fade, up=False)
            self._emitted += merged.size
            return merged

        # Hold at most half the span so a short span cannot be swallowed whole.
        h = min(self.hold, merged.size // 2)
        if h > 0:
            self._held = merged[-h:].copy()
            out = merged[:-h]
        else:
            out = merged
        self._emitted += out.size
        return out

    def flush(self) -> np.ndarray:
        """Emit anything still held back (call when a stream ends early)."""
        if not self._held.size:
            return np.zeros(0, dtype=np.float32)
        out = self._held
        if self.final_fade and out.size > self.final_fade:
            out = out.copy()
            out[-self.final_fade:] *= self._ramp(self.final_fade, up=False)
        self._held = np.zeros(0, dtype=np.float32)
        self._emitted += out.size
        return out

    def stats(self) -> dict:
        return {"spans": self._spans, "emitted_samples": self._emitted}
