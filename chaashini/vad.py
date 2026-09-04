"""TenVAD wrapper: frame-level speech probabilities and smoothed speech regions."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("chaashini.vad")


@dataclass
class VADFrames:
    probs: np.ndarray        # float32 per hop
    flags: np.ndarray        # uint8 per hop
    hop: int
    sr: int

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.hop / self.sr

    def prob_at(self, t_ms: float) -> float:
        i = int(t_ms / self.frame_ms)
        return float(self.probs[min(max(i, 0), len(self.probs) - 1)]) if len(self.probs) else 0.0


def run_vad(pcm_int16: np.ndarray, sr: int = 16000, hop: int = 256, threshold: float = 0.5) -> VADFrames:
    from ten_vad import TenVad
    assert sr == 16000, "TenVAD expects 16 kHz"
    n = len(pcm_int16) // hop
    probs = np.zeros(n, dtype=np.float32)
    flags = np.zeros(n, dtype=np.uint8)
    vad = TenVad(hop, threshold)
    x = np.ascontiguousarray(pcm_int16[: n * hop]).reshape(n, hop)
    for i in range(n):
        p, f = vad.process(x[i])
        probs[i] = p
        flags[i] = f
    del vad
    return VADFrames(probs, flags, hop, sr)


def speech_regions(v: VADFrames, threshold: float = 0.5, min_speech_ms: int = 250, merge_gap_ms: int = 200,
                   pad_ms: int = 100, hangover_ms: int = 120) -> list[tuple[int, int]]:
    """Binary regions [start_ms, end_ms) from probabilities with hangover + merge + pad."""
    fm = v.frame_ms
    on = v.probs >= threshold
    if not on.any():
        return []
    hang = max(1, int(round(hangover_ms / fm)))
    # extend each speech frame by `hang` frames (hangover) to keep trailing consonants
    idx = np.flatnonzero(on)
    ext = np.zeros_like(on)
    for i in idx:
        ext[i: i + hang + 1] = True
    regions: list[list[int]] = []
    start = None
    for i, f in enumerate(ext):
        if f and start is None:
            start = i
        elif not f and start is not None:
            regions.append([start, i])
            start = None
    if start is not None:
        regions.append([start, len(ext)])
    merged: list[list[int]] = []
    gap = merge_gap_ms / fm
    for r in regions:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    total_ms = len(v.probs) * fm
    out: list[tuple[int, int]] = []
    for s, e in merged:
        s_ms, e_ms = s * fm, e * fm
        if e_ms - s_ms < min_speech_ms:
            continue
        out.append((int(max(0, s_ms - pad_ms)), int(min(total_ms, e_ms + pad_ms))))
    return out


def speech_ratio(v: VADFrames, start_ms: int, end_ms: int, threshold: float = 0.5) -> float:
    a, b = int(start_ms / v.frame_ms), int(end_ms / v.frame_ms)
    seg = v.probs[a:b]
    return float((seg >= threshold).mean()) if len(seg) else 0.0
