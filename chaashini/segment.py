"""Fuse VAD speech regions with diarization to produce single-speaker, overlap-free chunks of
0.5 s .. 30 s, split preferentially at the quietest pauses."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DiarCfg, VADCfg
from .vad import VADFrames


@dataclass
class Chunk:
    start_ms: int
    end_ms: int
    speaker: int
    vad_ratio: float
    dominance: float
    diar_coverage: float
    n_speakers_seen: int
    split_reason: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def dur_ms(self) -> int:
        return self.end_ms - self.start_ms


def _median_filter_int(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(a) < k:
        return a
    out = a.copy()
    h = k // 2
    for i in range(len(a)):
        w = a[max(0, i - h): i + h + 1]
        vals, counts = np.unique(w, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def diar_frames(probs: np.ndarray, frame_ms: int, total_ms: int, active_prob: float, vad_frame_ms: float,
                smooth_frames: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Resample diarization probs [T, S] (frame_ms per row) to the VAD frame grid.
    Returns (dominant_speaker[-1 = none], n_active) per VAD frame."""
    n_vad = int(np.ceil(total_ms / vad_frame_ms))
    if probs is None or len(probs) == 0:
        return np.full(n_vad, -1, dtype=np.int16), np.zeros(n_vad, dtype=np.int16)
    probs = np.asarray(probs, dtype=np.float32)
    active = probs >= active_prob
    n_active = active.sum(axis=1).astype(np.int16)
    dom = np.where(active.any(axis=1), probs.argmax(axis=1), -1).astype(np.int16)
    dom = _median_filter_int(dom, smooth_frames)
    # map VAD frames -> diar frames
    t_ms = (np.arange(n_vad) + 0.5) * vad_frame_ms
    idx = np.clip((t_ms / frame_ms).astype(int), 0, len(probs) - 1)
    return dom[idx], n_active[idx]


def _split_points(probs: np.ndarray, fm: float, s: int, e: int, cfg: VADCfg) -> list[int]:
    """Choose split frame indices inside [s, e) so every piece <= max_chunk and near target."""
    max_f = int(cfg.max_chunk_ms / fm)
    tgt_f = int(cfg.target_chunk_ms / fm)
    min_f = int(cfg.min_speech_ms / fm)
    min_sil = max(1, int(cfg.min_silence_split_ms / fm))
    out: list[int] = []
    cur = s
    # smoothed prob for dip search
    k = 3
    sm = np.convolve(probs, np.ones(k) / k, mode="same") if len(probs) >= k else probs
    while e - cur > max_f:
        lo = cur + max(min_f, int(tgt_f * 0.5))
        hi = min(cur + max_f, e - min_f)
        if hi <= lo:
            out.append(cur + max_f)
            cur = cur + max_f
            continue
        seg = sm[lo:hi]
        # prefer a genuine pause: a run of >= min_sil frames under 0.35, closest to target length
        best, best_score = None, None
        below = seg < 0.35
        i = 0
        while i < len(below):
            if below[i]:
                j = i
                while j < len(below) and below[j]:
                    j += 1
                if j - i >= min_sil:
                    mid = lo + (i + j) // 2
                    score = abs((mid - cur) - tgt_f) / tgt_f - 0.15 * min(3, (j - i) / min_sil)
                    if best_score is None or score < best_score:
                        best, best_score = mid, score
                i = j
            else:
                i += 1
        if best is None:
            best = lo + int(np.argmin(seg))          # quietest single frame
        out.append(best)
        cur = best
    return out


def make_chunks(vad: VADFrames, regions: list[tuple[int, int]], dom: np.ndarray, n_active: np.ndarray,
                vcfg: VADCfg, dcfg: DiarCfg) -> list[Chunk]:
    fm = vad.frame_ms
    probs = vad.probs
    margin = int(dcfg.overlap_margin_ms / fm)
    n = len(probs)
    # overlap mask with margin
    ov = (n_active >= 2)
    if ov.any() and margin > 0:
        idx = np.flatnonzero(ov)
        ovm = np.zeros(n, dtype=bool)
        for i in idx:
            ovm[max(0, i - margin): i + margin + 1] = True
        ov = ovm
    chunks: list[Chunk] = []
    for (s_ms, e_ms) in regions:
        s, e = int(s_ms / fm), min(n, int(np.ceil(e_ms / fm)))
        if e - s <= 0:
            continue
        # split region by overlap and by speaker change
        pieces: list[tuple[int, int, int]] = []   # (s, e, spk)
        i = s
        while i < e:
            if ov[i]:
                i += 1
                continue
            j = i
            spk = int(dom[i])
            run_spk = spk
            while j < e and not ov[j]:
                d = int(dom[j])
                if d != -1 and run_spk == -1:
                    run_spk = d
                if d != -1 and run_spk != -1 and d != run_spk:
                    break
                j += 1
            pieces.append((i, j, run_spk))
            i = j
        for (ps, pe, spk) in pieces:
            if (pe - ps) * fm < vcfg.min_speech_ms:
                continue
            cuts = _split_points(probs, fm, ps, pe, vcfg)
            bounds = [ps] + cuts + [pe]
            for a, b in zip(bounds[:-1], bounds[1:]):
                if (b - a) * fm < vcfg.min_speech_ms:
                    continue
                seg_dom = dom[a:b]
                seen = set(int(x) for x in np.unique(seg_dom) if x >= 0)
                if spk == -1 and seen:
                    spk = int(max(seen, key=lambda k: int((seg_dom == k).sum())))
                cov = float((seg_dom >= 0).mean()) if b > a else 0.0
                dominance = float((seg_dom == spk).sum() / max(1, (seg_dom >= 0).sum())) if spk >= 0 else 0.0
                vr = float((probs[a:b] >= vcfg.threshold).mean())
                chunks.append(Chunk(int(a * fm), int(b * fm), spk, vr, dominance, cov, len(seen),
                                    "pause" if cuts else "region"))
    return chunks
