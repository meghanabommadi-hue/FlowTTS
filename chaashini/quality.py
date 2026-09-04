"""Audio quality scoring on the CPU box.

* DNSMOS P.835 (SIG / BAK / OVRL) + P.808 MOS  - non-intrusive perceptual quality (ONNX)
* AudioSet tagger (CNN14) - music / singing / background-music / speech / noise probabilities
* Signal metrics - RMS, peak, clipping ratio, DC offset, SNR from VAD-informed noise floor,
  effective bandwidth from the long-term average spectrum.

All scorers are lazily constructed once per process and reused.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("chaashini.quality")

DNSMOS_SR = 16000
DNSMOS_LEN = 9.01

_MUSIC_LABELS = ("Music", "Background music", "Singing", "Musical instrument", "Theme music", "Soundtrack music",
                 "Jingle (music)", "Song", "Pop music", "Electronic music", "Hip hop music", "Rock music",
                 "Music of Bollywood", "Music for children", "Drum", "Guitar", "Piano", "Synthesizer", "Rapping",
                 "Chant", "Mantra", "Choir", "Keyboard (musical)", "Percussion", "Tabla", "Sitar", "Harmonium", "Flute", "Violin, fiddle")
_SPEECH_LABELS = ("Speech", "Narration, monologue", "Conversation", "Male speech, man speaking",
                  "Female speech, woman speaking", "Child speech, kid speaking", "Speech synthesizer")
_NOISE_LABELS = ("Noise", "Static", "White noise", "Pink noise", "Hum", "Wind noise (microphone)", "Wind", "Vehicle",
                 "Traffic noise, roadway noise", "Crowd", "Applause", "Laughter", "Television", "Radio", "Echo",
                 "Reverberation", "Distortion", "Clicking", "Air conditioning", "Mechanical fan", "Rain", "Buzz",
                 "Environmental noise", "Inside, small room", "Inside, large room or hall", "Inside, public space",
                 "Outside, urban or manmade", "Outside, rural or natural", "Cacophony", "Crackle")


@dataclass
class DNSMOSScore:
    sig: float
    bak: float
    ovrl: float
    p808: float


class DNSMOS:
    def __init__(self, model_dir: str | Path, threads: int = 2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        d = Path(model_dir)
        self.sess = ort.InferenceSession(str(d / "sig_bak_ovr.onnx"), so, providers=["CPUExecutionProvider"])
        self.p808 = ort.InferenceSession(str(d / "model_v8.onnx"), so, providers=["CPUExecutionProvider"])
        self._lock = threading.Lock()

    @staticmethod
    def _melspec(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        import librosa
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=321, hop_length=160, n_mels=120)
        mel = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
        return mel.T

    @staticmethod
    def _poly(sig, bak, ovr):
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return float(p_sig(sig)), float(p_bak(bak)), float(p_ovr(ovr))

    def score(self, audio: np.ndarray, sr: int = 16000, max_windows: int = 3) -> DNSMOSScore:
        assert sr == DNSMOS_SR
        x = audio.astype(np.float32)
        if x.size == 0:
            return DNSMOSScore(1.0, 1.0, 1.0, 1.0)
        need = int(DNSMOS_LEN * sr)
        short = len(x) < need
        while len(x) < need:                       # tile short clips, as the reference implementation does
            x = np.concatenate([x, x])
        n_hops = 1 if short else max(1, int(math.floor(len(x) / sr) - DNSMOS_LEN) + 1)
        # spread at most `max_windows` windows over the clip
        if n_hops > max_windows:
            starts = np.linspace(0, n_hops - 1, max_windows).round().astype(int)
        else:
            starts = np.arange(n_hops)
        sig, bak, ovr, p8 = [], [], [], []
        with self._lock:
            for h in starts:
                seg = x[int(h * sr): int((h + DNSMOS_LEN) * sr)]
                if len(seg) < need:
                    continue
                feats = seg[np.newaxis, :].astype(np.float32)
                mel = self._melspec(seg[:-160])[np.newaxis, :, :].astype(np.float32)
                p808 = float(self.p808.run(None, {"input_1": mel})[0][0][0])
                s, b, o = self.sess.run(None, {"input_1": feats})[0][0]
                s, b, o = self._poly(s, b, o)
                sig.append(s); bak.append(b); ovr.append(o); p8.append(p808)
        if not sig:
            return DNSMOSScore(1.0, 1.0, 1.0, 1.0)
        return DNSMOSScore(float(np.mean(sig)), float(np.mean(bak)), float(np.mean(ovr)), float(np.mean(p8)))


class Tagger:
    """AudioSet CNN14 tagger (32 kHz). Returns per-window probabilities for music/speech/noise groups."""
    SR = 32000

    def __init__(self, device: str = "cpu", threads: int = 8):
        import torch
        torch.set_num_threads(threads)
        from panns_inference import AudioTagging, labels
        self.at = AudioTagging(checkpoint_path=None, device=device)
        self.labels = list(labels)
        idx = {l: i for i, l in enumerate(self.labels)}
        self.music_idx = [idx[l] for l in _MUSIC_LABELS if l in idx]
        self.speech_idx = [idx[l] for l in _SPEECH_LABELS if l in idx]
        self.noise_idx = [idx[l] for l in _NOISE_LABELS if l in idx]
        self._lock = threading.Lock()

    def tag_windows(self, audio32k: np.ndarray, win_s: float = 4.0, hop_s: float | None = None,
                    batch: int = 32, starts_s: list[float] | None = None) -> dict[str, np.ndarray]:
        """Tag fixed windows. Returns dict of arrays (start_s, music, speech, noise, singing)."""
        import torch
        sr = self.SR
        w = int(win_s * sr)
        hop = int((hop_s or win_s) * sr)
        x = audio32k.astype(np.float32)
        if len(x) < w:
            x = np.pad(x, (0, w - len(x)))
        if starts_s is None:
            starts = list(range(0, max(1, len(x) - w + 1), hop))
            if starts[-1] + w < len(x) - hop // 2:
                starts.append(len(x) - w)
        else:
            starts = [min(max(0, int(s * sr)), len(x) - w) for s in starts_s]
        music, speech, noise, singing = [], [], [], []
        sing_i = self.labels.index("Singing") if "Singing" in self.labels else None
        with self._lock, torch.no_grad():
            for i in range(0, len(starts), batch):
                bs = starts[i: i + batch]
                arr = np.stack([x[s: s + w] for s in bs])
                clip, _ = self.at.inference(arr)
                clip = np.asarray(clip)
                music.extend(clip[:, self.music_idx].max(axis=1).tolist())
                speech.extend(clip[:, self.speech_idx].max(axis=1).tolist())
                noise.extend(clip[:, self.noise_idx].max(axis=1).tolist())
                singing.extend(clip[:, sing_i].tolist() if sing_i is not None else [0.0] * len(bs))
        return {"start_s": np.array([s / sr for s in starts], dtype=np.float32), "win_s": win_s,
                "music": np.array(music, dtype=np.float32), "speech": np.array(speech, dtype=np.float32),
                "noise": np.array(noise, dtype=np.float32), "singing": np.array(singing, dtype=np.float32)}


def window_stat(tags: dict, start_s: float, end_s: float, key: str, reducer=np.max) -> float:
    st = tags["start_s"]
    w = float(tags["win_s"])
    sel = (st < end_s) & (st + w > start_s)
    vals = tags[key][sel]
    if vals.size == 0:
        i = int(np.argmin(np.abs(st - start_s)))
        return float(tags[key][i]) if len(tags[key]) else 0.0
    return float(reducer(vals))


# ----------------------------------------------------------------------------- signal metrics
def frame_rms(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    f = x[: n * hop].reshape(n, hop).astype(np.float32)
    return np.sqrt((f ** 2).mean(axis=1) + 1e-12)


def dbfs(v: float) -> float:
    return 20.0 * math.log10(max(v, 1e-9))


@dataclass
class SignalMetrics:
    rms_dbfs: float
    peak_dbfs: float
    clipping_ratio: float
    dc_offset: float
    snr_db: float
    bandwidth_hz: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"rms_dbfs": round(self.rms_dbfs, 2), "peak_dbfs": round(self.peak_dbfs, 2),
                "clipping_ratio": round(self.clipping_ratio, 6), "dc_offset": round(self.dc_offset, 5),
                "snr_db": round(self.snr_db, 2), "bandwidth_hz": round(self.bandwidth_hz, 0), **self.extra}


def noise_floor_from_vad(x16: np.ndarray, vad_probs: np.ndarray, hop: int, lo: float = 0.15) -> float:
    """Global noise floor (linear RMS) for a whole recording: 10th percentile of non-speech frame RMS."""
    r = frame_rms(x16, hop)
    n = min(len(r), len(vad_probs))
    r, p = r[:n], vad_probs[:n]
    ns = r[p < lo]
    if ns.size < 20:
        return float(np.percentile(r, 5)) if r.size else 1e-4
    return float(np.percentile(ns, 10))


def signal_metrics(x16: np.ndarray, sr16: int, vad_probs: np.ndarray | None, hop: int,
                   global_noise_floor: float | None, x_hi: np.ndarray | None = None, sr_hi: int | None = None) -> SignalMetrics:
    """x16: float32 chunk at 16 kHz (for level/SNR); x_hi/sr_hi: same chunk at export rate (for bandwidth)."""
    x = x16.astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12)) if x.size else 0.0
    clip = clipping_ratio(x)
    dc = float(np.mean(x)) if x.size else 0.0
    r = frame_rms(x, hop)
    snr = 0.0
    if vad_probs is not None and len(r):
        n = min(len(r), len(vad_probs))
        r, p = r[:n], vad_probs[:n]
        sp = r[p >= 0.6]
        ns = r[p < 0.15]
        speech_lvl = float(np.sqrt(np.mean(sp ** 2))) if sp.size >= 5 else rms
        if ns.size >= 8:
            local_floor = float(np.percentile(ns, 20))
        else:
            local_floor = global_noise_floor if global_noise_floor else float(np.percentile(r, 5))
        floor = local_floor
        if global_noise_floor:
            floor = min(local_floor, max(global_noise_floor, 1e-5)) if ns.size >= 8 else global_noise_floor
        snr = 20.0 * math.log10(max(speech_lvl, 1e-9) / max(floor, 1e-9))
        snr = float(min(max(snr, -10.0), 70.0))
    bw = bandwidth_hz(x_hi if x_hi is not None else x, sr_hi or sr16)
    return SignalMetrics(dbfs(rms), dbfs(peak), clip, dc, snr, bw)


def clipping_ratio(x: np.ndarray, thresh: float = 0.985, min_run: int = 3) -> float:
    """Fraction of samples that sit inside flat-topped runs (>= `min_run` consecutive samples at
    full scale). Lone peaks from loudness limiting do not count; real digital clipping does."""
    if x.size == 0:
        return 0.0
    hot = np.abs(x) >= thresh
    if not hot.any():
        return 0.0
    d = np.diff(np.concatenate([[0], hot.view(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = ends - starts
    return float(runs[runs >= min_run].sum() / x.size)


def bandwidth_hz(x: np.ndarray, sr: int, drop_db: float = 60.0) -> float:
    """Highest frequency whose long-term average spectrum is within `drop_db` of the peak band."""
    if x.size < sr // 4:
        return 0.0
    from scipy.signal import welch
    f, p = welch(x.astype(np.float32), fs=sr, nperseg=2048)
    p_db = 10 * np.log10(p + 1e-14)
    # smooth over ~200 Hz
    k = max(1, int(200 / (f[1] - f[0])))
    if k > 1:
        p_db = np.convolve(p_db, np.ones(k) / k, mode="same")
    ref = p_db[(f >= 300) & (f <= 3000)].max()
    above = np.flatnonzero(p_db >= ref - drop_db)
    return float(f[above[-1]]) if above.size else 0.0


# ----------------------------------------------------------------------------- lazy singletons
_dnsmos: DNSMOS | None = None
_tagger: Tagger | None = None
_lock = threading.Lock()


def get_dnsmos(model_dir: str | Path, threads: int = 4) -> DNSMOS:
    global _dnsmos
    with _lock:
        if _dnsmos is None:
            _dnsmos = DNSMOS(model_dir, threads)
    return _dnsmos


def get_tagger(threads: int = 8) -> Tagger:
    global _tagger
    with _lock:
        if _tagger is None:
            _tagger = Tagger("cpu", threads)
    return _tagger
