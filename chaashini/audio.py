"""ffmpeg-based decode / cut / encode helpers. Everything streams through pipes; no temp WAVs
except the per-video 16 kHz analysis master."""
from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger("chaashini.audio")
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def probe(path: str | Path) -> dict:
    out = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def probe_duration_sr(path: str | Path) -> tuple[float, int]:
    p = probe(path)
    dur = float(p.get("format", {}).get("duration") or 0.0)
    sr = 0
    for s in p.get("streams", []):
        if s.get("codec_type") == "audio":
            sr = int(s.get("sample_rate") or 0)
            if not dur and s.get("duration"):
                dur = float(s["duration"])
            break
    return dur, sr


def decode_to_wav(src: str | Path, dst: str | Path, sr: int = 16000) -> tuple[float, int]:
    """Decode any container to mono 16-bit PCM WAV at `sr` (the analysis master)."""
    cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr),
           "-sample_fmt", "s16", "-af", "aresample=resampler=soxr", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)
    info = sf.info(str(dst))
    return float(info.duration), int(info.samplerate)


def read_wav_int16(path: str | Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="int16", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    return data, int(sr)


def cut_to_array(src: str | Path, start_s: float, end_s: float, sr: int) -> np.ndarray:
    """Accurately cut [start_s, end_s) from `src`, resampled to `sr`, mono float32 in [-1, 1]."""
    dur = max(0.0, end_s - start_s)
    cmd = [FFMPEG, "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{dur:.3f}", "-vn",
           "-ac", "1", "-ar", str(sr), "-af", "aresample=resampler=soxr", "-f", "f32le", "-"]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def cut_to_wav(src: str | Path, start_s: float, end_s: float, sr: int, dst: str | Path) -> None:
    dur = max(0.0, end_s - start_s)
    cmd = [FFMPEG, "-v", "error", "-y", "-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{dur:.3f}", "-vn",
           "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", "-af", "aresample=resampler=soxr", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr_from, sr_to)
    return resample_poly(x.astype(np.float32), sr_to // g, sr_from // g).astype(np.float32)


def encode_bytes(x: np.ndarray, sr: int, fmt: str = "flac", subtype: str = "PCM_16") -> bytes:
    buf = io.BytesIO()
    x = np.clip(x, -1.0, 1.0)
    sf.write(buf, x, sr, format=fmt.upper(), subtype=subtype)
    return buf.getvalue()


def write_file(x: np.ndarray, sr: int, path: str | Path, subtype: str = "PCM_16") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype=subtype)


def peak_normalize(x: np.ndarray, peak_dbfs: float = -1.0) -> np.ndarray:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    if p <= 0:
        return x
    target = 10 ** (peak_dbfs / 20)
    if p > target:
        x = x * (target / p)
    return x
