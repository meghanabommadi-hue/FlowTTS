"""Pipeline position: API ENCODING — float32 waveform → wire bytes.

Role in pipeline:
  The last step before audio leaves the process, for both the streaming and the
  whole-file paths.

      StreamChunk.audio (float32 @ 24 kHz)
        → resample to the requested rate
        → int16 PCM  ─┬─ raw            (WebSocket frames, format="pcm")
                      ├─ streaming WAV  (chunked HTTP: header first, then PCM)
                      └─ container      (wav / mp3 / opus for a complete response)

Streaming a WAV over chunked HTTP needs a header written before the length is
known. :func:`streaming_wav_header` emits one with placeholder sizes, which is
what every browser, ffmpeg and curl-to-file consumer expects from a live stream;
:func:`encode_audio` writes a correct header when the whole waveform is in hand.
"""

from __future__ import annotations

import io
import shutil
import struct
import subprocess
from typing import Optional

import numpy as np

# Formats we can always produce. mp3/opus additionally need ffmpeg or libsndfile
# with Opus support; encode_audio reports what it actually produced.
ALWAYS_AVAILABLE = ("wav", "pcm")

CONTENT_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/L16",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
}


def resample(wav: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample float32 audio, preferring a real resampler over interpolation.

    ``soxr`` (VHQ) and ``scipy.signal.resample_poly`` are both band-limited;
    the linear-interpolation fallback aliases audibly when downsampling 24 kHz
    to 8 kHz, which is exactly the telephony case this server serves.
    """
    if source_rate == target_rate or wav.size == 0:
        return wav

    try:
        import soxr
        return soxr.resample(wav, source_rate, target_rate, quality="VHQ").astype(np.float32)
    except Exception:  # noqa: BLE001 — optional dependency
        pass

    try:
        from math import gcd

        from scipy.signal import resample_poly
        divisor = gcd(source_rate, target_rate)
        return resample_poly(wav, target_rate // divisor,
                             source_rate // divisor).astype(np.float32)
    except Exception:  # noqa: BLE001 — optional dependency
        pass

    target_len = int(round(len(wav) * target_rate / source_rate))
    indices = np.linspace(0, len(wav) - 1, target_len, dtype=np.float64)
    return np.interp(indices, np.arange(len(wav)), wav).astype(np.float32)


def to_pcm16(wav: np.ndarray) -> bytes:
    """float32 in [-1, 1] → raw little-endian int16 bytes (no header)."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return b""
    np.clip(wav, -1.0, 1.0, out=wav)
    return (wav * 32767.0).astype("<i2").tobytes()


def wav_header(sample_rate: int, data_bytes: int, channels: int = 1,
               bits: int = 16) -> bytes:
    """A 44-byte canonical PCM WAV header for a known payload size."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return b"".join([
        b"RIFF", struct.pack("<I", 36 + data_bytes), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                             byte_rate, block_align, bits),
        b"data", struct.pack("<I", data_bytes),
    ])


def streaming_wav_header(sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """A WAV header for a stream whose length is not yet known.

    The two size fields are set to 0xFFFFFFFF, the conventional "unknown /
    streaming" marker. Players read until the connection closes instead of
    trusting the length.
    """
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return b"".join([
        b"RIFF", struct.pack("<I", 0xFFFFFFFF), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                             byte_rate, block_align, bits),
        b"data", struct.pack("<I", 0xFFFFFFFF),
    ])


def _encode_with_soundfile(wav: np.ndarray, sample_rate: int,
                           fmt: str, subtype: str) -> Optional[bytes]:
    try:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, wav, sample_rate, format=fmt, subtype=subtype)
        return buf.getvalue()
    except Exception:  # noqa: BLE001 — format not built into this libsndfile
        return None


def _encode_with_ffmpeg(pcm: bytes, sample_rate: int, args: list[str]) -> Optional[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error",
           "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
           *args, "pipe:1"]
    try:
        done = subprocess.run(cmd, input=pcm, capture_output=True, check=True)
        return done.stdout or None
    except Exception:  # noqa: BLE001
        return None


def encode_audio(
    wav: np.ndarray,
    sample_rate: int,
    fmt: str = "wav",
) -> tuple[bytes, str, str]:
    """Encode a complete waveform. Returns ``(bytes, actual_format, content_type)``.

    mp3 and opus fall back to WAV when no encoder is available rather than
    failing the request — a caller who asked for mp3 to save bandwidth still
    wants their audio, and the returned format tells them what they got.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)

    if fmt == "pcm":
        return to_pcm16(wav), "pcm", CONTENT_TYPES["pcm"]

    if fmt == "wav":
        pcm = to_pcm16(wav)
        return wav_header(sample_rate, len(pcm)) + pcm, "wav", CONTENT_TYPES["wav"]

    if fmt == "opus":
        data = _encode_with_soundfile(wav, sample_rate, "OGG", "OPUS")
        if data is None:
            data = _encode_with_ffmpeg(to_pcm16(wav), sample_rate,
                                       ["-c:a", "libopus", "-b:a", "48k", "-f", "ogg"])
        if data is not None:
            return data, "opus", CONTENT_TYPES["opus"]

    if fmt == "mp3":
        data = _encode_with_ffmpeg(to_pcm16(wav), sample_rate,
                                   ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"])
        if data is not None:
            return data, "mp3", CONTENT_TYPES["mp3"]

    pcm = to_pcm16(wav)
    return wav_header(sample_rate, len(pcm)) + pcm, "wav", CONTENT_TYPES["wav"]


def streaming_content_type(fmt: str) -> str:
    """Content-Type for a chunked stream in *fmt* (streaming is PCM or WAV only)."""
    return CONTENT_TYPES.get("pcm" if fmt == "pcm" else "wav", "audio/wav")


def available_formats() -> list[str]:
    """Formats this box can actually produce right now."""
    formats = list(ALWAYS_AVAILABLE)
    if shutil.which("ffmpeg"):
        formats += ["mp3", "opus"]
    elif _encode_with_soundfile(np.zeros(8, dtype=np.float32), 24000, "OGG", "OPUS"):
        formats.append("opus")
    return formats
