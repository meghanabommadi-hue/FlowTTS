"""Pipeline position: VOICE PREPARATION — reference clip -> conditioning tensors.

Role in pipeline:
  Runs ONCE per voice, at creation time. Everything expensive about a reference
  clip happens here so the synthesis path never touches audio decoding.

      upload -> decode -> mono -> 24 kHz -> silence removal -> trim
             -> RMS normalise -> Vocos mel -> * feat_scale -> VoicePrompt

Why this is not on the hot path
-------------------------------
`remove_silence` alone (pydub split-on-silence plus edge trimming) costs
100-300 ms of CPU, and mel extraction another few. Doing that per request would
add more to time-to-first-byte than the entire flow decoder does. Upstream's
`generate_sentence` pays it every call; we pay it once and cache the result,
which is the single largest latency win in the whole system. Upstream's own
Triton runtime reaches the same conclusion -- its "speaker cache" cuts p50
latency by roughly a third at concurrency 8.

Why the prompt is trimmed short
-------------------------------
The prompt's mel frames are part of the flow decoder's sequence on EVERY span,
for every ODE step. A 3-second prompt is 281 frames of pure recurring overhead.
The model card recommends 3-10 s for quality; in practice 2-3 s clones just as
well and is markedly cheaper, so `voice.max_prompt_seconds` defaults to 3.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from flowtts.dhvaani.config import MODEL_SAMPLE_RATE, dhv_settings
from flowtts.dhvaani.types import InvalidReferenceAudio

logger = structlog.get_logger(__name__)

# Resampling goes through model/audio_compat, which caches its filter kernels
# and works with or without torchaudio (NGC containers cannot install it).


@dataclass
class PreparedPrompt:
    mel: object          # torch.Tensor (T_p, 100), feat-scaled, engine dtype, on device
    frames: int
    prompt_rms: float
    duration_s: float
    wav_24k: np.ndarray  # post-processing waveform, for preview / debugging


def _decode_audio(audio) -> tuple[np.ndarray, int]:
    """Decode bytes / path to (float32 mono-or-multi array [C, T], sample_rate)."""
    if isinstance(audio, (str, Path)):
        raw = Path(audio).read_bytes()
        name = str(audio)
    elif isinstance(audio, (bytes, bytearray, memoryview)):
        raw = bytes(audio)
        name = "<upload>"
    else:
        raise InvalidReferenceAudio(f"unsupported audio input type: {type(audio)!r}")

    # soundfile handles wav/flac/ogg natively and is the fastest path.
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        return data.T.copy(), int(sr)
    except Exception as e_sf:
        last = e_sf

    try:
        import torchaudio  # optional; unavailable on NGC containers

        wav, sr = torchaudio.load(io.BytesIO(raw))
        return wav.numpy().astype(np.float32), int(sr)
    except Exception as e_ta:
        last = e_ta

    # mp3 / m4a / aac and anything else with an ffmpeg decoder behind it.
    try:
        from pydub import AudioSegment

        seg = AudioSegment.from_file(io.BytesIO(raw))
        arr = np.array(seg.get_array_of_samples()).astype(np.float32)
        peak = float(1 << (8 * seg.sample_width - 1))
        arr /= peak
        if seg.channels > 1:
            arr = arr.reshape(-1, seg.channels).T
        else:
            arr = arr[None, :]
        return arr, int(seg.frame_rate)
    except Exception as e_pd:
        last = e_pd

    raise InvalidReferenceAudio(
        f"could not decode {name}: {last}. Supported: wav, flac, ogg, mp3, m4a "
        "(mp3/m4a need ffmpeg on PATH)."
    )


def _to_mono_24k(data: np.ndarray, sr: int):
    """Mono-mix and resample to 24 kHz, returning a torch tensor of shape (1, T)."""
    import torch

    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)
    wav = torch.from_numpy(np.ascontiguousarray(data)).float()

    if sr != MODEL_SAMPLE_RATE:
        from flowtts.dhvaani.model.audio_compat import resample

        wav = resample(wav, sr, MODEL_SAMPLE_RATE)
    return wav


def _remove_silence(wav, trail_ms: int):
    """Strip long internal silences and edge silence, then add a short tail.

    Mirrors `zipvoice.utils.infer.remove_silence(only_edge=False, trail_sil=200)`.
    The trailing silence matters: without it the prompt's final phoneme runs
    straight into the generated speech and the clone "leaks" the prompt's last
    word into the output.

    pydub is optional; the numpy fallback below does edge trimming only, which
    is the part that actually affects quality most.
    """
    import torch

    try:
        from pydub import AudioSegment
        from pydub.silence import detect_leading_silence, split_on_silence

        arr = (wav.cpu().numpy() * 32768.0).clip(-32768, 32767).astype(np.int16)
        seg = AudioSegment(
            data=arr.tobytes(), sample_width=2,
            frame_rate=MODEL_SAMPLE_RATE, channels=arr.shape[0],
        )
        parts = split_on_silence(
            seg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
        )
        if parts:
            merged = AudioSegment.silent(duration=0)
            for p in parts:
                merged += p
            seg = merged
        for _ in range(2):  # leading, then trailing via reverse
            idx = detect_leading_silence(seg, silence_threshold=-50)
            seg = seg[max(0, idx - 100):].reverse()
        seg = seg + AudioSegment.silent(duration=trail_ms)
        out = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
        return torch.from_numpy(out[None, :])
    except ImportError:
        logger.info("pydub_unavailable_edge_trim_only")
    except Exception as e:
        logger.warning("silence_removal_failed_edge_trim_only", error=str(e)[:200])

    return _edge_trim(wav, trail_ms)


def _edge_trim(wav, trail_ms: int, threshold_db: float = -50.0):
    """numpy-only edge silence trim, used when pydub is unavailable."""
    import torch

    x = wav[0].cpu().numpy()
    win = MODEL_SAMPLE_RATE // 100  # 10 ms
    if x.size < win * 2:
        return wav
    n = x.size // win
    rms = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12)
    thresh = 10 ** (threshold_db / 20.0)
    loud = np.nonzero(rms > thresh)[0]
    if loud.size == 0:
        return wav
    keep_pad = 10  # 100 ms of silence retained at each edge, matching upstream
    lo = max(0, (loud[0] - keep_pad)) * win
    hi = min(n, loud[-1] + 1 + keep_pad) * win
    out = x[lo:hi]
    tail = np.zeros(int(MODEL_SAMPLE_RATE * trail_ms / 1000.0), dtype=np.float32)
    return torch.from_numpy(np.concatenate([out, tail])[None, :])


def _trim_to_seconds(wav, max_seconds: float):
    """Cut the clip to `max_seconds`, preferring a quiet boundary.

    Cutting mid-phoneme leaves a click and a truncated formant that the model
    happily reproduces in the clone, so we look for the quietest 20 ms window in
    the last 25% of the allowance and cut there instead.
    """
    import torch

    limit = int(max_seconds * MODEL_SAMPLE_RATE)
    if wav.shape[-1] <= limit:
        return wav

    x = wav[0].cpu().numpy()
    win = MODEL_SAMPLE_RATE // 50  # 20 ms
    search_lo = max(0, int(limit * 0.75))
    region = x[search_lo:limit]
    n = region.size // win
    if n >= 2:
        rms = np.sqrt((region[: n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12)
        cut = search_lo + (int(np.argmin(rms)) + 1) * win
    else:
        cut = limit
    return torch.from_numpy(np.ascontiguousarray(x[:cut])[None, :])


def _rms_norm(wav, target_rms: float) -> tuple[object, float]:
    """Scale UP quiet clips only, returning the original RMS.

    Identical to `zipvoice.utils.infer.rms_norm`. The original RMS is kept so
    the generated audio can be scaled back down to match the speaker's real
    loudness -- otherwise every clone comes out at the same volume.
    """
    import torch

    rms = float(torch.sqrt(torch.mean(torch.square(wav))))
    if rms < target_rms and rms > 0:
        wav = wav * (target_rms / rms)
    return wav, rms


def prepare_prompt(audio, loaded, settings=None) -> PreparedPrompt:
    """Full reference-audio preparation. Raises InvalidReferenceAudio on bad input."""
    import torch

    st = settings or dhv_settings
    vs, fl = st.voice, st.flow

    data, sr = _decode_audio(audio)
    if data.size == 0:
        raise InvalidReferenceAudio("reference audio is empty")

    wav = _to_mono_24k(data, sr)
    raw_seconds = wav.shape[-1] / MODEL_SAMPLE_RATE
    if raw_seconds < 0.2:
        raise InvalidReferenceAudio(
            f"reference audio is only {raw_seconds:.2f}s; at least "
            f"{vs.min_prompt_seconds}s of speech is required"
        )

    if vs.remove_silence:
        wav = _remove_silence(wav, vs.trailing_silence_ms)

    wav = _trim_to_seconds(wav, vs.max_prompt_seconds)

    seconds = wav.shape[-1] / MODEL_SAMPLE_RATE
    if seconds < vs.min_prompt_seconds:
        raise InvalidReferenceAudio(
            f"after silence removal only {seconds:.2f}s remains; at least "
            f"{vs.min_prompt_seconds}s of actual speech is required. "
            "Check the clip is not mostly silence or noise."
        )

    peak = float(torch.max(torch.abs(wav)))
    if peak < 1e-4:
        raise InvalidReferenceAudio("reference audio is silent")

    wav, prompt_rms = _rms_norm(wav, fl.target_rms)

    mel = loaded.feature_extractor.extract(wav, sampling_rate=MODEL_SAMPLE_RATE)
    if not isinstance(mel, torch.Tensor):
        mel = torch.from_numpy(mel)
    mel = (mel * fl.feat_scale).to(loaded.device, dtype=loaded.dtype)

    if not bool(torch.isfinite(mel).all()):
        raise InvalidReferenceAudio("mel extraction produced non-finite values")

    return PreparedPrompt(
        mel=mel,
        frames=int(mel.shape[0]),
        prompt_rms=prompt_rms,
        duration_s=seconds,
        wav_24k=wav[0].cpu().numpy(),
    )
