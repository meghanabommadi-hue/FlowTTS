"""Pipeline position: AUDIO COMPAT — mel and resampling without torchaudio.

Role in pipeline:
  Used by the mel front end (voice creation) and by the vocode stage (output
  resampling). Prefers torchaudio when it is importable and working, and falls
  back to torch-native implementations that produce numerically identical
  results.

Why this exists
---------------
torchaudio ships a compiled C++ extension that is linked against a specific
libtorch ABI. On NVIDIA NGC PyTorch containers -- a very common GPU deployment
target -- the bundled torch is a custom build (e.g. ``2.8.0a0+...nv25.06``) and
no released torchaudio wheel links against it:

    OSError: libtorchaudio.so: undefined symbol: _ZN3c104cuda9SetDeviceEab

Installing a stock torchaudio would drag in a stock torch and break CUDA for the
whole container. So torchaudio is treated as optional here, and DhVaani only
needs three things from it:

  * ``MelSpectrogram``  -- reproduced exactly by :func:`mel_spectrogram`
  * ``Resample``        -- reproduced by :func:`resample`
  * ``load``            -- already covered by soundfile / pydub in voices/clone.py

Equivalence against torchaudio is asserted by
``flowtts/dhvaani/test/test_audio_compat.py``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

try:  # pragma: no cover - depends on the install
    import torchaudio as _ta

    # Importing is not enough: the extension is loaded lazily on first real use,
    # so probe it here and treat any failure as "unavailable".
    _ta.transforms.MelSpectrogram(sample_rate=16000, n_fft=64, hop_length=16, n_mels=8)
    HAVE_TORCHAUDIO = True
except Exception:  # noqa: BLE001 - any failure means we use the fallback
    _ta = None
    HAVE_TORCHAUDIO = False


# ---------------------------------------------------------------------------
# Mel filterbank (HTK scale, matching torchaudio's defaults)
# ---------------------------------------------------------------------------
def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    """Hz -> mel. Signature matches ``torchaudio.functional.functional._hz_to_mel``,
    which Vocos imports directly."""
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + freq / 700.0)
    if mel_scale != "slaney":
        raise ValueError(f'mel_scale must be "htk" or "slaney", got {mel_scale!r}')
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels, mel_scale: str = "htk"):
    """Mel -> Hz, accepting a tensor or a float, matching torchaudio."""
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    if mel_scale != "slaney":
        raise ValueError(f'mel_scale must be "htk" or "slaney", got {mel_scale!r}')
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if isinstance(mels, torch.Tensor):
        return torch.where(
            mels >= min_log_mel,
            min_log_hz * torch.exp(logstep * (mels - min_log_mel)),
            freqs,
        )
    if mels >= min_log_mel:
        return min_log_hz * math.exp(logstep * (mels - min_log_mel))
    return freqs


def _triangular_filterbank(
    n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``(n_freqs, n_mels)`` filterbank, matching ``torchaudio.functional.melscale_fbanks``
    with ``norm=None, mel_scale="htk"``."""
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs, dtype=dtype)

    m_min, m_max = _hz_to_mel(f_min, "htk"), _hz_to_mel(f_max, "htk")
    m_pts = torch.linspace(m_min, m_max, n_mels + 2, dtype=dtype)
    f_pts = _mel_to_hz(m_pts, "htk")

    f_diff = f_pts[1:] - f_pts[:-1]                       # (n_mels + 1,)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return torch.clamp(torch.minimum(down, up), min=0.0)


@lru_cache(maxsize=8)
def _cached_fbank(n_freqs, f_min, f_max, n_mels, sample_rate, device_str, dtype_str):
    fb = _triangular_filterbank(n_freqs, f_min, f_max, n_mels, sample_rate)
    return fb.to(device=torch.device(device_str), dtype=getattr(torch, dtype_str))


@lru_cache(maxsize=8)
def _cached_window(n: int, device_str: str, dtype_str: str):
    return torch.hann_window(
        n, periodic=True, device=torch.device(device_str), dtype=getattr(torch, dtype_str)
    )


def mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 100,
    power: float = 1.0,
    center: bool = True,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Mel spectrogram identical to ``torchaudio.transforms.MelSpectrogram``.

    Defaults mirror the ones DhVaani's ``VocosFbank`` uses. torchaudio's defaults
    that matter here and are reproduced: a periodic Hann window of ``n_fft``,
    ``pad_mode="reflect"``, ``onesided=True``, ``normalized=False``,
    ``norm=None`` and ``mel_scale="htk"``.

    Args:
        waveform: ``(..., time)``
    Returns:
        ``(..., n_mels, frames)``
    """
    if f_max is None:
        f_max = float(sample_rate // 2)

    dtype = waveform.dtype if waveform.is_floating_point() else torch.float32
    # STFT is done in fp32: at fp16 the reflect-padded edges and the log() below
    # lose enough precision to shift the mel by a visible amount.
    work = waveform.to(torch.float32)
    window = _cached_window(n_fft, str(work.device), "float32")

    spec = torch.stft(
        work.reshape(-1, work.shape[-1]),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = spec.abs() ** power                    # (B, n_freqs, frames)

    fb = _cached_fbank(
        n_fft // 2 + 1, float(f_min), float(f_max), n_mels, sample_rate,
        str(work.device), "float32",
    )
    mel = torch.matmul(spec.transpose(-1, -2), fb).transpose(-1, -2)

    out_shape = waveform.shape[:-1] + mel.shape[-2:]
    return mel.reshape(out_shape).to(dtype)


class VocosFbankCompat:
    """Drop-in replacement for the vendored ``zipvoice.utils.feature.VocosFbank``.

    Same constructor and the same ``extract(samples, sampling_rate)`` contract,
    including its frame-count fix-up, but without importing torchaudio.
    """

    name = "VocosFbankCompat"

    def __init__(self, num_channels: int = 1):
        assert num_channels in (1, 2)
        self.num_channels = num_channels
        self.sampling_rate = 24000
        self.n_mels = 100
        self.n_fft = 1024
        self.hop_length = 256

    @property
    def frame_shift(self) -> float:
        return self.hop_length / self.sampling_rate

    def feature_dim(self, sampling_rate: int) -> int:
        return self.n_mels

    def extract(self, samples, sampling_rate: int):
        import numpy as np

        assert sampling_rate == self.sampling_rate, (
            f"extractor expects {self.sampling_rate}, got {sampling_rate}"
        )
        is_numpy = not isinstance(samples, torch.Tensor)
        if is_numpy:
            samples = torch.from_numpy(np.asarray(samples))
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        else:
            assert samples.ndim == 2, samples.shape
        if self.num_channels == 1 and samples.shape[0] == 2:
            samples = samples.mean(dim=0, keepdims=True)

        mel = mel_spectrogram(
            samples, self.sampling_rate, self.n_fft, self.hop_length,
            self.n_mels, power=1.0, center=True,
        )
        mel = mel.clamp(min=1e-7).log()
        mel = mel.reshape(-1, mel.shape[-1]).t()          # (time, n_mels)

        # Upstream's frame-count reconciliation, reproduced verbatim so a prompt
        # yields the same number of frames either way -- frame count feeds the
        # duration model directly.
        num_samples = samples.shape[1]
        window_hop = round(self.frame_shift * sampling_rate)
        num_frames = int((num_samples + window_hop // 2) // window_hop)
        if mel.shape[0] > num_frames:
            mel = mel[:num_frames]
        elif mel.shape[0] < num_frames:
            mel = mel.unsqueeze(0)
            mel = torch.nn.functional.pad(
                mel, (0, 0, 0, num_frames - mel.shape[1]), mode="replicate"
            ).squeeze(0)
        return mel.cpu().numpy() if is_numpy else mel


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _resample_kernel(orig: int, new: int, device_str: str, width: int = 16):
    """Windowed-sinc polyphase kernel, matching torchaudio's default
    ``resampling_method="sinc_interp_hann"`` with ``lowpass_filter_width=16``."""
    g = math.gcd(orig, new)
    up, down = new // g, orig // g
    device = torch.device(device_str)

    # Cutoff at the lower of the two Nyquist rates, in units of the upsampled rate.
    cutoff = min(up, down)
    n = width * max(up, down)
    idx = torch.arange(-n, n + 1, device=device, dtype=torch.float64)
    t = idx / max(up, down)

    sinc = torch.where(t == 0, torch.ones_like(t), torch.sin(math.pi * t) / (math.pi * t))
    # Hann window over the filter support.
    window = 0.5 + 0.5 * torch.cos(math.pi * idx / n)
    window = torch.where(idx.abs() > n, torch.zeros_like(window), window)
    kernel = (sinc * window).to(torch.float32)
    kernel = kernel / kernel.sum() * up
    return kernel, up, down


def resample(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """Resample ``(..., time)`` from ``orig_freq`` to ``new_freq``.

    Uses torchaudio when available (it is faster and is the reference), else a
    polyphase windowed-sinc via ``conv1d``.
    """
    if orig_freq == new_freq:
        return waveform
    if HAVE_TORCHAUDIO:
        return _ta.functional.resample(waveform, orig_freq, new_freq)
    return resample_fallback(waveform, orig_freq, new_freq)


def resample_fallback(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """The torchaudio-free path, exposed separately so it can be tested even on
    machines where torchaudio works."""
    if orig_freq == new_freq:
        return waveform
    shape = waveform.shape
    x = waveform.reshape(-1, 1, shape[-1]).to(torch.float32)
    kernel, up, down = _resample_kernel(orig_freq, new_freq, str(waveform.device))

    # Upsample by zero-stuffing, low-pass, then decimate. conv_transpose1d with
    # stride=up does the stuffing and filtering in one pass.
    pad = (kernel.numel() - 1) // 2
    y = torch.nn.functional.conv_transpose1d(
        x, kernel.view(1, 1, -1), stride=up, padding=pad
    )
    y = y[..., ::down]

    n_out = int(math.ceil(shape[-1] * new_freq / orig_freq))
    if y.shape[-1] > n_out:
        y = y[..., :n_out]
    elif y.shape[-1] < n_out:
        y = torch.nn.functional.pad(y, (0, n_out - y.shape[-1]))
    return y.reshape(shape[:-1] + (n_out,)).to(waveform.dtype)

# ---------------------------------------------------------------------------
# Compatibility shims
# ---------------------------------------------------------------------------
class _ShimSpectrogram(torch.nn.Module):
    """Stands in for ``torchaudio.transforms.Spectrogram``.

    Exists so the module tree -- and therefore the state_dict key names -- match
    the real torchaudio. Checkpoints that were saved with a real MelSpectrogram
    as a submodule (the released Vocos weights are) carry
    ``spectrogram.window`` and ``mel_scale.fb``, and ``load_state_dict`` rejects
    them outright if those buffers are missing.
    """

    def __init__(self, n_fft: int, hop_length: int, power: float, center: bool):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.center = center
        # Persistent, matching torchaudio: released checkpoints (Vocos) carry
        # these keys, and load_state_dict rejects unexpected ones.
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            waveform.reshape(-1, waveform.shape[-1]).to(torch.float32),
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
            window=self.window.to(waveform.device, torch.float32),
            center=self.center, pad_mode="reflect", normalized=False,
            onesided=True, return_complex=True,
        )
        return spec.abs() ** self.power


class _ShimMelScale(torch.nn.Module):
    """Stands in for ``torchaudio.transforms.MelScale`` (buffer ``fb``)."""

    def __init__(self, n_mels: int, sample_rate: int, f_min: float, f_max: float,
                 n_stft: int):
        super().__init__()
        fb = _triangular_filterbank(n_stft, f_min, f_max, n_mels, sample_rate)
        self.register_buffer("fb", fb)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        fb = self.fb.to(spec.device, spec.dtype)
        return torch.matmul(spec.transpose(-1, -2), fb).transpose(-1, -2)


class _ShimMelSpectrogram(torch.nn.Module):
    """``torchaudio.transforms.MelSpectrogram`` stand-in, structurally faithful.

    Same submodule and buffer names as the real thing, so state dicts round-trip,
    and numerically identical to it (asserted in test_audio_compat.py).
    """

    def __init__(self, sample_rate=16000, n_fft=400, win_length=None, hop_length=None,
                 f_min=0.0, f_max=None, pad=0, n_mels=128, power=2.0, normalized=False,
                 center=True, pad_mode="reflect", onesided=None, norm=None,
                 mel_scale="htk", **kwargs):
        super().__init__()
        if norm is not None or mel_scale != "htk":
            raise NotImplementedError(
                "the DhVaani torchaudio shim implements only norm=None and "
                f"mel_scale='htk'; got norm={norm!r}, mel_scale={mel_scale!r}"
            )
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 2
        self.n_mels = n_mels
        self.power = power
        self.center = center
        self.f_min = f_min
        self.f_max = float(f_max) if f_max is not None else float(sample_rate // 2)

        self.spectrogram = _ShimSpectrogram(n_fft, self.hop_length, power, center)
        self.mel_scale = _ShimMelScale(
            n_mels, sample_rate, f_min, self.f_max, n_fft // 2 + 1
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_scale(self.spectrogram(waveform))
        return mel.reshape(waveform.shape[:-1] + mel.shape[-2:])


def _build_torchaudio_shim():
    import types

    mod = types.ModuleType("torchaudio")
    mod.__version__ = "0.0.0+dhvaani-shim"
    mod._DHVAANI_SHIM = True

    transforms = types.ModuleType("torchaudio.transforms")
    transforms.MelSpectrogram = _ShimMelSpectrogram

    class _Resample(torch.nn.Module):
        def __init__(self, orig_freq=16000, new_freq=16000, **kwargs):
            super().__init__()
            self.orig_freq, self.new_freq = int(orig_freq), int(new_freq)

        def forward(self, waveform):
            return resample(waveform, self.orig_freq, self.new_freq)

    transforms.Resample = _Resample

    functional = types.ModuleType("torchaudio.functional")
    functional.resample = resample
    functional._hz_to_mel = _hz_to_mel
    functional._mel_to_hz = _mel_to_hz

    # Vocos does `from torchaudio.functional.functional import _hz_to_mel, ...`,
    # so the nested module has to exist as well.
    functional_inner = types.ModuleType("torchaudio.functional.functional")
    functional_inner._hz_to_mel = _hz_to_mel
    functional_inner._mel_to_hz = _mel_to_hz
    functional_inner.resample = resample
    functional.functional = functional_inner

    def _unsupported(*_a, **_k):
        raise NotImplementedError(
            "torchaudio I/O is not available in this environment (its compiled "
            "extension does not match this torch build). DhVaani decodes audio "
            "with soundfile/pydub instead."
        )

    mod.load = _unsupported
    mod.save = _unsupported
    mod.transforms = transforms
    mod.functional = functional
    return mod, transforms, functional, functional_inner


def _build_encodec_shim():
    """Vocos imports `encodec` at module scope for its EncodecFeatures class.

    The mel-24kHz vocoder never constructs that class, so a name-only stub is
    enough -- and it fails loudly if anything actually tries to use it.
    """
    import types

    mod = types.ModuleType("encodec")
    mod._DHVAANI_SHIM = True

    class EncodecModel:
        @staticmethod
        def _unavailable(*_a, **_k):
            raise NotImplementedError(
                "EncodecFeatures is not available: DhVaani only uses the "
                "mel-conditioned Vocos vocoder."
            )

        encodec_model_24khz = _unavailable
        encodec_model_48khz = _unavailable

    mod.EncodecModel = EncodecModel
    return mod


def install_compat_shims() -> list[str]:
    """Register stand-ins for torchaudio/encodec when the real ones are unusable.

    Vocos imports both at module scope even though the mel-24kHz vocoder uses
    neither for decoding. On an NGC container the real torchaudio raises
    `OSError: undefined symbol` on import, which would take Vocos with it.

    Returns the module names that were shimmed (empty when nothing was needed).
    """
    import importlib
    import sys

    shimmed: list[str] = []

    if not HAVE_TORCHAUDIO:
        existing = sys.modules.get("torchaudio")
        if existing is None or not getattr(existing, "_DHVAANI_SHIM", False):
            mod, transforms, functional, functional_inner = _build_torchaudio_shim()
            sys.modules["torchaudio"] = mod
            sys.modules["torchaudio.transforms"] = transforms
            sys.modules["torchaudio.functional"] = functional
            sys.modules["torchaudio.functional.functional"] = functional_inner
            shimmed.append("torchaudio")

    try:
        importlib.import_module("encodec")
    except Exception:
        if not getattr(sys.modules.get("encodec"), "_DHVAANI_SHIM", False):
            sys.modules["encodec"] = _build_encodec_shim()
            shimmed.append("encodec")

    return shimmed
