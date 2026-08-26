"""Equivalence tests for the torchaudio-free audio front end.

DhVaani's mel features feed the model directly, so the fallback extractor must
be numerically identical to `torchaudio.transforms.MelSpectrogram`, not merely
close. Resampling only shapes the output waveform, so a high-SNR match is
sufficient there.

These matter because torchaudio cannot be installed on NVIDIA NGC PyTorch
containers without replacing the container's custom torch build -- see
`model/audio_compat.py`.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from flowtts.dhvaani.model.audio_compat import (  # noqa: E402
    HAVE_TORCHAUDIO,
    VocosFbankCompat,
    mel_spectrogram,
    resample_fallback,
)

ta = pytest.importorskip("torchaudio") if HAVE_TORCHAUDIO else None
needs_ta = pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="torchaudio not installed")

SR, N_FFT, HOP, N_MELS = 24000, 1024, 256, 100


@needs_ta
@pytest.mark.parametrize("n_samples", [24000, 3 * 24000, 12345, 1000])
def test_mel_is_bit_identical_to_torchaudio(n_samples):
    """Exact equality, not approximate: this output IS the model's input."""
    torch.manual_seed(n_samples)
    x = torch.randn(1, n_samples)
    ref = ta.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        center=True, power=1,
    )(x)
    got = mel_spectrogram(x, SR, N_FFT, HOP, N_MELS, power=1.0, center=True)
    assert got.shape == ref.shape
    assert torch.equal(got, ref), f"max abs diff {(got - ref).abs().max()}"


@needs_ta
def test_mel_matches_for_real_signal_and_power_2():
    t = torch.arange(SR, dtype=torch.float32) / SR
    x = (torch.sin(2 * math.pi * 220 * t) + 0.3 * torch.sin(2 * math.pi * 3000 * t))[None]
    for power in (1.0, 2.0):
        ref = ta.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            center=True, power=power,
        )(x)
        got = mel_spectrogram(x, SR, N_FFT, HOP, N_MELS, power=power, center=True)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


@needs_ta
def test_vocos_fbank_compat_matches_vendored_contract():
    """Same shape and the same frame count as upstream's VocosFbank, whose
    frame-count fix-up feeds the duration model."""
    fb = VocosFbankCompat()
    for seconds in (0.5, 1.0, 3.0, 3.04):
        n = int(seconds * SR)
        x = torch.randn(1, n)
        mel = fb.extract(x, SR)
        expected_frames = int((n + (HOP // 2)) // HOP)
        assert mel.shape == (expected_frames, N_MELS), (seconds, mel.shape)
        assert torch.isfinite(mel).all()


def test_vocos_fbank_compat_accepts_numpy_and_returns_numpy():
    import numpy as np

    fb = VocosFbankCompat()
    out = fb.extract(np.random.randn(SR).astype(np.float32), SR)
    assert isinstance(out, np.ndarray)
    assert out.shape[1] == N_MELS


def test_vocos_fbank_compat_downmixes_stereo():
    fb = VocosFbankCompat()
    stereo = torch.randn(2, SR)
    mono = fb.extract(stereo, SR)
    assert mono.shape[1] == N_MELS


def test_vocos_fbank_rejects_wrong_sample_rate():
    fb = VocosFbankCompat()
    with pytest.raises(AssertionError):
        fb.extract(torch.randn(1, 16000), 16000)


@needs_ta
@pytest.mark.parametrize(
    "orig,new", [(44100, 24000), (48000, 24000), (24000, 16000), (24000, 8000),
                 (16000, 24000), (22050, 24000)]
)
def test_resample_fallback_matches_torchaudio(orig, new):
    """Length must match exactly; waveform must match to well below audibility."""
    t = torch.arange(orig, dtype=torch.float32) / orig
    x = (torch.sin(2 * math.pi * 220 * t) + 0.5 * torch.sin(2 * math.pi * 1000 * t))[None]
    ref = ta.functional.resample(x, orig, new)
    got = resample_fallback(x, orig, new)

    assert got.shape == ref.shape, (got.shape, ref.shape)
    # Ignore filter warm-up at the very edges.
    lo, hi = int(0.02 * ref.shape[-1]), int(0.98 * ref.shape[-1])
    r, g = ref[0, lo:hi], got[0, lo:hi]
    err = r - g
    snr = 10 * math.log10((r**2).mean().item() / max((err**2).mean().item(), 1e-20))
    assert snr > 55.0, f"{orig}->{new} SNR only {snr:.1f} dB"


def test_resample_identity_is_a_noop():
    x = torch.randn(1, 1000)
    assert resample_fallback(x, 24000, 24000) is x


def test_resample_preserves_shape_prefix():
    x = torch.randn(3, 2, 24000)
    out = resample_fallback(x, 24000, 8000)
    assert out.shape[:-1] == (3, 2)
    assert out.shape[-1] == 8000
