"""Unit tests for the voice-clone npz format (NumPy only — no GPU/torch).

Guards the flagged serialization caveat: VoiceClonePrompt tokens must survive a
save→load round-trip losslessly, without pickle.

Run:  python -m flowtts.test.test_voice_npz
  or: python -m pytest flowtts/test/test_voice_npz.py -q
"""

import tempfile
from pathlib import Path

import numpy as np

from flowtts.voices.npz_io import SCHEMA_VERSION, load_voice_npz, save_voice_npz


def test_round_trip():
    tokens = np.random.randint(0, 1025, size=(8, 137), dtype=np.int64)  # OmniVoice: 8 codebooks
    with tempfile.TemporaryDirectory() as d:
        out = save_voice_npz(
            Path(d) / "priya",
            ref_audio_tokens=tokens,
            ref_text="नमस्ते, मैं प्रिया बोल रही हूँ।",
            ref_rms=0.1234,
            sample_rate=24000,
            frame_rate=25.0,
            alias="priya",
            language="hi",
        )
        assert out.suffix == ".npz"
        data = load_voice_npz(out)

    assert data["ref_audio_tokens"].shape == (8, 137)
    assert data["ref_audio_tokens"].dtype == np.int16
    assert np.array_equal(data["ref_audio_tokens"].astype(np.int64), tokens)  # lossless
    assert data["ref_text"] == "नमस्ते, मैं प्रिया बोल रही हूँ।"
    assert abs(data["ref_rms"] - 0.1234) < 1e-6
    assert data["sample_rate"] == 24000
    assert abs(data["frame_rate"] - 25.0) < 1e-6
    assert data["alias"] == "priya"
    assert data["language"] == "hi"
    assert data["schema_version"] == SCHEMA_VERSION


def test_rejects_non_2d():
    with tempfile.TemporaryDirectory() as d:
        try:
            save_voice_npz(Path(d) / "x", ref_audio_tokens=np.zeros(10, dtype=np.int64),
                           ref_text="t", ref_rms=0.1, sample_rate=24000, frame_rate=25.0, alias="x")
        except ValueError:
            return
        raise AssertionError("expected ValueError for 1-D tokens")


def test_rejects_out_of_int16_range():
    with tempfile.TemporaryDirectory() as d:
        big = np.full((8, 4), 40000, dtype=np.int64)  # > int16 max
        try:
            save_voice_npz(Path(d) / "x", ref_audio_tokens=big, ref_text="t", ref_rms=0.1,
                           sample_rate=24000, frame_rate=25.0, alias="x")
        except ValueError:
            return
        raise AssertionError("expected ValueError for out-of-int16-range tokens")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
