"""Unit tests for PCM encoding + resampling (NumPy only — no GPU/torch).

Run:  python -m flowtts.test.test_pcm
  or: python -m pytest flowtts/test/test_pcm.py -q
"""

import numpy as np

from flowtts.decoder.decoder import pcm_to_int16_bytes
from flowtts.processing.audio_processing import resample_audio


def test_pcm_int16_clip_and_dtype():
    wav = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5], dtype=np.float32)  # includes out-of-range
    raw = pcm_to_int16_bytes(wav)
    arr = np.frombuffer(raw, dtype="<i2")
    assert len(arr) == 6
    assert arr[0] == 0
    assert arr[1] == 32767          # +1.0
    assert arr[2] == -32767         # -1.0
    assert arr[3] == 32767          # +2.0 clipped
    assert arr[4] == -32767         # -2.0 clipped
    assert abs(int(arr[5]) - int(0.5 * 32767)) <= 1


def test_pcm_empty():
    assert pcm_to_int16_bytes(np.zeros(0, dtype=np.float32)) == b""


def test_resample_length_math():
    sr_in, sr_out = 24000, 16000
    wav = np.random.randn(24000).astype(np.float32)  # 1.0 s
    out = resample_audio(wav, sr_in, sr_out)
    assert abs(len(out) - 16000) <= 1                 # ~1.0 s at 16 kHz
    # identity resample is a no-op
    assert resample_audio(wav, sr_in, sr_in) is wav


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
