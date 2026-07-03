"""Unit tests for the reference-voice manifest store (stdlib only — no GPU/torch/numpy).

A Fish S2 Pro voice is a reference clip + transcript persisted as `<alias>.json`
beside `<alias>.wav`. This guards the manifest save→load round-trip and validation.

Run:  python -m flowtts.test.test_voice_store
  or: python -m pytest flowtts/test/test_voice_store.py -q
"""

import tempfile
from pathlib import Path

from flowtts.voices.store import SCHEMA_VERSION, load_voice, manifest_path, save_voice


def test_round_trip():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "priya.wav").write_bytes(b"RIFF....WAVE")
        out = save_voice(d, alias="priya", ref_text="नमस्ते, मैं प्रिया बोल रही हूँ।",
                         audio_file="priya.wav", language="hi")
        assert out == manifest_path(d, "priya")
        data = load_voice(out)

    assert data["alias"] == "priya"
    assert data["ref_text"] == "नमस्ते, मैं प्रिया बोल रही हूँ।"
    assert data["audio_file"] == "priya.wav"
    assert data["language"] == "hi"
    assert data["schema_version"] == SCHEMA_VERSION


def test_audio_file_stored_as_basename():
    """An absolute/nested audio path is normalized to a basename in the manifest."""
    with tempfile.TemporaryDirectory() as d:
        out = save_voice(d, alias="v", ref_text="hi", audio_file="/somewhere/else/v.wav")
        assert load_voice(out)["audio_file"] == "v.wav"


def test_language_optional():
    with tempfile.TemporaryDirectory() as d:
        out = save_voice(d, alias="v", ref_text="hi", audio_file="v.wav")
        assert load_voice(out)["language"] == ""


def test_rejects_missing_ref_text():
    with tempfile.TemporaryDirectory() as d:
        try:
            save_voice(d, alias="v", ref_text="  ", audio_file="v.wav")
        except ValueError:
            return
        raise AssertionError("expected ValueError for empty ref_text")


def test_rejects_newer_schema():
    import json
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "v.json"
        p.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1, "alias": "v",
                                 "ref_text": "hi", "audio_file": "v.wav"}), encoding="utf-8")
        try:
            load_voice(p)
        except ValueError:
            return
        raise AssertionError("expected ValueError for newer schema_version")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
