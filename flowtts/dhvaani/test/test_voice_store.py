"""Tests for the voice registry's persistence and cache behaviour.

Voice creation itself needs audio decoding and a real mel extractor, so these
tests exercise the parts that do not: id validation, the on-disk round trip,
LRU eviction and reload. `smoke.py` covers real cloning on a GPU box.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from flowtts.dhvaani.config import DhvaaniSettings  # noqa: E402
from flowtts.dhvaani.types import VoiceNotFound, VoicePrompt  # noqa: E402
from flowtts.dhvaani.voices.store import VoiceStore, validate_voice_id  # noqa: E402
from flowtts.dhvaani.types import InvalidReferenceAudio  # noqa: E402


class StubLoaded:
    device = torch.device("cpu")
    dtype = torch.float32

    @staticmethod
    def token_ids(text):
        return [ord(c) % 1000 for c in text]


def make_store(tmp: Path, cache_size: int = 8) -> VoiceStore:
    s = DhvaaniSettings()
    s.voice.store_dir = str(tmp)
    s.voice.gpu_cache_size = cache_size
    return VoiceStore(StubLoaded(), s)


def make_prompt(vid: str, frames: int = 40, tokens: int = 12) -> VoicePrompt:
    return VoicePrompt(
        voice_id=vid,
        mel=torch.arange(frames * 100, dtype=torch.float32).reshape(frames, 100),
        mel_frames=frames,
        token_ids=list(range(tokens)),
        prompt_rms=0.087,
        frames_per_token=frames / tokens,
        name=vid.title(),
        language="hi",
        transcript="नमस्ते",
        duration_s=frames / 93.75,
    )


@pytest.mark.parametrize("vid", ["a", "simran", "voice_1", "V-2", "a" * 64])
def test_valid_ids(vid):
    assert validate_voice_id(vid) == vid


@pytest.mark.parametrize("vid", ["", " ", "_leading", "-leading", "has space",
                                 "has/slash", "a" * 65, "hindi/../etc"])
def test_invalid_ids_rejected(vid):
    with pytest.raises(InvalidReferenceAudio):
        validate_voice_id(vid)


def test_persist_and_reload_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        store = make_store(tmp)
        p = make_prompt("simran")
        store._persist(p)

        # Exactly one npz and one json, and no stray temp files -- np.savez
        # appends ".npz" to names lacking it, which previously left the real
        # payload at "<id>.npz.tmp.npz" and os.replace moving the wrong path.
        names = sorted(f.name for f in tmp.iterdir())
        assert names == ["simran.json", "simran.npz"], names

        store2 = make_store(tmp)
        assert store2.reload() == 1
        got = store2.get("simran")
        assert got.voice_id == "simran"
        assert got.mel_frames == p.mel_frames
        assert got.token_ids == p.token_ids
        assert abs(got.frames_per_token - p.frames_per_token) < 1e-6
        assert abs(got.prompt_rms - p.prompt_rms) < 1e-4
        assert got.transcript == p.transcript
        # mel is stored as float16, so compare loosely.
        assert torch.allclose(got.mel.float(), p.mel.float(), rtol=1e-2, atol=1.0)


def test_metadata_is_valid_json_with_unicode():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        store = make_store(tmp)
        store._persist(make_prompt("v"))
        meta = json.loads((tmp / "v.json").read_text())
        assert meta["transcript"] == "नमस्ते"
        assert meta["voice_id"] == "v"


def test_missing_voice_raises():
    with tempfile.TemporaryDirectory() as d:
        store = make_store(Path(d))
        with pytest.raises(VoiceNotFound):
            store.get("nope")
        with pytest.raises(VoiceNotFound):
            store.delete("nope")
        with pytest.raises(VoiceNotFound):
            store.default()


def test_delete_removes_both_files():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        store = make_store(tmp)
        store._persist(make_prompt("gone"))
        store.reload()
        assert store.exists("gone")
        store.delete("gone")
        assert not store.exists("gone")
        assert list(tmp.iterdir()) == []


def test_lru_eviction_reloads_from_disk():
    """An evicted entry drops its GPU mel; get() must page it back in rather
    than handing out a VoicePrompt with mel=None."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        store = make_store(tmp, cache_size=2)
        for i in range(4):
            store._persist(make_prompt(f"v{i}"))
        store.reload()

        for i in range(4):
            got = store.get(f"v{i}")
            assert got.mel is not None
        assert len(store._cache) <= 2

        # v0 was evicted long ago; it must still come back intact.
        again = store.get("v0")
        assert again.mel is not None
        assert again.mel_frames == 40
        assert store.stats()["misses"] >= 1


def test_default_picks_oldest_when_unconfigured():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        store = make_store(tmp)
        a = make_prompt("aaa")
        a.created_at = 100.0
        b = make_prompt("bbb")
        b.created_at = 50.0
        store._persist(a)
        store._persist(b)
        store.reload()
        assert store.default().voice_id == "bbb"


def test_resolve_falls_back_to_default():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        store = make_store(tmp)
        store._persist(make_prompt("only"))
        store.reload()
        assert store.resolve(None).voice_id == "only"
        assert store.resolve("only").voice_id == "only"


def test_reload_skips_voice_with_missing_tensors():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        store = make_store(tmp)
        store._persist(make_prompt("half"))
        (tmp / "half.npz").unlink()
        assert make_store(tmp).reload() == 0
