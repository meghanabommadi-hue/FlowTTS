"""API wiring tests, driven by a stub engine.

These exercise routing, request validation, streaming framing, the WAV header,
error mapping and auth without loading a model, so they run in CI on any box.
The engine itself is covered by `smoke.py` on a GPU.
"""

from __future__ import annotations

import struct

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from flowtts.dhvaani.api.app import create_app  # noqa: E402
from flowtts.dhvaani.config import DhvaaniSettings  # noqa: E402
from flowtts.dhvaani.types import (  # noqa: E402
    AudioChunk,
    RequestMetrics,
    VoiceNotFound,
)

SR = 24000


class StubVoices:
    def __init__(self):
        self._v = {
            "simran": {
                "voice_id": "simran", "name": "Simran", "description": "test",
                "language": "hi", "transcript": "नमस्ते", "duration_s": 2.0,
                "mel_frames": 187, "n_tokens": 60, "frames_per_token": 3.1,
                "sample_rate": SR, "created_at": 1.0, "source_filename": "s.wav",
                "checksum": "abc", "prompt_rms": 0.1,
            }
        }

    def list(self):
        return list(self._v.values())

    def exists(self, vid):
        return vid in self._v

    def get(self, vid):
        if vid not in self._v:
            raise VoiceNotFound(f"voice {vid!r} not found")
        meta = self._v[vid]
        obj = type("V", (), dict(meta))()
        obj.to_metadata = lambda m=meta: m
        return obj

    def delete(self, vid):
        if vid not in self._v:
            raise VoiceNotFound(f"voice {vid!r} not found")
        del self._v[vid]

    def stats(self):
        return {"voices": len(self._v)}


class StubEngine:
    ready = True

    def __init__(self, n_chunks: int = 3, samples: int = 2400):
        self.voices = StubVoices()
        self.n_chunks = n_chunks
        self.samples = samples
        self.cancelled = 0

    def _pcm(self):
        return (b"\x01\x00") * self.samples

    async def synthesize_stream(self, text, voice_id=None, language=None,
                                params=None, request_id=None, cancel_event=None):
        for i in range(self.n_chunks):
            final = i == self.n_chunks - 1
            meta = {}
            if final:
                meta["metrics"] = RequestMetrics(
                    request_id=request_id or "r", voice_id=voice_id or "simran",
                    language=language or "hi", n_chars=len(text), n_spans=self.n_chunks,
                    ttfb_ms=120.0, total_ms=450.0, audio_s=1.0,
                ).__dict__
            yield AudioChunk(
                request_id=request_id or "r", chunk_index=i, audio=self._pcm(),
                sample_rate=SR, encoding="pcm_int16", is_final=final, meta=meta,
            )

    async def synthesize(self, text, voice_id=None, language=None, params=None,
                         request_id=None, cancel_event=None):
        parts = []
        m = None
        async for c in self.synthesize_stream(text, voice_id, language, params, request_id):
            parts.append(c.audio)
            if c.meta.get("metrics"):
                m = RequestMetrics(**c.meta["metrics"])
        return b"".join(parts), m

    def stats(self):
        return {"ready": True, "backend": "stub", "scheduler": {}, "vram": {}}


@pytest.fixture
def client():
    s = DhvaaniSettings()
    s.server.api_keys = []
    app = create_app(StubEngine(), s)
    return TestClient(app)


# --- metadata --------------------------------------------------------------
def test_root_lists_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "POST /v1/audio/speech" in r.json()["endpoints"]


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "dhvaani-0.5"


def test_languages_lists_27(client):
    r = client.get("/v1/languages")
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 27
    assert {d["normalization"] for d in data} == {"full", "partial"}


def test_health_and_ready(client):
    assert client.get("/healthz").json()["ready"] is True
    assert client.get("/readyz").status_code == 200


def test_metrics_is_prometheus_text(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "dhvaani_" in r.text


# --- speech ----------------------------------------------------------------
def test_speech_wav_has_correct_header(client):
    r = client.post("/v1/audio/speech", json={"input": "नमस्ते", "voice": "simran"})
    assert r.status_code == 200
    body = r.content
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"
    data_size = struct.unpack("<I", body[40:44])[0]
    assert data_size == len(body) - 44          # exact, because length is known
    assert struct.unpack("<I", body[24:28])[0] == SR


def test_speech_pcm_is_raw(client):
    r = client.post("/v1/audio/speech",
                    json={"input": "hi", "voice": "simran", "response_format": "pcm"})
    assert r.status_code == 200
    assert r.content[:4] != b"RIFF"
    assert len(r.content) % 2 == 0
    assert "audio/L16" in r.headers["content-type"]


def test_speech_reports_timing_headers(client):
    r = client.post("/v1/audio/speech", json={"input": "hi", "voice": "simran"})
    assert float(r.headers["X-TTFB-Ms"]) == 120.0
    assert r.headers["X-Request-Id"]


def test_streaming_wav_uses_placeholder_sizes(client):
    """A streamed response cannot know its length, so RIFF/data carry 0xFFFFFFFF."""
    r = client.post("/v1/audio/speech",
                    json={"input": "hi", "voice": "simran", "stream": True})
    assert r.status_code == 200
    body = r.content
    assert body[:4] == b"RIFF"
    assert struct.unpack("<I", body[4:8])[0] == 0xFFFFFFFF
    assert struct.unpack("<I", body[40:44])[0] == 0xFFFFFFFF
    assert len(body) > 44


def test_sse_stream_format(client):
    r = client.post("/v1/audio/speech",
                    json={"input": "hi", "voice": "simran", "stream_format": "sse"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    text = r.text
    assert "speech.audio.delta" in text
    assert "speech.audio.done" in text
    assert text.rstrip().endswith("[DONE]")


def test_sse_via_accept_header(client):
    r = client.post("/v1/audio/speech", json={"input": "hi", "voice": "simran"},
                    headers={"Accept": "text/event-stream"})
    assert "text/event-stream" in r.headers["content-type"]


def test_empty_input_rejected(client):
    r = client.post("/v1/audio/speech", json={"input": "   ", "voice": "simran"})
    assert r.status_code == 422


def test_oversized_input_rejected(client):
    from flowtts.dhvaani.config import dhv_settings

    r = client.post("/v1/audio/speech",
                    json={"input": "x" * (dhv_settings.server.max_text_chars + 1)})
    assert r.status_code == 422


def test_bad_sample_rate_rejected(client):
    r = client.post("/v1/audio/speech",
                    json={"input": "hi", "voice": "simran", "sample_rate": 44100})
    assert r.status_code == 422


def test_streaming_compressed_format_rejected(client):
    r = client.post("/v1/audio/speech",
                    json={"input": "hi", "voice": "simran",
                          "response_format": "mp3", "stream": True})
    assert r.status_code == 400
    assert "cannot be streamed" in r.json()["detail"]


def test_speed_bounds(client):
    assert client.post("/v1/audio/speech",
                       json={"input": "hi", "speed": 0.1}).status_code == 422
    assert client.post("/v1/audio/speech",
                       json={"input": "hi", "speed": 9.0}).status_code == 422


# --- voices ----------------------------------------------------------------
def test_list_voices(client):
    r = client.get("/v1/voices")
    assert r.status_code == 200
    assert r.json()["data"][0]["voice_id"] == "simran"


def test_get_voice(client):
    assert client.get("/v1/voices/simran").json()["language"] == "hi"


def test_missing_voice_is_404(client):
    r = client.get("/v1/voices/nope")
    assert r.status_code == 404


def test_delete_voice(client):
    assert client.delete("/v1/voices/simran").json()["deleted"] is True
    assert client.get("/v1/voices/simran").status_code == 404


def test_create_voice_requires_file(client):
    r = client.post("/v1/voices", data={"voice_id": "x", "transcript": "t"})
    assert r.status_code == 422


def test_preview_returns_wav(client):
    r = client.post("/v1/voices/simran/preview")
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"
    assert r.headers["X-Voice-Id"] == "simran"


def test_preview_header_is_latin1_safe(client):
    """Preview sentences are Devanagari/Tamil/Perso-Arabic. HTTP header values
    must be latin-1 encodable, so the text is percent-encoded."""
    from urllib.parse import unquote

    r = client.post("/v1/voices/simran/preview")
    raw = r.headers["X-Text"]
    raw.encode("latin-1")                      # would raise before the fix
    assert unquote(raw)                        # round-trips to real text


# --- auth ------------------------------------------------------------------
def test_api_key_enforced_when_configured():
    from flowtts.dhvaani.config import dhv_settings

    original = list(dhv_settings.server.api_keys)
    dhv_settings.server.api_keys = ["secret"]
    try:
        c = TestClient(create_app(StubEngine(), dhv_settings))
        assert c.post("/v1/audio/speech", json={"input": "hi"}).status_code == 401
        ok = c.post("/v1/audio/speech", json={"input": "hi"},
                    headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200
        ok2 = c.post("/v1/audio/speech", json={"input": "hi"},
                     headers={"x-api-key": "secret"})
        assert ok2.status_code == 200
        # Reads stay open so health checks and dashboards do not need a key.
        assert c.get("/v1/voices").status_code == 200
    finally:
        dhv_settings.server.api_keys = original


def test_engine_not_ready_returns_503():
    eng = StubEngine()
    eng.ready = False
    c = TestClient(create_app(eng, DhvaaniSettings()))
    assert c.post("/v1/audio/speech", json={"input": "hi"}).status_code == 503
    assert c.get("/healthz").status_code == 503


# --- wire framing ----------------------------------------------------------
def test_split_frame_handles_braces_in_client_ids():
    """The audio frame is `json.dumps(header).encode() + pcm`. Splitting on the
    FIRST closing brace breaks when a client-supplied call_id contains one."""
    import json as _json

    from flowtts.dhvaani.test.loadtest import split_frame

    pcm = bytes(range(64))
    for call_id in ["normal", "has}brace", 'quote"and}brace', "nested{}stuff"]:
        header = {
            "type": "audio_chunk", "call_id": call_id, "text_id": "t1",
            "chunk_index": 0, "sample_rate": 24000, "encoding": "pcm_int16",
            "wav_bytes": len(pcm), "tokens": 32, "is_final": True, "cache_hit": False,
        }
        frame = _json.dumps(header).encode() + pcm
        got_header, got_audio = split_frame(frame)
        assert got_header["call_id"] == call_id
        assert got_audio == pcm


def test_split_frame_rejects_malformed():
    from flowtts.dhvaani.test.loadtest import split_frame

    with pytest.raises(ValueError):
        split_frame(b'{"unterminated": ')
