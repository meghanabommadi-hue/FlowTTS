"""Transport-level tests for the FastAPI app, against a fake GPU engine.

Proves the request path end to end — validation, normalization, chunking,
stitching, encoding, the WebSocket frame format and the OpenAI-compatible
shim — without CUDA, so it runs in CI and on a laptop. The fake engine returns
audio shaped like OmniVoice's (silence-padded at both edges), which is what the
stitcher has to cope with.
"""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from flowtts.api import service as service_module  # noqa: E402
from flowtts.api.http_app import create_app  # noqa: E402
from flowtts.synthesis.chunker import estimate_duration  # noqa: E402
from flowtts.synthesis.models import OmniVoiceSynthesizer  # noqa: E402

SR = 24000


class _FakeRegistry:
    def describe(self):
        return [{"voice_id": "priya", "language": "hi", "reference_frames": 120,
                 "ref_text": "नमस्ते", "sample_rate": SR, "is_default": True}]

    def aliases(self):
        return ["priya"]

    def has(self, alias):
        return alias == "priya"


class _FakeEngine:
    """Returns plausible audio for the requested text, and records every call."""

    sampling_rate = SR
    frame_rate = 25.0

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.engine_info = {"backbone": {"backend": "fake"}, "sampling_rate": SR}
        self.registry = _FakeRegistry()

    async def initialize(self):
        return None

    async def synthesize(self, text, **kwargs):
        self.calls.append((text, kwargs))
        await asyncio.sleep(0)
        samples = max(1, int(estimate_duration(text) * SR))
        t = np.arange(samples) / SR
        body = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        pad = np.zeros(int(0.1 * SR), dtype=np.float32)   # OmniVoice's edge padding
        return np.concatenate([pad, body, pad])

    async def create_prompt(self, path, ref_text):
        return {"fake_prompt": ref_text}

    async def create_voice(self, voice_id, path, ref_text, language=None):
        return {"voice_id": voice_id, "tokens": [8, 120], "reference_frames": 120,
                "reference_seconds": 4.8, "ref_rms": 0.05, "ref_text": ref_text,
                "language": language, "sample_rate": SR, "npz": f"/tmp/{voice_id}.npz"}

    def delete_voice(self, voice_id):
        return voice_id == "priya"

    def snapshot(self):
        return {"engine": self.engine_info, "requests": len(self.calls)}


@pytest.fixture()
def client():
    synthesizer = OmniVoiceSynthesizer()
    synthesizer.engine = _FakeEngine()

    svc = service_module.service
    svc.synthesizer = synthesizer
    svc.ready = True
    svc.restarting = False
    svc.oom_recovery = False
    svc._semaphore = asyncio.Semaphore(64)
    svc._cache_dir = None

    app = create_app(load_on_startup=False)   # never load a real model here
    with TestClient(app) as test_client:
        test_client.engine = synthesizer.engine
        yield test_client

    svc.ready = False
    svc.synthesizer = None


HINDI = "आपका बकाया ₹2,500 है, कृपया 15/04/2026 तक payment करें। धन्यवाद।"


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def test_tts_returns_a_wav(client):
    r = client.post("/v1/tts", json={"text": HINDI, "language": "hi", "voice_id": "priya"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF" and r.content[8:12] == b"WAVE"
    assert float(r.headers["x-duration-seconds"]) > 1.0


def test_tts_json_response_when_requested(client):
    r = client.post("/v1/tts", json={"text": "Hello there."},
                    headers={"accept": "application/json"})
    assert r.status_code == 200
    body = r.json()
    assert base64.b64decode(body["audio_base64"])[:4] == b"RIFF"
    assert body["metadata"]["sample_rate"] == SR


def test_requested_sample_rate_is_honoured(client):
    r = client.post("/v1/tts", json={"text": "Hello there.", "sample_rate": 8000,
                                     "format": "pcm"},
                    headers={"accept": "application/json"})
    meta = r.json()["metadata"]
    assert meta["sample_rate"] == 8000 and meta["format"] == "pcm"


def test_generation_overrides_reach_the_engine(client):
    client.post("/v1/tts", json={"text": "Hello there.",
                                 "generation": {"num_step": 3, "guidance_scale": 0.0,
                                                "class_temperature": 0.7}})
    _, kwargs = client.engine.calls[-1]
    config = kwargs["params"].config
    assert config["num_step"] == 3
    assert config["guidance_scale"] == 0.0
    assert config["class_temperature"] == 0.7


def test_streaming_chunks_disable_the_models_edge_padding(client):
    """Otherwise every seam gains ~200 ms of silence the stitcher must remove."""
    with client.stream("POST", "/v1/tts/stream", json={"text": HINDI, "language": "hi"}) as s:
        for _ in s.iter_bytes():
            pass
    _, kwargs = client.engine.calls[-1]
    assert kwargs["params"].config["pad_duration"] == 0.0
    assert kwargs["params"].config["fade_duration"] == 0.0


def test_stream_returns_wav_and_audio(client):
    with client.stream("POST", "/v1/tts/stream",
                       json={"text": HINDI, "language": "hi"}) as s:
        assert s.status_code == 200
        assert s.headers["x-accel-buffering"] == "no"
        body = b"".join(s.iter_bytes())
    assert body[:4] == b"RIFF"
    assert len(body) > 44


def test_stream_pcm_has_no_header(client):
    with client.stream("POST", "/v1/tts/stream",
                       json={"text": "Hello there.", "format": "pcm"}) as s:
        body = b"".join(s.iter_bytes())
    assert body[:4] != b"RIFF"
    assert len(body) % 2 == 0


def test_long_text_is_chunked_into_several_generate_calls(client):
    before = len(client.engine.calls)
    client.post("/v1/tts", json={"text": HINDI, "language": "hi"})
    assert len(client.engine.calls) - before > 1


def test_short_text_is_a_single_generate_call(client):
    before = len(client.engine.calls)
    client.post("/v1/tts", json={"text": "Hello."})
    assert len(client.engine.calls) - before == 1


def test_control_tags_reach_the_engine_intact(client):
    client.post("/v1/tts", json={"text": "[laughter] You got me.", "language": "en"})
    assert any("[laughter]" in text for text, _ in client.engine.calls)


def test_nbsp_never_reaches_the_engine(client):
    """The chunker's non-breaking hints are an internal detail."""
    client.post("/v1/tts", json={"text": HINDI, "language": "hi"})
    assert all(" " not in text for text, _ in client.engine.calls)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload", [
    {},                                        # no text
    {"text": ""},                              # empty
    {"text": "   "},                           # blank
    {"text": "hi", "speed": 0.0},              # out of range
    {"text": "hi", "generation": {"num_step": 0}},
    {"text": "hi", "generation": {"guidance_scale": -1}},
    {"text": "hi", "format": "flac"},          # unsupported
])
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/v1/tts", json=payload).status_code == 422


def test_reference_audio_without_transcript_is_rejected(client):
    r = client.post("/v1/tts", json={
        "text": "hello",
        "reference_audio": base64.b64encode(b"RIFFfake").decode(),
    })
    assert r.status_code == 400
    assert "reference_text" in r.json()["detail"]


def test_bad_base64_reference_is_rejected(client):
    r = client.post("/v1/tts", json={"text": "hello", "reference_audio": "!!!not base64!!!",
                                     "reference_text": "hi"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# OpenAI compatibility
# ---------------------------------------------------------------------------
def test_openai_speech_endpoint(client):
    r = client.post("/v1/audio/speech", json={
        "model": "omnivoice", "input": "Hello there.", "voice": "priya",
        "response_format": "wav",
    })
    assert r.status_code == 200 and r.content[:4] == b"RIFF"


def test_openai_streaming(client):
    with client.stream("POST", "/v1/audio/speech",
                       json={"model": "omnivoice", "input": HINDI, "stream": True}) as s:
        assert s.status_code == 200
        assert len(b"".join(s.iter_bytes())) > 44


def test_openai_extensions_are_accepted(client):
    r = client.post("/v1/audio/speech", json={
        "model": "omnivoice", "input": "Hello.", "language": "en",
        "generation": {"num_step": 6},
    })
    assert r.status_code == 200
    assert client.engine.calls[-1][1]["params"].config["num_step"] == 6


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------
def test_list_voices(client):
    body = client.get("/v1/voices").json()
    assert body["voices"][0]["voice_id"] == "priya"


def test_clone_voice_multipart(client):
    r = client.post("/v1/voices",
                    files={"audio": ("ref.wav", b"RIFF" + b"\0" * 100, "audio/wav")},
                    data={"voice_id": "newvoice", "reference_text": "नमस्ते",
                          "language": "hi"})
    assert r.status_code == 200, r.text
    assert r.json()["voice_id"] == "newvoice"


def test_clone_voice_json(client):
    r = client.post("/v1/voices", json={
        "voice_id": "jsonvoice", "reference_text": "hello",
        "audio_base64": base64.b64encode(b"RIFF" + b"\0" * 100).decode(),
    })
    assert r.status_code == 200 and r.json()["voice_id"] == "jsonvoice"


def test_clone_rejects_a_duplicate_without_overwrite(client):
    r = client.post("/v1/voices", json={
        "voice_id": "priya", "reference_text": "hello",
        "audio_base64": base64.b64encode(b"RIFF").decode(),
    })
    assert r.status_code == 409


def test_clone_requires_a_transcript(client):
    r = client.post("/v1/voices", json={"voice_id": "x",
                                        "audio_base64": base64.b64encode(b"RIFF").decode()})
    assert r.status_code == 422


def test_delete_voice(client):
    assert client.delete("/v1/voices/priya").status_code == 200
    assert client.delete("/v1/voices/nope").status_code == 404


def test_voice_preview_returns_audio_without_saving(client):
    r = client.post("/v1/voices/preview",
                    files={"audio": ("ref.wav", b"RIFF" + b"\0" * 100, "audio/wav")},
                    data={"reference_text": "नमस्ते", "text": "आप कैसे हैं?",
                          "language": "hi"})
    assert r.status_code == 200 and r.content[:4] == b"RIFF"


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------
def test_languages_lists_the_indic_set(client):
    body = client.get("/v1/languages").json()
    codes = {entry["code"] for entry in body["languages"]}
    # All 22 scheduled languages of India must be present.
    for code in ("hi", "bn", "mr", "te", "ta", "gu", "ur", "kn", "or", "ml", "pa",
                 "as", "mai", "sat", "ks", "ne", "sd", "kok", "doi", "mni", "brx", "sa"):
        assert code in codes, f"{code} missing"


def test_normalize_endpoint_shows_chunks(client):
    body = client.post("/v1/normalize", json={"text": HINDI, "language": "hi"}).json()
    assert not any(ch.isdigit() for ch in body["normalized"])
    assert " " not in body["normalized"]
    assert len(body["chunks"]) >= 1
    assert body["chunks"][0]["estimated_seconds"] > 0


def test_normalize_requires_text(client):
    assert client.post("/v1/normalize", json={}).status_code == 400


def test_health_and_ready(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").json()["ready"] is True


def test_stats(client):
    body = client.get("/v1/stats").json()
    assert body["ready"] is True
    assert "counters" in body and "ttfb_ms" in body


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
def _drain_ws(ws) -> tuple[list[dict], dict]:
    headers, audio_bytes = [], 0
    while True:
        message = ws.receive()
        payload = message.get("bytes")
        if payload:
            split = payload.index(b"}") + 1
            headers.append(json.loads(payload[:split]))
            audio_bytes += len(payload) - split
            continue
        done = json.loads(message["text"])
        done["_audio_bytes"] = audio_bytes
        return headers, done


def test_ws_streams_audio_frames(client):
    with client.websocket_connect("/ws/call-1") as ws:
        ws.send_text(json.dumps({"type": "synthesize", "text": HINDI,
                                 "language": "hi", "voice_id": "priya", "text_id": "u1"}))
        headers, done = _drain_ws(ws)

    assert done["type"] == "audio_done"
    assert done["chunks"] == len(headers) > 0
    assert done["llm_ttft_ms"] is not None
    assert headers[0]["chunk_index"] == 0
    assert headers[0]["encoding"] == "pcm_int16"
    assert headers[-1]["is_final"] is True
    assert done["_audio_bytes"] == done["total_wav_bytes"]


def test_ws_reports_validation_errors_without_closing(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "synthesize", "text": ""}))
        error = json.loads(ws.receive()["text"])
        assert error["type"] == "error"

        ws.send_text(json.dumps({"type": "synthesize", "text": "Hello."}))
        _, done = _drain_ws(ws)
        assert done["type"] == "audio_done"


def test_ws_rejects_malformed_json(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("{not json")
        assert json.loads(ws.receive()["text"])["type"] == "error"


def test_ws_ping(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        assert json.loads(ws.receive()["text"])["type"] == "pong"


def test_ws_accepts_generation_parameters(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "synthesize", "text": "Hello.",
                                 "generation": {"num_step": 5}}))
        _drain_ws(ws)
    assert client.engine.calls[-1][1]["params"].config["num_step"] == 5


def test_ws_honours_the_requested_sample_rate(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "synthesize", "text": "Hello.",
                                 "sample_rate": 8000}))
        headers, done = _drain_ws(ws)
    assert done["sample_rate"] == 8000
    assert all(h["sample_rate"] == 8000 for h in headers)


# ---------------------------------------------------------------------------
# Degenerate output — a 200 with an empty body is the worst failure mode
# ---------------------------------------------------------------------------
def test_silent_generation_becomes_an_error_not_an_empty_200(client):
    """A stream that yields no audio must fail loudly, before the headers go out."""
    async def _silent(text, **kwargs):
        return np.zeros(24000, dtype=np.float32)

    client.engine.synthesize = _silent
    with client.stream("POST", "/v1/tts/stream", json={"text": "Hello there."}) as s:
        assert s.status_code >= 400, "silent synthesis returned a success status"


def test_silent_generation_fails_the_whole_utterance_route_too(client):
    async def _silent(text, **kwargs):
        return np.zeros(0, dtype=np.float32)

    client.engine.synthesize = _silent
    r = client.post("/v1/tts", json={"text": "Hello there."})
    assert r.status_code >= 400 or len(r.content) > 44
