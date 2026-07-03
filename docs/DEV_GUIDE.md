# FlowTTS (Fish Audio S2 Pro) — Dev Guide

A practical, copy-paste guide to running and using the gateway. For deeper design see
[../README.md](../README.md); for speed tuning see
[fish_s2pro_acceleration.md](fish_s2pro_acceleration.md).

**What it is:** a streaming text-to-speech WebSocket gateway for **Fish Audio S2 Pro**,
served by **sglang-omni**. You clone voices from short reference clips (addressed by
`voice_id`) and synthesize speech with realtime AR streaming.

```
Client ──WS(text, voice_id)──► gateway (CPU) ──HTTP──► fish-s2pro (sglang, GPU) ──► int16 PCM ──► Client
                                 └── clone voices via POST /voices (REST)
```

> ⚠ `fishaudio/s2-pro` is under the **Fish Audio Research License** (non-commercial);
> commercial use needs a separate license (`business@fish.audio`).

---

## 1. Prerequisites

- A Linux VM with an NVIDIA GPU (**H200** recommended), driver +
  **nvidia-container-toolkit** + Docker Compose v2.
- `HF_TOKEN` with access to `fishaudio/s2-pro` (gated repo), or a local weights path.

---

## 2. Start the stack (Docker)

```bash
cd ~/FlowTTS
export HF_TOKEN=hf_...
docker compose up -d fish-s2pro           # GPU backend (first run downloads weights)
docker compose ps                          # wait until fish-s2pro: healthy
docker compose up -d flowtts-gateway       # CPU gateway
docker compose logs -f flowtts-gateway
```

Ports: **8080** = WebSocket, **8764** = control API + `/metrics`, **8000** = backend.
Check readiness (200 once the gateway can reach the backend):
```bash
curl -s http://localhost:8764/ready
```

Behind nginx (as configured): WS at `ws://<vm>/flowtts/ws/<call_id>`, control API at
`http://<vm>/flowtts-ctrl/...`.

---

## 3. Clone a voice (REST — easiest)

`POST /voices` on the running gateway. **`ref_text` is mandatory** (no auto-transcription)
and must be the **exact transcript of the clip, in the clip's language/script**. No GPU
work — the gateway stores the clip + manifest; the backend encodes it on first use.

```bash
curl -i -X POST http://localhost:8764/voices \
  -F voice_id=saavi \
  -F preferred_lang=as \
  -F ref_text="<exact words spoken in the clip>" \
  -F audio=@sample_files/saavi_assamese.wav
```
- `-F audio=@path` reads from **wherever you run curl** — use `~/FlowTTS` or an absolute path.
- Success → `{"status":"ok","voice_id":"saavi", ...}`. Usable **immediately**, no restart.
- List voices: `curl -s http://localhost:8764/voices`
- Use `-i` (not `-sf`) while debugging — `-f` hides error bodies.

Tips: clean **10–30 s** mono clip; `ref_text` matches the audio exactly; `preferred_lang`
matches the clip. Reusing a `voice_id` across requests hits the backend's prefix cache.

---

## 4. Synthesize

### a) Jupyter notebook (quickest to hear)
Open [../sample_files/test_tts_ws.ipynb](../sample_files/test_tts_ws.ipynb), edit the params
(`HOST/PORT/PATH_PREFIX`, `VOICE`, `LANG`, `TEXT`), run — it prints TTFB/RTF, saves a WAV,
plays it inline.

### b) CLI test client
```bash
PYTHONPATH=. python3 -m flowtts.test.test_voice_ws \
  --host 127.0.0.1 --port 8080 \
  --voice saavi --lang as \
  --text "নমস্কাৰ, মই বাজাজ ফাইনান্সৰ পৰা অংকিতা কৈ আছোঁ।" \
  --out saavi_as.wav
```

### c) WebSocket contract (unchanged)
Connect to `ws://<host>:<port>/ws/<call_id>` and send:
```json
{ "type":"synthesize", "call_id":"c1", "text_id":"t1",
  "text":"…", "voice_id":"saavi", "language":"as", "speed":1.0, "streaming":true }
```
Server replies with repeated **binary** frames = JSON header **+ appended int16 PCM**:
```
{"type":"audio_chunk","chunk_index":0,"sample_rate":24000,"encoding":"pcm_int16","is_final":false,...}<PCM…>
```
then a final JSON `{"type":"audio_done","chunks":N,"rtf":...,"sample_rate":24000}`.
Concatenate the PCM from each `audio_chunk` → the full utterance. Cancel mid-stream:
`{"type":"cancel","text_id":"t1"}`. Errors: `{"type":"error","error":"…"}`.

---

## 5. Endpoints

| Method | Path (direct / behind nginx) | Purpose |
|---|---|---|
| WS   | `:8080/ws/{call_id}` · `/flowtts/ws/{call_id}` | synthesize (streaming) |
| POST | `:8764/voices` · `/flowtts-ctrl/voices` | clone a voice |
| GET  | `:8764/voices` | list voices |
| GET  | `:8764/ready` | readiness (gateway + backend reachable) |
| GET  | `:8764/metrics` | Prometheus metrics |
| GET  | `:8764/health`, `:8080/health` | liveness |
| POST | `:8000/v1/audio/speech` (backend) | raw sglang TTS (debug) |

---

## 6. Config & tuning (env vars — set in `docker-compose.yml`)

**Gateway:**

| Env var | Meaning | Default |
|---|---|---|
| `FLOWTTS_FISH__BACKEND_URL` | sglang backend URL | `http://fish-s2pro:8000` |
| `FLOWTTS_FISH__REFERENCE_MODE` | `local` / `base64` reference passing | `local` |
| `FLOWTTS_FISH__INITIAL_CODEC_CHUNK_FRAMES` | ↓ for lower TTFB | `null` |
| `FLOWTTS_OUTPUT__SAMPLE_RATE` | resample from 24k (16000/8000) | `24000` |
| `FLOWTTS_VOICES__DEFAULT_VOICE` | alias when `voice_id` omitted | `priya` |

**Backend:** `TTS_BATCH_MAX_ITEMS` (throughput), `MEM_FRACTION` (KV cache), `FISH_MODEL`,
`HF_TOKEN`.

After editing env: `docker compose up -d` (recreates). After code changes:
`docker compose build && docker compose up -d`.

Watch throughput: `nvidia-smi` (backend util/VRAM), `/metrics` (RTF), gateway logs
(`first_chunk ttft=…`, `stream_done rtf=…`).

---

## 7. Troubleshooting

| Symptom | Cause → Fix |
|---|---|
| **Gateway `/ready` 503** | Backend not up/reachable. `docker compose ps` (fish-s2pro healthy?); check `FLOWTTS_FISH__BACKEND_URL`; `curl :8000/health`. |
| **Backend never healthy** | First-run weight download (gated) — set `HF_TOKEN`; watch `docker compose logs fish-s2pro`; the healthcheck grace is 10 min. |
| **Bad cloned-voice audio** | `ref_text` must match the clip exactly, in its script/language; use a clean 10–30 s mono clip; set `preferred_lang`. |
| **Voice clone works but backend errors on synth** | Backend can't read the shared clip path → set `FLOWTTS_FISH__BACKEND_VOICES_DIR` or `FLOWTTS_FISH__REFERENCE_MODE=base64`. |
| **`POST /voices` returns nothing** | You used `-sf`; `-f` hides error bodies. Use `-i`. Check the file path at curl's CWD. |
| **Client hangs after streaming** | WS close handshake. Use the provided client (`close_timeout`, `ping_interval=None`, finalize on final chunk). |
| **High TTFB** | Reuse the voice (prefix cache); lower `INITIAL_CODEC_CHUNK_FRAMES`; measure from the VM to exclude network. |
| **`default_voice_missing` in logs** | The default alias has no reference yet → clone it (§3) or set `DEFAULT_VOICE`; the backend `default` voice is used meanwhile. |
| **nginx WS drops after ~60s idle** | Add `proxy_read_timeout 3600s;` to the WS location; `client_max_body_size 64m;` on the control-API location (audio uploads). |

---

## 8. File map

```
flowtts/server.py                 primary WS gateway + control API (clone endpoint)
flowtts/core/config.py            all settings (FLOWTTS_* env overrides)
flowtts/synthesis/
  fish_engine.py                  sglang backend client + create_voice() (no GPU)
  models.py                       synthesize() / synthesize_stream()
flowtts/voices/                   reference registry + manifest store + offline CLI
flowtts/test/test_voice_ws.py     CLI WebSocket test client
docker/                           gateway Dockerfile + fish_s2pro.Dockerfile
docker-compose.yml                two-service stack
sample_files/test_tts_ws.ipynb    notebook test cell
```

---

## 9. Everyday commands

```bash
# start / restart / logs
docker compose up -d
docker compose restart flowtts-gateway
docker compose logs -f flowtts-gateway
docker compose logs -f fish-s2pro

# clone + verify + hear
curl -i -X POST http://localhost:8764/voices -F voice_id=v1 -F preferred_lang=hi \
     -F ref_text="…" -F audio=@sample_files/vikram.wav
curl -s http://localhost:8764/voices
PYTHONPATH=. python3 -m flowtts.test.test_voice_ws --voice v1 --lang hi --text "…" --out v1.wav

# unit tests (no GPU)
python -m pytest flowtts/test/test_text_chunker.py flowtts/test/test_voice_store.py flowtts/test/test_pcm.py -q
```
