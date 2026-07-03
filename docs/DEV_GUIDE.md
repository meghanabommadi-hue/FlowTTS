# OmniVoice TTS — Dev Guide

A practical, copy-paste guide to running and using the server. For deeper design see
[../README.md](../README.md); for speed tuning see [omnivoice_acceleration.md](omnivoice_acceleration.md).

**What it is:** a streaming text-to-speech WebSocket server built on **k2-fsa/OmniVoice**
(multilingual, 24 kHz). You clone voices from short reference clips (addressed by
`voice_id`) and synthesize speech with realtime streaming + dynamic batching.

```
Client ──WS(text, voice_id)──► server (OmniVoice, GPU) ──► streamed int16 PCM ──► Client
                └── clone voices via POST /voices (REST)
```

---

## 1. Prerequisites

- A Linux VM with an NVIDIA GPU (compute capability ≥ 8.0 — RTX 6000 Ada / H200 / …).
- NVIDIA driver + **nvidia-container-toolkit** + Docker Compose v2.
- OmniVoice weights on disk at `model_dir/base/` (config.json, model.safetensors,
  `audio_tokenizer/`, tokenizer files). If absent, the server downloads from HuggingFace.

---

## 2. Start the server (Docker)

```bash
cd ~/FlowTTS
docker compose build omnivoice-tts          # after any code change
docker compose up -d                         # serve
docker compose logs -f omnivoice-tts         # watch startup (first run compiles/warms up)
```

Ports (published to the host): **8080** = WebSocket, **8764** = control API + `/metrics`.
Check it's ready:
```bash
curl -s http://localhost:8764/ready          # {"ready": true}
```

Behind nginx (as configured): WS at `ws://<vm>/omnivoice-tts/ws/<call_id>`, control API at
`http://<vm>/omnivoice-ctrl/...`.

---

## 3. Clone a voice (REST — easiest)

`POST /voices` on the running server. **`ref_text` is mandatory** (no auto-transcription)
and must be the **exact transcript of the clip, in the clip's language/script**.

```bash
curl -i -X POST http://localhost:8764/voices \
  -F voice_id=saavi \
  -F preferred_lang=as \
  -F ref_text="<exact words spoken in the clip>" \
  -F audio=@voices/saavi-assamese.mp3
```
- `-F audio=@path` reads the file from **wherever you run curl** — run from `~/FlowTTS`
  (or use an absolute path). The file does **not** need to be pre-placed in the container.
- Success → `{"status":"ok","voice_id":"saavi","tokens":[8,N],...}`. Usable **immediately**, no restart.
- List voices: `curl -s http://localhost:8764/voices`
- Use `-i` (not `-sf`) while debugging — `-f` hides error bodies, making it "return nothing".

Tips for a good clone: clean **3–10 s** mono clip; `ref_text` matches the audio exactly;
`preferred_lang` matches the clip (`as`=Assamese, `bn`=Bengali, `hi`=Hindi, `en`=English).

---

## 4. Synthesize

### a) Jupyter notebook (quickest to hear)
Open [../sample_files/test_tts_ws.ipynb](../sample_files/test_tts_ws.ipynb), edit the params
in the last cell (`HOST/PORT/PATH_PREFIX`, `VOICE`, `LANG`, `TEXT`), run — it prints
TTFB/RTF, saves a WAV, and plays it inline.

### b) CLI test client
```bash
PYTHONPATH=. python3 -m flowtts.test.test_voice_ws \
  --host 172.16.1.4 --port 80 --path-prefix /omnivoice-tts \
  --voice saavi --lang as \
  --text "নমস্কাৰ, মই বাজাজ ফাইনান্সৰ পৰা অংকিতা কৈ আছোঁ।" \
  --out saavi_as.wav
# direct (no nginx): --host 127.0.0.1 --port 8080  (drop --path-prefix)
```

### c) WebSocket contract (for your own client)
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
Concatenate the PCM from each `audio_chunk` (same `sample_rate`) → the full utterance.
Cancel mid-stream: `{"type":"cancel","text_id":"t1"}`. Errors: `{"type":"error","error":"…"}`.

---

## 5. Endpoints

| Method | Path (direct / behind nginx) | Purpose |
|---|---|---|
| WS   | `:8080/ws/{call_id}` · `/omnivoice-tts/ws/{call_id}` | synthesize (streaming) |
| POST | `:8764/voices` · `/omnivoice-ctrl/voices` | clone a voice |
| GET  | `:8764/voices` · `/omnivoice-ctrl/voices` | list voices |
| GET  | `:8764/ready` · `/omnivoice-ctrl/ready` | readiness |
| GET  | `:8764/metrics` · `/omnivoice-ctrl/metrics` | Prometheus metrics |
| GET  | `:8764/health`, `:8080/health` | liveness |

---

## 6. Config & tuning (env vars — set in `docker-compose.yml`)

| Env var | Meaning | Default |
|---|---|---|
| `FLOWTTS_OMNIVOICE__MODEL_PATH` | local weights dir (used if it exists) | `model_dir/base` |
| `FLOWTTS_OMNIVOICE__NUM_STEP` | diffusion steps (main latency knob) | `16` (try 10–12) |
| `FLOWTTS_OMNIVOICE__MAX_BATCH` | dynamic batch size (throughput) | `48` (watch VRAM) |
| `FLOWTTS_OMNIVOICE__BATCH_TIMEOUT_MS` | batch fill window | `20` |
| `FLOWTTS_OMNIVOICE__COMPILE_MODEL` | torch.compile (big per-step speedup) | `true` |
| `FLOWTTS_OMNIVOICE__DTYPE` | `bfloat16` / `float16` | `bfloat16` |
| `FLOWTTS_OUTPUT__SAMPLE_RATE` | resample from 24k (e.g. 16000/8000) | `24000` |
| `FLOWTTS_VOICES__DEFAULT_VOICE` | alias when `voice_id` omitted | `priya` |
| `FLOWTTS_STREAMING__FIRST_CHUNK_MAX_CHARS` | size of low-TTFB first chunk | `60` |

After editing env: `docker compose up -d` (recreates, no rebuild). After code changes:
`docker compose build omnivoice-tts && docker compose up -d`.

Watch throughput: `docker compose logs omnivoice-tts | grep omnivoice_batch` (want `n` near
`MAX_BATCH`, low `per_item_ms`). Watch VRAM/util: `nvidia-smi`.

---

## 7. Troubleshooting (the common ones)

| Symptom | Cause → Fix |
|---|---|
| **Rubbish audio for non-Hindi/English text** | Older image stripped non-Devanagari scripts. **Rebuild** — normalization is now multilingual (control-char strip only). Also match `language` to the text. |
| **HF download crawls (~250 kB/s)** | Anonymous rate limit. Use local weights at `model_dir/base` (auto-detected → no download), or pass `HF_TOKEN`. |
| **`POST /voices` returns nothing** | You used `-sf`; `-f` hides error bodies. Use `-i`. Check the file path exists at curl's CWD, and `docker compose logs`. |
| **Clone/synth CUDA OOM** | Shared/full GPU (`nvidia-smi`). Prefer the **REST** clone (reuses the loaded model — no 2nd load). Or lower `MAX_BATCH`/`NUM_STEP`; stop other GPU processes. |
| **Client hangs after streaming** | WebSocket close handshake / keepalive. Use the provided client (`close_timeout`, `ping_interval=None`, finalize-on-final-chunk). |
| **High TTFB on short text** | Short text = 1 chunk, so TTFB ≈ full gen. Lower `NUM_STEP` and `FIRST_CHUNK_MAX_CHARS`; enable `COMPILE_MODEL`; measure from the VM to exclude network. |
| **`default_voice_missing` in logs** | The default alias has no npz yet → server uses OmniVoice auto voice. Clone it (§3) or set `DEFAULT_VOICE`. |
| **nginx WS drops after ~60s idle** | Add `proxy_read_timeout 3600s;` to the WS location; `client_max_body_size 64m;` on the control-API location (audio uploads). |

---

## 8. File map

```
flowtts/server.py                 primary WS server + control API (clone endpoint)
flowtts/core/config.py            all settings (FLOWTTS_* env overrides)
flowtts/synthesis/
  omnivoice_engine.py             model load + dynamic batcher + create_voice()
  models.py                       synthesize() / synthesize_stream()
  text_chunker.py                 multilingual normalize + streaming split
flowtts/voices/                   npz registry + REST-backed clone + offline CLI
flowtts/test/test_voice_ws.py     CLI WebSocket test client
docker/ , docker-compose.yml      containerized run
sample_files/test_tts_ws.ipynb    notebook test cell
```

---

## 9. Everyday commands

```bash
# start / restart / logs
docker compose up -d
docker compose restart omnivoice-tts
docker compose logs -f omnivoice-tts

# clone + verify + hear
curl -i -X POST http://localhost:8764/voices -F voice_id=v1 -F preferred_lang=hi \
     -F ref_text="…" -F audio=@voices/clip.wav
curl -s http://localhost:8764/voices
PYTHONPATH=. python3 -m flowtts.test.test_voice_ws --voice v1 --lang hi --text "…" --out v1.wav

# unit tests (no GPU)
python -m pytest flowtts/test/test_text_chunker.py flowtts/test/test_voice_npz.py flowtts/test/test_pcm.py -q
```
