# FlowTTS — Fish Audio S2 Pro streaming TTS gateway

A production-grade, low-latency **text-to-speech WebSocket gateway** for
[**Fish Audio S2 Pro**](https://huggingface.co/fishaudio/s2-pro), served by
[**sglang-omni**](https://github.com/sgl-project/sglang-omni) for high throughput and
ultra-low streaming latency. Built for voice-bot / IVR workloads (Hindi + English, 80+
languages), with **realtime AR streaming**, **reusable voice clones**, and a stable WS
protocol.

> **S2 Pro** is a Dual-AR TTS model (Qwen3-4B slow-AR + 400M fast-AR) with an
> EVA-GAN / RVQ codec (10 codebooks, ~21 Hz, 24 kHz). Because the slow-AR is a
> standard decoder-only LLM, sglang gives it **continuous batching, paged KV cache,
> CUDA-graph replay, and RadixAttention prefix caching** — inheriting all LLM-native
> optimizations. Published single-H200 figures: **RTF ≈ 0.195, TTFA < 100 ms,
> 3000+ acoustic tok/s**.
>
> ⚠ **License:** `fishaudio/s2-pro` is under the **Fish Audio Research License**
> (non-commercial). **Commercial use requires a separate license** from Fish Audio
> (`business@fish.audio`). `FISH_MODEL` is configurable so you can point at a licensed
> or self-hosted checkpoint.

---

## Architecture

Two services on one H200:

```
Client ─WS(text, voice_id)─▶ flowtts-gateway  (CPU-only, no model)
                               │ normalize + WAV-cache lookup ─hit─▶ send cached PCM ▶ done
                               │ FishSpeechEngine.synthesize_stream(...)
                               │   → POST http://fish-s2pro:8000/v1/audio/speech
                               │     {input, references:[{audio_path,text}], language,
                               │      speed, stream:true, response_format:"pcm"}
                               ▼
                            fish-s2pro : sgl-omni serve fishaudio/s2-pro   (GPU)
                               │ Dual-AR + EVA-GAN codec, continuous batch, prefix cache
                               │ streams 16-bit mono PCM @ 24 kHz
                               ▼
                            gateway: PCM → float32 → (resample) → int16 → WS audio_chunk
Client ◀── audio_chunk (JSON header + PCM bytes) … audio_done ──
```

All GPU work lives in the **sglang backend**; the **gateway** is a lightweight async
proxy that keeps FlowTTS's WebSocket protocol, voice registry, WAV cache, metrics, and
control API. A secondary Redis-backed multi-process path (`main.py` + `worker.py`) still
exists for cross-machine scaling.

---

## Features

- 🎙️ **Realtime AR streaming** — the backend streams contiguous 16-bit PCM token-by-token,
  forwarded as int16 frames, targeting **TTFB < 200 ms** (backend TTFA ~100 ms).
- ⚡ **Throughput from sglang** — continuous batching + paged KV cache + CUDA graphs; no
  gateway-side batching needed.
- 🗣️ **Voice cloning by alias** — a voice is a reference clip + transcript; clone live via
  `POST /voices` (no restart) or the offline CLI. Reused voices hit the **prefix cache**.
- 🚀 **WAV cache** — a SHA-256 cache that bypasses the backend on repeated prompts.
- 🔌 **Stable protocol** — the same WebSocket in/out contract (binary PCM frames).
- 📊 **Ops built-in** — `/health`, `/ready`, `/metrics` (Prometheus), on-demand port
  binding, idle-connection reaping.
- 🐳 **Containerized** — one `docker compose` stack (backend + gateway).

---

## Requirements

- NVIDIA GPU, compute capability ≥ 9.0 recommended (**H200**); the backend needs enough
  VRAM for a ~10 GB model + KV cache.
- Linux, NVIDIA driver + **nvidia-container-toolkit** + Docker Compose v2.
- Gateway is CPU-only (Python 3.12).

---

## Quick start (Docker — recommended)

```bash
git clone <your-repo-url> FlowTTS && cd FlowTTS

# 1) Start the GPU backend (first run downloads ~10GB weights; wait until healthy).
export HF_TOKEN=hf_...                # needed to pull the gated weights
docker compose up -d fish-s2pro
docker compose ps                     # wait for fish-s2pro: healthy

# 2) Start the gateway (:8080 WebSocket, :8764 control API / Prometheus).
docker compose up -d flowtts-gateway
docker compose logs -f flowtts-gateway

# 3) Smoke test.
docker compose exec flowtts-gateway python -m flowtts.test.test_pipeline \
    --ctrl-port 8764 --requests 5 --streaming
```

Full container guide + tuning: [`docs/fish_s2pro_acceleration.md`](docs/fish_s2pro_acceleration.md).

## Quick start (bare metal)

```bash
# Backend (GPU box): install sglang-omni, then:
hf download fishaudio/s2-pro
sgl-omni serve --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml --tts-batch-max-items 32 --port 8000

# Gateway (CPU): install deps with uv and run, pointing at the backend.
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt          # or: bash flowtts/setup/setup.sh
bash run.sh --ctrl-port 8764 --ports 1 --backend-url http://127.0.0.1:8000
```

---

## WebSocket API  (unchanged)

Connect to `ws://<host>:8080/ws/<call_id>` and send JSON messages.

**Client → Server**
```json
{
  "type": "synthesize",
  "call_id": "call-123",
  "text_id": "utt-1",
  "text": "नमस्ते, मैं प्रिया बोल रही हूँ बजाज फाइनेंस से।",
  "voice_id": "priya",         // optional — alias of a cloned voice; omit for default
  "speed": 1.0,                 // optional — >1 faster, <1 slower
  "language": "hi",             // optional — omit to auto-detect
  "streaming": true             // optional — defaults to server setting
}
```
Cancel an in-flight utterance: `{ "type": "cancel", "text_id": "utt-1" }`

**Server → Client** (streaming) — repeated **binary** frames, each a JSON header
immediately followed by raw little-endian **int16 PCM** bytes:
```
{"type":"audio_chunk","call_id":"call-123","text_id":"utt-1","chunk_index":0,
 "sample_rate":24000,"encoding":"pcm_int16","tokens":12000,"is_final":false,"cache_hit":false}<PCM…>
```
Then a final JSON frame:
```json
{ "type":"audio_done","call_id":"call-123","text_id":"utt-1","chunks":3,
  "total_wav_bytes":153600,"sample_rate":24000,"rtf":0.21 }
```
Errors: `{ "type":"error", "call_id", "text_id", "error" }`.
Concatenate the PCM from each `audio_chunk` (same `sample_rate`) for the full utterance.

---

## Voices (clone by alias)

A voice is a reference clip + its transcript stored in `voices/` (`<alias>.wav` +
`<alias>.json`). No codec tokens are precomputed — the backend encodes the clip on first
use and caches it. `ref_text` is **mandatory** (no ASR) and must be the exact transcript
in the clip's language/script.

**REST (easiest — live, no restart):**
```bash
curl -sf -X POST http://localhost:8764/voices \
  -F voice_id=priya -F preferred_lang=hi \
  -F ref_text="नमस्ते, मैं प्रिया बोल रही हूँ।" \
  -F audio=@sample_files/simran.wav
# → {"status":"ok","voice_id":"priya", ...}   → usable immediately
```
`POST /voices` inputs: `audio` (file), `voice_id`, `ref_text` (**required**),
`preferred_lang` (optional). `GET /voices` lists loaded voices.

**Offline CLI (no GPU):**
```bash
python -m flowtts.voices.clone --build-all --manifest voices/manifest.json
python -m flowtts.voices.clone --add priya --ref-audio sample_files/simran.wav \
    --ref-text "नमस्ते, मैं प्रिया बोल रही हूँ।" --lang hi
python -m flowtts.voices.clone --list
```
`voices/` is a shared volume the backend mounts (read-only), so cloned references
resolve on both sides. See [`voices/README.md`](voices/README.md).

---

## Configuration & tuning

All settings are overridable via `FLOWTTS_*` env vars (nested with `__`). Key gateway knobs:

| Env var | Meaning | Default |
|---|---|---|
| `FLOWTTS_FISH__BACKEND_URL` | sglang backend base URL | `http://fish-s2pro:8000` |
| `FLOWTTS_FISH__MODEL` | model id echoed in metrics / OpenAI `model` field | `fishaudio/s2-pro` |
| `FLOWTTS_FISH__REFERENCE_MODE` | `local` (shared volume) or `base64` (inline clip) | `local` |
| `FLOWTTS_FISH__BACKEND_VOICES_DIR` | backend's voices mount path (if it differs) | `null` |
| `FLOWTTS_FISH__INITIAL_CODEC_CHUNK_FRAMES` | frames before first decode (↓ = lower TTFB) | `null` |
| `FLOWTTS_OUTPUT__SAMPLE_RATE` | resample from native 24 kHz (e.g. `16000`, `8000`) | `24000` |
| `FLOWTTS_VOICES__DEFAULT_VOICE` | alias used when `voice_id` omitted | `priya` |

Backend knobs (in `docker-compose.yml`): `TTS_BATCH_MAX_ITEMS`, `MEM_FRACTION`,
`FISH_MODEL`, `HF_TOKEN`. The full speedup playbook is in
[`docs/fish_s2pro_acceleration.md`](docs/fish_s2pro_acceleration.md).

---

## Performance targets

Aiming for **~200 RPS** and **TTFB < 200 ms** on a single H200. Throughput comes from
sglang's continuous batching + KV cache and from **voice reuse** (RadixAttention prefix
cache, ~86–90% hit); latency comes from AR streaming. This is a **tuning target** — reach
it by raising `TTS_BATCH_MAX_ITEMS` / `MEM_FRACTION`, reusing voices, and leaning on the
WAV cache. **Benchmark on your hardware.**

---

## Project layout

```
flowtts/
├── server.py            # primary single-process WS gateway (production entry point)
├── main.py, worker.py   # secondary Redis-backed multi-process path
├── core/config.py       # all settings (backend URL, generation, output, streaming)
├── synthesis/
│   ├── fish_engine.py       # sglang backend client + live voice cloning (no GPU)
│   ├── models.py            # FishSpeechSynthesizer (whole + streaming facade)
│   ├── engine.py            # process-wide singleton
│   └── text_chunker.py      # normalize + (secondary-path) text splitter
├── voices/              # reference-clip registry + manifest store + offline builder
├── decoder/             # waveform → PCM/WAV helpers
├── processing/          # resample, crossfade, fades
├── api/                 # WebSocket gateway (Redis path) + message models
├── monitoring/          # structlog + Prometheus metrics
└── test/                # unit tests + benchmark client
docker/                  # gateway Dockerfile + fish_s2pro.Dockerfile + entrypoint
docs/                    # acceleration playbook
```

---

## Testing

```bash
# unit tests (no GPU required)
python -m pytest flowtts/test/test_text_chunker.py \
                 flowtts/test/test_voice_store.py \
                 flowtts/test/test_pcm.py -q

# end-to-end streaming benchmark against a running gateway
python -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75 --streaming
```

---

## License

The serving code in this repository is provided as-is. **Fish Audio S2 Pro** and its
weights are governed by the **Fish Audio Research License** (non-commercial; commercial
use requires a separate license — `business@fish.audio`). **sglang-omni** and **SGLang**
have their own licenses. Review all of them before production or commercial use.

## Acknowledgements

- [Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro) — the TTS model.
- [sglang-omni](https://github.com/sgl-project/sglang-omni) / [SGLang](https://github.com/sgl-project/sglang) — the serving engine.
