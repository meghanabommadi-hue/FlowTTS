# FlowTTS — OmniVoice streaming TTS server

A production-grade, low-latency **text-to-speech WebSocket server** built around
[**k2-fsa/OmniVoice**](https://github.com/k2-fsa/OmniVoice) — a massively multilingual
(600+ languages) zero-shot TTS model. Designed for high-throughput voice-bot / IVR
workloads (Hindi + English), with **realtime chunk-wise streaming**, **dynamic in-flight
batching**, and **reusable voice clones**.

> OmniVoice is a non-autoregressive **discrete-diffusion language model** (Qwen3-0.6B
> backbone + Higgs-Audio-v2 neural codec, **24 kHz**). It replaced this repo's previous
> sglang + ncodec stack; the serving framework (WebSocket gateway, batching, metrics,
> WAV cache, streaming protocol) was kept.

---

## Features

- 🎙️ **Realtime streaming** — text is split into chunks (short first chunk); each is
  synthesized and streamed as raw int16 PCM, targeting **TTFB < 200 ms**.
- ⚡ **Dynamic in-flight batching** — concurrent requests *and* streamed chunks are
  coalesced into single batched `generate()` calls, length-bucketed to minimize padding.
- 🗣️ **Voice cloning by alias** — reference voices are precomputed once into tiny `.npz`
  artifacts and loaded at startup; requests pick one via `voice_id`.
- 🚀 **Acceleration levers** — fewer denoise steps, `torch.compile` + CUDA graphs, bf16/FP8
  on Hopper/Ada, plus a SHA-256 **WAV cache** that bypasses the model on repeated prompts.
- 🔌 **Plug-n-play protocol** — a stable WebSocket in/out contract (binary PCM frames).
- 📊 **Ops built-in** — `/health`, `/ready`, `/metrics` (Prometheus), OOM recovery,
  on-demand port binding, idle-connection reaping.
- 🐳 **Containerized** — one `docker compose` command; keeps the host VM clean.

---

## Architecture

```
Client ──WS(text)──▶ server.py  (OmniVoice loaded once, N ports, one asyncio loop)
                        │  normalize + split into streaming chunks (short first chunk)
                        │  WAV-cache lookup (sha256(text)) ─hit─▶ send cached PCM ▶ done
                        ▼
                     OmniVoiceEngine.synthesize(chunk, voice_id, speed, language)
                        │  enqueue (text, VoiceClonePrompt, gen_cfg, future)
                        ▼
                     dynamic batch queue  (length-bucketed, ~8 ms window)
                        │  model.generate([...])  → 24 kHz waveforms   [GPU worker thread]
                        ▼
                     resample (optional) → int16 PCM → boundary crossfade
                        ▼
Client ◀── audio_chunk (JSON header + PCM bytes) … audio_done ──
```

A secondary Redis-backed multi-process path (`main.py` + `worker.py`) exists for
cross-machine scaling; the single-process `server.py` above is the primary, lowest-latency path.

---

## Requirements

- NVIDIA GPU, compute capability **≥ 8.0** (Ada RTX 6000 = 8.9, Hopper H200 = 9.0).
- Linux, Python **3.10–3.12**, PyTorch ≥ 2.4 (CUDA build), `transformers ≥ 5.3`.
- For Docker: NVIDIA driver + **nvidia-container-toolkit** + Docker Compose v2.

---

## Quick start (Docker — recommended)

```bash
git clone <your-repo-url> FlowTTS && cd FlowTTS

# one-time: build image, download OmniVoice (~3.3 GB), build voice npz from sample_files/
docker compose run --rm omnivoice-tts setup

# serve on :8080 (WebSocket) + :8764 (control API / Prometheus)
docker compose up -d
docker compose logs -f omnivoice-tts

# smoke test
docker compose exec omnivoice-tts python -m flowtts.test.test_pipeline \
    --ctrl-port 8764 --requests 5 --streaming
```

Full container guide + tuning: [`docker/README.md`](docker/README.md).

## Quick start (bare metal)

```bash
# install a CUDA build of torch first, e.g.:
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
bash flowtts/setup/setup.sh            # deps + model + voices
bash run.sh --ctrl-port 8764 --ports 1 # serve
```

---

## WebSocket API

Connect to `ws://<host>:8080/ws/<call_id>` and send JSON messages.

**Client → Server**
```json
{
  "type": "synthesize",
  "call_id": "call-123",
  "text_id": "utt-1",
  "text": "नमस्ते, मैं प्रिया बोल रही हूँ बजाज फाइनेंस से।",
  "voice_id": "priya",        // optional — alias of a built voice; omit for default
  "speed": 1.0,                // optional — >1 faster, <1 slower
  "language": "hi",            // optional — omit to auto-detect
  "streaming": true            // optional — defaults to server setting
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

A voice is a precomputed `voices/<alias>.npz` (Higgs-codec tokens of a reference clip +
its transcript + loudness). Build once, offline; the server loads them at startup.

```bash
# build all voices from sample_files/ (+ voices/manifest.json overrides)
python -m flowtts.voices.clone --build-all --manifest voices/manifest.json
# add one (ref_text is required — no ASR/auto-transcribe)
python -m flowtts.voices.clone --add priya --ref-audio sample_files/priya.wav \
    --ref-text "नमस्ते, मैं प्रिया बोल रही हूँ।"
python -m flowtts.voices.clone --list
```
Restart the server to pick up new voices. In Docker, prefix with
`docker compose run --rm omnivoice-tts clone …`. See [`voices/README.md`](voices/README.md).

---

## Configuration & tuning

All settings are overridable via `FLOWTTS_*` env vars (nested with `__`). Key knobs:

| Env var | Meaning | Default |
|---|---|---|
| `FLOWTTS_OMNIVOICE__MODEL_PATH` | local weights dir (used if it exists; else HF repo) | `model_dir/base` |
| `FLOWTTS_OMNIVOICE__NUM_STEP` | diffusion steps (dominant latency knob) | `16` |
| `FLOWTTS_OMNIVOICE__MAX_BATCH` | dynamic batch size | `32` |
| `FLOWTTS_OMNIVOICE__BATCH_TIMEOUT_MS` | batch collection window (ms) | `8` |
| `FLOWTTS_OMNIVOICE__GUIDANCE_SCALE` | classifier-free guidance strength | `2.0` |
| `FLOWTTS_OMNIVOICE__COMPILE_MODEL` | `torch.compile` (+ CUDA graphs) | `false` |
| `FLOWTTS_OMNIVOICE__DTYPE` | `bfloat16` / `float16` | `bfloat16` |
| `FLOWTTS_OUTPUT__SAMPLE_RATE` | resample from native 24 kHz (e.g. `16000`, `8000`) | `24000` |
| `FLOWTTS_VOICES__DEFAULT_VOICE` | alias used when `voice_id` omitted | `priya` |
| `FLOWTTS_STREAMING__FIRST_CHUNK_MAX_CHARS` | size of the low-TTFB first chunk | `60` |

`run.sh` exposes the common ones as flags (`--num-step`, `--max-batch`, `--compile`, …).
The full speedup playbook (fewer-step distillation, CFG-cost reduction, FP8, codec
overlap, TensorRT, …) with per-technique tradeoffs is in
[`docs/omnivoice_acceleration.md`](docs/omnivoice_acceleration.md).

---

## Performance targets

Aiming for **~200 RPS** and **TTFB < 200 ms** on a single H200. This is a **tuning
target**, not a guarantee — reach it by combining a large `MAX_BATCH`, low `NUM_STEP`,
short first chunks, `torch.compile`, and the WAV cache (call-centre prompts repeat
heavily). No absolute RPS numbers are published upstream for OmniVoice; **benchmark on
your hardware** with the throughput sweep and tune from there.

---

## Project layout

```
flowtts/
├── server.py            # primary single-process WS server (production entry point)
├── main.py, worker.py   # secondary Redis-backed multi-process path
├── core/config.py       # all settings (model, output, streaming, batching, accel)
├── synthesis/
│   ├── omnivoice_engine.py  # model load + dynamic batcher + accel hooks
│   ├── models.py            # OmniVoiceSynthesizer (whole + streaming)
│   ├── engine.py            # process-wide singleton
│   └── text_chunker.py      # streaming text splitter (pure stdlib)
├── voices/              # npz voice-clone registry + offline builder CLI
├── decoder/             # waveform → PCM/WAV helpers
├── processing/          # resample, crossfade, fades
├── api/                 # WebSocket gateway (Redis path) + message models
├── monitoring/          # structlog + Prometheus metrics
└── test/                # unit tests + benchmark client
docker/                  # Dockerfile, entrypoint, container README
docs/                    # acceleration playbook
```

---

## Testing

```bash
# unit tests (no GPU required)
python -m pytest flowtts/test/test_text_chunker.py \
                 flowtts/test/test_voice_npz.py \
                 flowtts/test/test_pcm.py -q

# end-to-end streaming benchmark against a running server
python -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75 --streaming
```

---

## License

The serving code in this repository is provided as-is. **k2-fsa/OmniVoice** and its
weights are governed by their own upstream license (Apache-2.0) — review it before
production or commercial use.

## Acknowledgements

- [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) — the TTS model.
- [Higgs-Audio-v2](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) — the neural audio codec used by OmniVoice.
