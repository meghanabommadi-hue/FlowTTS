# FlowTTS — OmniVoice streaming TTS server

A production-grade, low-latency **text-to-speech server** built around
[**k2-fsa/OmniVoice**](https://github.com/k2-fsa/OmniVoice) — a massively multilingual
(600+ languages) zero-shot TTS model — with a **TensorRT-accelerated backbone**,
**voice cloning**, **streaming synthesis**, and a **multilingual text preprocessor**
covering all 22 scheduled languages of India. Built for voice-bot / IVR workloads.

> OmniVoice is a non-autoregressive **discrete-diffusion language model** (Qwen3-0.6B
> backbone + Higgs-Audio-v2 neural codec, **24 kHz**). Because it is not
> autoregressive it cannot emit audio token by token, so streaming is done by
> chunking the text, generating the chunks as one batch, and stitching the
> results — which is what most of this repo is about.

Acceleration follows [tlitech/omnivoice-trtllm](https://github.com/tlitech/omnivoice-trtllm);
text preprocessing is ported from
[Ajaj-Ali/text_preprocessor_for_TTS](https://github.com/Ajaj-Ali/text_preprocessor_for_TTS).

---

## Features

- 🚀 **TensorRT-accelerated backbone** — the Qwen3 transformer runs from a
  compiled engine, following [tlitech/omnivoice-trtllm](https://github.com/tlitech/omnivoice-trtllm):
  only `llm.forward` is replaced, every other line of OmniVoice runs untouched.
  **2.0x lower TTFB, 1.44x throughput** vs PyTorch, validated at cosine 0.999998
  against the real module before it is ever installed.
- 🌏 **All 22 scheduled languages of India** — plus a multilingual text
  preprocessor (numbers, dates, currency, OTPs, phone numbers, URLs,
  abbreviations) ported and extended from
  [Ajaj-Ali/text_preprocessor_for_TTS](https://github.com/Ajaj-Ali/text_preprocessor_for_TTS),
  with per-script normalization of code-mixed (Hinglish) input.
- 🎙️ **Sentence-aligned chunking** — text is cut at sentence ends, ~200 chars
  (±50), so an ordinary utterance is generated in **one piece** and never
  stitched at all. When a cut is unavoidable the stitcher knows *why* it
  happened and gives it the right pause, and normalizes every chunk to one level
  for the whole utterance. Median TTFB ~90 ms.
- 🗣️ **Voice cloning** — REST clone usable immediately with no restart, a
  one-shot preview that saves nothing, and inline reference audio per request.
- 🎛️ **Every OmniVoice parameter exposed** — all 13 generation-config fields plus
  speed, duration, instruct and language, per request, on every transport.
- 🔌 **Four transports, one model load** — REST, streaming REST,
  OpenAI-compatible `/v1/audio/speech`, and the FlowTTS WebSocket protocol.
- ⚡ **Dynamic in-flight batching** — concurrent requests and streamed chunks are
  coalesced into single batched `generate()` calls, bucketed by generation
  config, voice-prompt presence and length.
- 📊 **Ops built-in** — `/healthz`, `/readyz`, `/v1/stats`, `/metrics`, OOM
  recovery, a WAV cache that serves repeats in ~1 ms.

---

## Architecture

```
Client ──HTTP/WS──▶ api/http_app.py   (one FastAPI app: REST + OpenAI + /ws)
                        │
                        ▼
                    text/pipeline.py            normalize per script run
                        │                        (numbers, dates, OTPs, tags kept)
                        ▼
                    synthesis/chunker.py        duration-aware chunks,
                        │                        short first chunk, ≥1 s floor
                        ▼
                    synthesis/omnivoice_engine.py
                        │  bucketed dynamic batch queue
                        ▼
                    OmniVoice.generate()        [upstream code, untouched]
                        └─ llm.forward ─────▶ trt/runtime.py
                                               TensorRT | TRT-LLM | torch
                        ▼
                    processing/stitch.py        trim, DC-remove, crossfade
                        ▼
Client ◀── audio chunks, first byte as soon as chunk 0 lands
```

The acceleration is one monkey-patch, exactly as upstream does it. Embeddings,
audio heads, the iterative unmasking loop, CFG scoring and the Higgs codec are
all upstream's code — which is what keeps the audio identical.

---

## Requirements

- NVIDIA GPU, compute capability **≥ 8.0** (Ada RTX 6000 = 8.9, Hopper H200 = 9.0).
- Linux, Python **3.10–3.12**, PyTorch ≥ 2.4 (CUDA build), `transformers ≥ 5.3`.
- For Docker: NVIDIA driver + **nvidia-container-toolkit** + Docker Compose v2.

---

## Quick start

```bash
# 1. install (a CUDA build of torch first, if you do not already have one)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. build the TensorRT engine for the Qwen3 backbone (once per box, ~60 s)
python -m flowtts.trt.build_trt --precision fp16 --max-batch 64 --max-seq 2048

# 3. serve: REST + OpenAI + WebSocket on :9000, control on :9764
python -m flowtts.service --profile balanced --http-port 9000 --ctrl-port 9764
```

The engine build is optional — without it the service runs on PyTorch and says
so in the startup log. `backbone_backend=auto` picks the best backend present
and validates it against the real module before installing it, so a missing or
broken engine degrades to correct-but-slower rather than to bad audio.

Profiles: `fast` (num_step 4), `balanced` (8, default), `quality` (32).

For deployment on a shared box behind nginx, see
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) and [`deploy/`](deploy/).

---

## API

Full OpenAPI browser at `/docs`. Every endpoint accepts the same parameter set.

### Synthesize

```bash
curl -X POST http://localhost:9000/v1/tts -H 'content-type: application/json' -d '{
  "text": "आपका बकाया ₹2,500 है, कृपया 15/04/2026 तक भुगतान करें।",
  "language": "hi",
  "voice_id": "anika",
  "format": "wav",
  "generation": {"num_step": 4, "guidance_scale": 2.0}
}' --output speech.wav
```

### Stream (low TTFB)

```bash
curl -N -X POST http://localhost:9000/v1/tts/stream \
  -H 'content-type: application/json' \
  -d '{"text":"...","language":"hi","voice_id":"anika","format":"pcm"}' > out.pcm
```

First bytes ship as soon as chunk 0 leaves the GPU, while later chunks are still
generating. Behind nginx this needs `proxy_buffering off` — see
[`deploy/nginx/omnivoice.conf`](deploy/nginx/omnivoice.conf).

### OpenAI-compatible

```bash
curl -X POST http://localhost:9000/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"omnivoice","input":"Hello there.","voice":"anika","response_format":"wav"}'
```

Works with the OpenAI SDKs unchanged; also accepts this server's own fields
(`language`, `instruct`, `generation`, `normalizer`).

### Voice cloning

```bash
# clone — usable immediately, no restart
curl -X POST http://localhost:9000/v1/voices \
  -F voice_id=priya -F language=hi \
  -F reference_text="नमस्ते, मैं प्रिया बोल रही हूं।" \
  -F audio=@sample.wav

# hear it before keeping it — nothing is written to disk
curl -X POST http://localhost:9000/v1/voices/preview \
  -F audio=@sample.wav -F reference_text="नमस्ते..." \
  -F text="अब मैं क्लोन की गई आवाज़ में बोल रही हूं।" --output preview.wav

curl http://localhost:9000/v1/voices
curl -X DELETE http://localhost:9000/v1/voices/priya
```

`reference_text` is required — this server runs no ASR, and a wrong transcript
degrades every future synthesis with that voice.

### The three OmniVoice modes

| mode | how |
|---|---|
| voice clone | `voice_id`, or `reference_audio` + `reference_text` inline |
| voice design | `instruct: "Female, Elderly, British Accent"` |
| auto voice | neither |

Inline control tags pass through untouched: `[laughter]`,
`[dissatisfaction-hnn]`, ARPAbet `[B EY1 S]`.

### Every generation parameter, per request

```json
{"generation": {
  "num_step": 4, "guidance_scale": 2.0, "t_shift": 0.1,
  "layer_penalty_factor": 5.0, "position_temperature": 5.0,
  "class_temperature": 0.0, "denoise": true,
  "preprocess_prompt": true, "postprocess_output": true,
  "pad_duration": 0.1, "fade_duration": 0.1,
  "audio_chunk_duration": 15.0, "audio_chunk_threshold": 30.0
}}
```

`num_step` is the only real latency dial. **`guidance_scale` is not a speed
knob on this model** — `_generate_iterative` builds the cond+uncond batch either
way, so 0 costs the same as 2.0 (186 ms vs 190 ms measured) while collapsing
short chunks to silence. Leave it at 2.0.

### Text preprocessing

```bash
curl -X POST http://localhost:9000/v1/normalize -H 'content-type: application/json' \
  -d '{"text":"आपका बकाया ₹2,500 है, OTP 4821","language":"hi"}'
# → "आपका बकाया दो हज़ार पाँच सौ रुपये है, O T P चार आठ दो एक"
#   plus the chunks and their estimated durations
```

Per-request toggles live under `"normalizer"`; `GET /v1/languages` lists the
30 languages with normalization tables (OmniVoice itself speaks 600+).

### WebSocket

Connect to `ws://<host>:9000/ws/<call_id>` and send JSON.

```json
{"type": "synthesize", "text_id": "utt-1",
 "text": "नमस्ते, मैं प्रिया बोल रही हूँ।",
 "voice_id": "priya", "language": "hi", "speed": 1.0,
 "generation": {"num_step": 4}}
```

Replies are binary frames — a JSON header immediately followed by raw
little-endian int16 PCM — then a final `audio_done` JSON frame. `{"type":"cancel",
"text_id":"utt-1"}` stops an utterance in flight; `{"type":"ping"}` → `pong`.

---

## Configuration & tuning

Everything is overridable via `FLOWTTS_*` env vars (nested with `__`). The knobs
worth knowing:

| Env var | Meaning | Default |
|---|---|---|
| `FLOWTTS_GENERATION__NUM_STEP` | denoise steps — the dominant latency dial | `16` |
| `FLOWTTS_GENERATION__GUIDANCE_SCALE` | CFG strength — **not** a speed dial here | `2.0` |
| `FLOWTTS_OMNIVOICE__BACKBONE_BACKEND` | `auto` \| `tensorrt` \| `trtllm` \| `torch` \| `pytorch` | `auto` |
| `FLOWTTS_OMNIVOICE__TRT_ENGINE_DIR` | where `build_trt` writes `backbone.plan` | `engines/omnivoice-backbone` |
| `FLOWTTS_OMNIVOICE__BACKBONE_MIN_COSINE` | reject an engine below this vs PyTorch | `0.99` |
| `FLOWTTS_OMNIVOICE__MAX_BATCH` | dynamic batch size | `16` |
| `FLOWTTS_OMNIVOICE__MAX_BATCH_FRAMES` | total audio frames per batch | `6000` |
| `FLOWTTS_STREAMING__FIRST_CHUNK_SECONDS` | target size of the low-TTFB first chunk | `1.2` |
| `FLOWTTS_STREAMING__MIN_CHUNK_SECONDS` | floor — below this the model returns silence | `1.0` |
| `FLOWTTS_OUTPUT__SAMPLE_RATE` | resample from native 24 kHz (`16000`, `8000`) | `24000` |
| `FLOWTTS_VOICES__DEFAULT_VOICE` | alias used when `voice_id` is omitted | `priya` |
| `FLOWTTS_TEXT__ENABLED` | master switch for text normalization | `true` |
| `FLOWTTS_TEXT__DETECT_LANGUAGE` | infer the inference language from the script | `false` |
| `FLOWTTS_API_KEYS` | if set, requires Bearer / `X-API-Key` | *(none)* |

Two findings worth carrying into any tuning you do, both measured on this model:

- **`guidance_scale` is not a throughput lever.** `_generate_iterative` builds
  the conditional + unconditional batch and runs the backbone over all of it on
  every step regardless of the value — 186 ms at 0.0 versus 190 ms at 2.0. What
  it does change is robustness: at 0 the model collapses to near-silence on short
  chunks. `num_step` is the dial; leave CFG at 2.0.
- **Always pass `language`.** It conditions the model's phonemes and changes the
  output completely: on identical text, `hi` versus `mr` differ by 1.71 mean
  absolute error against a 0.0 rerun noise floor. Script detection is off by
  default because it identifies a *script*, not a language — everything in
  Devanagari would be read as Hindi. Requests missing it are counted in
  `/v1/stats` under `no_language`.

---

## Project layout

```
flowtts/
├── service.py             process entry point — HTTP + WS + control, one model load
├── api/
│   ├── http_app.py        FastAPI: REST, streaming, OpenAI-compatible, /ws
│   ├── models.py          request/response schemas — every OmniVoice parameter
│   ├── audio_io.py        resample + wav/pcm/mp3/opus encoding
│   └── service.py         shared state: WAV cache, admission limiter, OOM recovery
├── trt/                   the omnivoice-trtllm port
│   ├── backbone.py        exportable Qwen3 mirror (bit-exact vs transformers)
│   ├── build_trt.py       ONNX → TensorRT engine  (uses the installed TensorRT)
│   ├── build_trtllm.py    checkpoint → TRT-LLM engine (needs tensorrt_llm)
│   ├── convert_checkpoint.py  vendored from upstream, unchanged
│   ├── patch/omnivoice/   vendored TRT-LLM network definition, unchanged
│   ├── runtime.py         TensorRT | TRT-LLM | torch, one contract
│   └── patcher.py         validate, then monkey-patch llm.forward
├── text/                  multilingual preprocessing (indic-tts-normalizer port)
│   ├── languages.py       30 languages incl. all 22 scheduled Indian languages
│   ├── script_detect.py   code-mixed segmentation
│   ├── numbers.py         cardinal backends with a total fallback chain
│   └── pipeline.py        orchestrator + control-tag protection
├── synthesis/
│   ├── chunker.py         duration-aware smart chunking
│   ├── omnivoice_engine.py  model load, bucketed batching, cloning
│   └── models.py          normalize → chunk → generate → stitch
├── processing/stitch.py   silence trim, DC removal, equal-power crossfade
├── voices/                npz voice-clone registry
└── test/                  unit tests + bench + deployment verification
deploy/                    nginx config, env, start/stop scripts
docs/DEPLOYMENT.md         the live deployment, with measured numbers
```

---

## Performance

Measured on an L40S **shared with a multi-day training job**. Streaming, Hindi
voice-bot text, cold cache, TensorRT FP16 backbone, `num_step=4`:

| requests/sec | TTFB p50 | TTFB p90 | TTFB p99 | failures |
|---|---|---|---|---|
| 1 | 87 ms | 114 ms | 165 ms | 0 |
| 4 | 102 ms | 157 ms | 163 ms | 0 |
| 8 | 105 ms | 146 ms | 176 ms | 0 |
| 10 | 96 ms | 162 ms | 181 ms | 0 |
| 12 | 125 ms | 190 ms | 272 ms | 0 |

**p99 TTFB stays under 200 ms to 10 requests/second**; the GPU saturates at
~18 rps (82x realtime). TensorRT vs PyTorch: **2.0x lower TTFB, 1.44x
throughput** — matching upstream's reported FP16 figure.

Under *fixed* concurrency the queue dominates: 64 simultaneous requests give a
~2.8 s median TTFB, because the card can only do ~18 requests/second no matter
how the queue is arranged. For a voice bot that distinction matters — 100 open
sessions are fine, 100 simultaneous *turns* are not. Full tables and the
reasoning are in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Testing

```bash
# unit tests — no GPU required (172 tests)
python -m pytest flowtts/test/test_text_normalizer.py flowtts/test/test_chunker.py \
                 flowtts/test/test_stitch.py flowtts/test/test_api.py \
                 flowtts/test/test_voice_npz.py flowtts/test/test_pcm.py -q

# against a running server
python -m flowtts.test.verify_deployment --url http://127.0.0.1:9000   # 38 checks
python -m flowtts.test.bench --rate 1,2,4,6,8 --duration 15            # offered load
python -m flowtts.test.bench --sweep 1,4,16,64,100 --requests 200      # concurrency

# engine correctness and generation-setting sweeps
python -m flowtts.test.diagnose_backbone       # transformers vs mirror vs TRT
python -m flowtts.test.sweep_generation        # num_step x guidance_scale
python -m flowtts.test.diagnose_chunk --text "..."
```

---

## License

The serving code in this repository is provided as-is. **k2-fsa/OmniVoice** and its
weights are governed by their own upstream license (Apache-2.0) — review it before
production or commercial use.

## Acknowledgements

- [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) — the TTS model.
- [Higgs-Audio-v2](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) — the neural audio codec used by OmniVoice.
