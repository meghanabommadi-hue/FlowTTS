# DhVaani on FlowTTS — operator guide

DhVaani-0.5 (ARTPARK-IISc) is a zero-shot voice-cloning TTS covering **27 Indian
languages** at 24 kHz. This document covers how it is served here, every knob
worth touching, and — honestly — what it takes to reach 200 RPS.

---

## 1. What DhVaani is, and why it changes the serving design

The existing FlowTTS path (MiraTTS) is an **autoregressive LLM plus an audio
codec**: sglang generates speech tokens one at a time, and the decoder turns
them into PCM. Streaming falls out for free — tokens arrive incrementally.

DhVaani is a fine-tune of [ZipVoice](https://github.com/k2-fsa/ZipVoice), a
123M-parameter **flow-matching** model. It is a completely different shape:

| | MiraTTS (existing) | DhVaani (new) |
|---|---|---|
| Generation | autoregressive, token by token | non-autoregressive, whole span at once |
| Iterations | 1 per token, variable | exactly `num_step`, fixed |
| State | KV cache grows with length | none |
| Streaming | free (tokens arrive over time) | must be engineered (see §3) |
| Output | codec tokens -> 16 kHz | mel -> Vocos -> 24 kHz |
| Batching | continuous, standard LLM style | continuous, but only via §2 |

Three consequences drive everything below:

1. **There is no partial output.** A span produces nothing until all
   `num_step` Euler iterations finish. Time-to-first-byte therefore equals
   time-to-everything unless the text is split.
2. **Cost is perfectly predictable.** `num_step x frames x constant`. No
   variable-length KV cache, no ragged attention — which makes it unusually
   easy to batch densely and to model capacity analytically (§8).
3. **The voice prompt is re-rendered on every span.** It is not a cache; it is
   part of the sequence the flow decoder attends over, every time.

### The request lifecycle

```
text
 -> TextNormalizer      cached; numbers, dates, abbreviations, native digits
 -> SmartChunker        span schedule: short first, then ramping
 -> tokenizer           character level, 1058-token Indic vocab
 -> FlowScheduler       continuous-batched Euler ODE                    [GPU]
 -> VocodeStage         micro-batched Vocos + resample                  [GPU]
 -> SpanStitcher        overlap-add crossfade across spans
 -> AudioChunk          PCM, emitted in span order
```

---

## 2. Continuous batching on a flow-matching model

The naive approach batches whole spans: everyone in a batch runs their 8 steps
in lockstep, the batch retires together, and a request arriving one step after a
batch starts waits for the whole thing. That is static batching, with all its
head-of-line blocking.

The escape hatch is a single fact about the model:

```python
# zipvoice/models/modules/zipformer.py :: TTSZipformer.forward
#   t: A t tensor of shape (batch_size,) or (batch_size, seq_len)
assert t.dim() == 1 or t.dim() == 2, t.shape
```

**The flow decoder accepts a per-sample timestep.** Upstream never exploits it —
`solver.DiffusionModel.forward` asserts `t.dim() == 0` because it only ever runs
homogeneous batches. But the network genuinely supports it, so one forward pass
can contain a span on its 1st Euler step next to one on its 7th.

`flowtts/dhvaani/engine/scheduler.py` is built on that. Every tick it:

1. **admits** queued spans into free arena slots (encoding the whole admitted
   group through the text encoder in one call),
2. **steps** each length bucket once, with a per-row `t` vector,
3. **retires** rows that reached `num_step`, freeing their slots immediately.

A new span joins at the very next tick. This is the direct analogue of
in-flight batching in LLM serving, and it is verified by
`test_scheduler.py::test_spans_at_different_steps_share_one_forward_pass`.

Classifier-free guidance is generalised the same way. Upstream branches on
`if t > 0.5` in Python; `ops.cfg_expand` turns that into an elementwise select
so rows either side of the threshold coexist in one batch.

### Arenas, and why VRAM stays flat

Every in-flight trajectory lives in a **pre-allocated** slot. Frame counts round
up to a multiple of 64 and clamp to `[128, 1536]`; each bucket owns fixed
`(max_batch, T, 100)` tensors for `x`, `text_condition` and `speech_condition`.

The per-request flow path allocates **nothing**: conditions are written into an
existing slot and `x` is mutated in place for all `num_step` iterations. (CFG
still materialises a doubled `[uncond; cond]` batch each step, but always at one
of the fixed bucket shapes, so the caching allocator hands back the same blocks
rather than growing.)

That matters more than it sounds: the usual
cause of "the TTS server slowly eats VRAM" is not a leak but *fragmentation* —
thousands of differently-sized allocations per second leave the caching
allocator holding blocks it cannot reuse, and `reserved` drifts up while
`allocated` stays flat. Three defences, in order of importance:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set automatically),
2. the pre-allocated arenas,
3. `VramWatchdog`, which collects only when reserved-minus-allocated actually
   exceeds `memory.gc_reserved_slack_fraction`, at most once per
   `memory.gc_min_interval_s`.

`torch.cuda.empty_cache()` synchronises the device, so it is deliberately rare.
`smoke.py` asserts VRAM is flat across 30 requests.

Occupied rows are kept **packed at the front** of each arena, so a step is a
zero-copy `arena.x[:n]` slice rather than an index_select that would copy the
whole batch every iteration. Releasing a slot swaps the last row into the hole —
one row copy per retirement instead of a full gather per step.

---

## 3. Smart chunking: how streaming works at all

Since a span yields nothing until it is finished, streaming comes from splitting
the text and pipelining spans. But spans are not all the same size, and the
reason is the prompt.

Every span costs:

```
(prompt_frames + span_frames) x num_step x (2 if CFG)
```

The prompt is paid **every time**. With a 2 s prompt (187 frames):

| span | prompt overhead |
|---|---|
| 1.0 s | 65% |
| 2.5 s | 44% |
| 4.5 s | 31% |

So short spans are latency-optimal and throughput-terrible, and long spans are
the reverse. The schedule ramps:

```python
first_chunk_seconds  = 1.2   # get audio moving
second_chunk_seconds = 2.5
steady_chunk_seconds = 4.5   # amortise the prompt
```

Time-to-first-byte depends only on the **first** span. By the time the client
has played 1.2 s, several more spans have landed — RTF is well below 1, so the
buffer fills far faster than it drains.

Boundaries are taken at sentence terminators first (including `।` `॥` `؟` `۔`),
then clause punctuation, then whitespace, and only then mid-token. Each span
gets terminal punctuation appended, matching upstream `add_punctuation()` —
ZipVoice was trained on punctuated text and a span without it ends abruptly.

The chunker also clamps against the largest arena bucket, so it can never emit a
span the scheduler would reject.

### Stitching spans back together

Spans come from **independent** flow trajectories: each starts from its own
Gaussian noise and lands with its own DC offset and phase. Concatenating them
clicks.

The older MiraTTS path fades in each chunk's head, which works there because
consecutive chunks decode from one token stream with overlapping context. Here
there is no shared context, and a fade-in alone leaves an audible seam.

`SpanStitcher` does true overlap-add: it holds back the last `crossfade` samples
of every non-final span and blends them against the next span's head with
complementary ramps. Measured on a step from +0.5 to -0.5, the seam
discontinuity drops from **1.0 to 0.0007**. It costs no perceived latency — the
held tail is emitted as soon as the next span lands.

---

## 4. Text normalisation

Backed by [`indic_tts_normalizer`](https://github.com/Ajaj-Ali/text_preprocessor_for_TTS).

```
"आपकी EMI ₹2,500 है"  ->  "आपकी ई एम आई दो हज़ार पाँच सौ रुपये है"
"Your OTP is 483920"   ->  "Your OTP is four eight three nine two zero"
```

Two deliberate deviations from its `normalize_text()` entry point:

* **Case is preserved.** The library lowercases unconditionally; DhVaani's vocab
  contains both ASCII cases, so we call its stage functions directly and skip
  that step. `DHVAANI_TEXT__LOWERCASE=true` restores the old behaviour.
* **All 27 languages are accepted.** The library has profiles for 14. The other
  13 get a local *partial* pass — native digits mapped to ASCII, symbols
  expanded, whitespace collapsed — instead of a `KeyError`. `GET /v1/languages`
  reports the tier per language.

| tier | languages |
|---|---|
| full (14) | en hi bn mr kn te mai mag hne bho ta gu ml pa |
| partial (13) | as mni or ne sd kok sat brx ur sa doi ks raj |

Latency is kept off the critical path by an LRU cache keyed on
`(text, language, config)` — IVR traffic replays the same prompts constantly —
plus an optional thread pool so a pathological input cannot block the loop.

Urdu/Sindhi/Kashmiri get **both** Arabic-Indic digit ranges (U+0660 and U+06F0)
mapped; real text mixes them. ZWJ/ZWNJ are deliberately *not* stripped — they
are meaningful inside Indic conjuncts.

---

## 5. Voices

DhVaani clones from a reference clip **plus its transcript**. The transcript is
not optional and not cosmetic:

```python
features_lens = prompt_features_lens + ceil(
    prompt_features_lens / prompt_tokens_lens * tokens_lens / speed
)
```

There is no duration predictor. Generated length comes straight from the
prompt's frames-per-character. **A transcript half the true length makes the
voice speak at half speed.** The store warns when the implied rate falls outside
2–45 characters per second.

Everything expensive happens once, at creation: decode, resample to 24 kHz,
silence removal, trim, RMS normalise, mel extraction. The synthesis path only
reads a GPU-resident tensor. Upstream's own Triton runtime reaches the same
conclusion — its "speaker cache" cuts p50 latency by about a third at
concurrency 8.

Prompts are trimmed to `voice.max_prompt_seconds` (default 3 s), preferring a
quiet boundary so the clip does not end mid-phoneme. **Shorter prompts are
materially cheaper** — see the table in §3 — and 2 s clones about as well as 3 s
for most speakers.

### Voice API

```bash
# create
curl -X POST http://localhost:8000/v1/voices \
  -F file=@sample_files/simran.wav \
  -F voice_id=simran \
  -F transcript="नमस्ते, मैं वाणी बोल रही हूं" \
  -F language=hi -F name=Simran

# list / inspect / delete
curl http://localhost:8000/v1/voices
curl http://localhost:8000/v1/voices/simran
curl -X DELETE http://localhost:8000/v1/voices/simran

# preview -- fastest way to sanity-check a fresh clone
curl -X POST http://localhost:8000/v1/voices/simran/preview -o preview.wav
```

Voices persist as `<store_dir>/<id>.npz` + `.json`, written atomically.

---

## 6. APIs

### 6.1 WebSocket (production protocol — unchanged)

Byte-compatible with `flowtts/server.py`, so the existing fleet can be pointed
at DhVaani with no client change.

```jsonc
// client -> server
{"text": "...", "call_id": "...", "text_id": "...", "voice_id": "simran", "streaming": true}
{"type": "cancel", "text_id": "..."}

// server -> client: ONE frame = JSON header bytes + raw PCM, concatenated
{"type":"audio_chunk","call_id":"...","chunk_index":0,"sample_rate":24000,
 "encoding":"pcm_int16","wav_bytes":9600,"tokens":4800,"is_final":false,"cache_hit":false}<PCM>
{"type":"audio_done","chunks":4,"total_tokens":...,"llm_s":...,"decode_s":...,"rtf":...}
```

Legacy field names are preserved with mapped meanings:

| field | now means |
|---|---|
| `llm_s` | seconds in the flow decoder |
| `decode_s` | seconds in the vocoder |
| `tokens` / `total_tokens` | mel frames (93.75 per second) |
| `llm_ttft_ms` | ms to the first span's mel |
| `decoder_ttft_ms` | ms to the first PCM bytes (the real TTFB) |

The per-voice WAV cache fast path is preserved too — a sha256 hit under
`cached_data_<voice>/` bypasses the GPU entirely.

### 6.2 REST (OpenAI-compatible)

```bash
# non-streaming WAV
curl -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"dhvaani-0.5","input":"नमस्ते, आपका स्वागत है।","voice":"simran"}' \
  -o out.wav

# streaming PCM -- play as it arrives
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"...","voice":"simran","response_format":"pcm","stream":true}' \
  | ffplay -f s16le -ar 24000 -ac 1 -

# OpenAI SSE shape
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"...","voice":"simran","stream_format":"sse"}'

# telephony
-d '{"input":"...","voice":"simran","sample_rate":8000,"response_format":"pcm"}'
```

Works with the OpenAI SDK:

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
c.audio.speech.create(model="dhvaani-0.5", voice="simran",
                      input="नमस्ते").stream_to_file("out.wav")
```

Extensions beyond OpenAI's schema: `language`, `sample_rate`, `num_step`,
`guidance_scale`, `seed`, `stream`. `mp3`/`opus`/`aac`/`flac` are transcoded via
ffmpeg when present, and return a clear 400 when it is not — rather than
pretending.

A streamed WAV carries `0xFFFFFFFF` placeholder sizes (the length is not known
up front); non-streaming responses get a correct header.

**Client disconnects cancel the request.** At 200 RPS an abandoned stream that
keeps rendering is pure waste, so the handler polls `request.is_disconnected()`
and fires the cancel event into `FlowScheduler.cancel()`.

### 6.3 Control API

Same routes as `flowtts/server.py`: `POST /ports/add?port=N`, `GET /ports`,
`GET /ready`, `GET /health`, `GET /metrics`, `GET /ws/log`, `GET /ws/active`,
plus `GET /stats`.

---

## 7. Backends

`DHVAANI_BACKEND__KIND` or `--backend`:

| kind | when |
|---|---|
| `torch` | default. Always works, CUDA graphs on. The correctness reference. |
| `trt` | in-process TensorRT. Lowest TTFB. Needs prebuilt engines. |
| `triton` | NVIDIA Triton Inference Server client. See the caveat below. |

A backend that cannot start degrades to `torch` with a warning rather than
preventing boot, and one that cannot serve a *shape* (`supports_bucket()`)
falls back per-shape instead of crashing.

### Building TensorRT engines

```bash
python -m flowtts.dhvaani.setup.build_trt --max-batch 128
python -m flowtts.dhvaani.setup.build_trt --validate    # vs PyTorch, same inputs
```

The engine contract is **identical to upstream ZipVoice's exporter**
(`python -m zipvoice.bin.tensorrt_export`), so engines are interchangeable:

```
x (N,T,300) fp16 | t (N,) fp16 | padding_mask (N,T) fp16  ->  v (N,T,100) fp16
```

Two upstream tricks are essential and reproduced in `setup/build_trt.py`:

* `CompactRelPositionalEncoding.extend_pe` caches `self.pe` and returns early;
  under tracing that bakes one sequence length into the graph. It is patched to
  always recompute.
* `convert_scaled_to_non_scaled(..., is_onnx=True)` folds Balancer/Whiten/
  Dropout3 into identities, swaps the Swoosh activations for ONNX-expressible
  forms, and scripts the positional encoding.

The rest of the Zipformer is already export-safe — every stochastic branch is
guarded by `torch.jit.is_scripting() or torch.jit.is_tracing()` — which is why
`dynamo=False` (the TorchScript tracer) is the right exporter.

> **CFG doubles the engine batch.** DhVaani-0.5 is the non-distilled ZipVoice,
> so guidance is applied by doubling the batch. `--max-batch` defaults to
> `2 x engine.max_batch_size` for that reason.

### NVIDIA Triton Inference Server

Two different things, do not confuse them:

* `--backend triton` makes the **flow step** a Triton call. That puts an IPC hop
  on *every ODE iteration* — 8-16 per span. It exists for an existing Triton
  fleet that already hosts the engine; it will not beat in-process TRT.
* `triton/model_repository/` hosts the **whole pipeline** as a Triton model,
  which is the deployment most people want. Generate it with:

```bash
python -m flowtts.dhvaani.setup.build_triton_repo --profile balanced
```

This follows upstream's `runtime/nvidia_triton/` structure, with one
improvement: the model wraps `DhvaaniEngine`, so requests are continuous-batched
*inside* the model across Triton batches rather than running a Triton batch in
lockstep.

Use Triton when the GPU is shared with other models or you need its model
repository, versioning and control plane. If DhVaani owns the box, the
in-process server is faster.

---

## 8. Capacity — what it actually takes to reach 200 RPS

Be honest with this section. The cost model is exact enough to plan with, and
`bench.py capacity` reproduces it.

### The cost model

The flow decoder's U-net stacks are layers `[2,2,4,4,4]` at downsampling
`[1,2,4,2,1]`, so the work is ~`10 x frames` token-layers, each roughly
`12 x 512²` MACs. Empirically that is about **63 MFLOPs per frame per forward**.

```
span_flops = (prompt_frames + span_frames) x 63 MFLOP x num_step x (2 if CFG)
```

An L40S is ~181 TFLOPS dense FP16. At a realistic 45% MFU that is ~81 effective
TFLOPS.

### What that buys, for 3-second utterances

| config | prompt | GFLOP/req | max RPS |
|---|---|---|---|
| quality — 16 step, CFG | 3.0 s | 2872 | **28** |
| balanced — 8 step, CFG | 3.0 s | 1436 | **57** |
| balanced — 8 step, no CFG | 3.0 s | 718 | **113** |
| fast — 4 step, CFG | 3.0 s | 719 | **113** |
| fast — 4 step, no CFG | 3.0 s | 359 | **227** |
| fast — 4 step, no CFG | 2.0 s | 278 | **294** |

**Superseded by measurement — see the next section.** This table assumes the
model is compute-bound and reaches ~45% MFU. On real hardware it reaches about
4%, because it is bandwidth-bound, so achievable rates are roughly an order of
magnitude below what this predicts. The table is kept because the *relative*
costs it shows do hold exactly — `num_step` is linear, CFG is exactly 2x, prompt
length is proportional — and those are what you tune against.

Levers, in order of effect:

1. **`num_step`** — exactly linear. 16 -> 8 halves the cost; 8 -> 4 halves it again.
2. **CFG** — exactly 2x. `guidance_scale=0` halves the cost outright.
   `cfg_until_t=0.5` is the middle ground: guidance where it matters, at ~1.5x.
3. **Prompt length** — 3 s -> 2 s is roughly a 20% win, at little quality cost.
4. **FP8** — Ada supports it; roughly doubles peak. Pass
   `--peak-tflops 362` to the capacity model to see the effect.
5. **The WAV cache** — for repetitive IVR prompts, a cache hit costs nothing at
   all. This is often the largest real-world win.

### Time-to-first-byte

TTFB depends only on the first span (~1.2 s of audio, so ~300 frames with a 2 s
prompt), not on total length. Budget at `balanced`:

```
normalise (cached)      < 0.1 ms
tokenise                < 0.5 ms
text encode (batched)   ~ 1 ms
8 ODE steps @ 300 frames  dominant
vocoder                 ~ 3-8 ms
```

For reference, upstream's Triton benchmark on an L20 reports ~99 ms for a whole
short utterance at 4 steps with the distilled model, and ~89 ms with a speaker
cache. Our first span is smaller than a whole utterance, so **sub-200 ms TTFB is
comfortably reachable** — the pressure at 200 RPS is throughput, not latency.

### Measured on a real L40S (2026-08-26)

The analytic model above is a **compute** roofline, and it turns out to be the
wrong bound for this architecture. Measured on an L40S:

| batch x frames | ms/step | implied TFLOPS | % of 181 TFLOPS |
|---|---|---|---|
| 1 x 384 | 6.9 | 5.8 | 3% |
| 32 x 384 | 101 | 7.6 | 4% |
| 32 x 512 | 158 | 6.5 | 4% |

**4% of peak, and it barely improves with batch size.** The Zipformer is not
compute-bound here -- it is bandwidth-bound. Its attention scores are
`O(batch x frames^2)`, and at 512 frames with 4 heads that is tens of megabytes
per score tensor, read and written across 16 layers and two attention blocks
each. The 122M parameters are almost irrelevant next to the activation traffic.

So scale `num_step`, CFG and prompt length -- which cut *work* -- and do not
expect a bigger batch to rescue throughput the way it would for an LLM.

End-to-end, REST streaming, ~3 s mean utterance, `fast` profile
(`num_step=4`, CFG off), on a **shared** L40S (a training job held 15 GiB and
bursty CPU alongside it):

| target RPS | achieved | TTFB p50 | TTFB p99 | x realtime | errors |
|---|---|---|---|---|---|
| 2 | 1.7 | 96 ms | 225 ms | 4.1x | 0 |
| 4 | 3.9 | 112 ms | 357 ms | 9.3x | 0 |
| 6 | 5.7 | 172 ms | 457 ms | 13.5x | 0 |
| **8** | **7.7** | **207 ms** | **503 ms** | **18.3x** | **0** |
| 12 | 11.5 | 1254 ms | 2449 ms | 26.0x | 0 |
| 16 | 14.8 | 2010 ms | 4445 ms | 33.3x | 0 |
| 50 | 23.1 | 20157 ms | 32427 ms | 51.4x | 0 |

Read that as: **~8 RPS while holding TTFB near 200 ms, ~23 RPS if latency does
not matter.** The `balanced` profile does 4x the work and tops out near 5 RPS.

Unloaded single-request latency: TTFB 93-242 ms, RTF 0.12-0.23.

**200 RPS is therefore not reachable on one L40S for this model.** At the knee
that is roughly 25 cards, or ~9 at the saturated throughput point. The realistic
routes to it are: distil the model (upstream ZipVoice-Distill runs 4 steps with
guidance folded in), cut the prompt to ~1.5 s with a matching transcript (the
prompt is re-rendered on every span and is over half the frames on a short one),
serve the WAV cache in front for repeated IVR prompts, or shard across GPUs.

### Measure, do not trust the model

```bash
python -m flowtts.dhvaani.test.bench step          # achieved TFLOPS vs roofline
python -m flowtts.dhvaani.test.bench capacity --mfu <measured>
python -m flowtts.dhvaani.test.bench throughput --max-concurrency 128
python -m flowtts.dhvaani.test.loadtest ws --url ws://localhost:8080 --rps 200 --duration 60
```

`bench step` prints achieved TFLOPS per `(bucket, batch)` against the analytic
roofline, so you can see how far from peak you actually are and re-run the
capacity model with a real MFU.

---

## 9. Configuration reference

All settings take `DHVAANI_` env vars with `__` nesting:
`DHVAANI_FLOW__NUM_STEP=4`, `DHVAANI_ENGINE__MAX_BATCH_SIZE=96`.

### The ones that matter

| setting | default | effect |
|---|---|---|
| `flow.num_step` | 8 | **Linear** in cost. 16 = model-card quality, 4 = aggressive. |
| `flow.guidance_scale` | 1.0 | Non-zero **doubles** flow FLOPs. |
| `flow.cfg_until_t` | 1.0 | Skip CFG past this t. 0.5 = guidance on the low-t half. |
| `flow.t_shift` | 0.5 | Timestep grid skew. Upstream's inference default. |
| `voice.max_prompt_seconds` | 3.0 | Prompt frames are re-paid on every span. |
| `chunk.first_chunk_seconds` | 1.2 | Directly sets TTFB. |
| `chunk.steady_chunk_seconds` | 4.5 | Larger = better throughput, coarser cancellation. |
| `engine.max_batch_size` | 64 | Rows per forward pass (2x under CFG). |
| `engine.max_active_streams` | 192 | Primary VRAM knob. |
| `engine.chunk_lookahead` | 2 | Spans of one request in flight at once. |
| `memory.arena_vram_fraction` | 0.45 | Share of VRAM the arenas may hold. |
| `audio.output_sample_rate` | 24000 | 8000/16000 resample on GPU for telephony. |
| `audio.crossfade_seconds` | 0.06 | Span seam blend. 0 disables. |
| `backend.kind` | torch | torch / trt / triton. |

### Profiles

```bash
--profile fast       # num_step=4,  CFG off
--profile balanced   # num_step=8,  CFG to t=0.5   (default)
--profile quality    # num_step=16, CFG on
```

Explicit env vars always beat a profile.

---

## 10. Metrics and alerting

Prometheus on `/metrics` (REST port, WS port, and control port). Namespace
`dhvaani_*`; the legacy `tts_*` series still work.

| metric | watch for |
|---|---|
| `dhvaani_ttfb_ms` | p99 above your target — first span too long, or queueing |
| `dhvaani_queue_depth` | sustained > 0 means the GPU cannot keep up |
| `dhvaani_mean_batch` | **low under load is the red flag** — batching is not working |
| `dhvaani_steps_per_second` | flow throughput; compare against `bench step` |
| `dhvaani_vram_reserved_bytes` - `allocated` | rising = fragmentation |
| `dhvaani_gc_collections_total` | frequent = arenas are sized wrong |
| `dhvaani_arena_occupancy{bucket}` | one bucket pinned at capacity = retune buckets |
| `dhvaani_rtf` | > 1 means slower than real time |
| `dhvaani_errors_total` | any sustained rate |

Histogram buckets are milliseconds and dense across 10–500 ms; Prometheus'
defaults are useless at this scale.

### Tuning playbook for a TTFB target

1. `bench latency` — where does the time go?
2. If **flow** dominates: lower `num_step`, or set `cfg_until_t=0.5`.
3. If **queue** dominates: you are throughput-bound, not latency-bound. Go to §8.
4. If **TTFB** is fine at concurrency 1 but bad under load: check
   `dhvaani_mean_batch`. Low means spans are not coalescing — raise
   `engine.max_batch_size` or `engine.batch_fill_wait_us`.
5. Shorten `chunk.first_chunk_seconds` — it maps almost linearly to TTFB.
6. Shorten the voice prompt.

---

## 11. Troubleshooting

**"gated repository" / 401 on startup.** Accept the terms at
<https://huggingface.co/ARTPARK-IISc/DhVaani-0.5> while signed in, then
`export HF_TOKEN=...` and run `python -m flowtts.dhvaani.setup.fetch_model`.

**A voice speaks too slowly or too fast.** Its transcript does not match its
audio. The duration model is literally `prompt_frames / prompt_tokens`. Check
the `voice_transcript_rate_implausible` warning in the logs and recreate it.

**"span needs N frames, exceeding the largest bucket".** A span outgrew the
arena. Lower `chunk.steady_chunk_seconds` or `voice.max_prompt_seconds`, or
raise `buckets.max_frames`. The chunker normally prevents this.

**Audible clicks between spans.** Raise `audio.crossfade_seconds`; check
`audio.trim_edge_silence` is not cutting speech (`silence_threshold_db`).

**VRAM climbing.** Confirm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
check `dhvaani_vram_reserved_bytes - dhvaani_vram_allocated_bytes`, and see
whether CUDA graphs are churning (`backend_detail.cuda_graphs` in `/v1/stats`) —
too many distinct shapes evict graphs and re-reserve pools.

**TensorRT engine will not load a shape.** `supports_bucket()` returned False
and it fell back to torch — check the log for `trt_shape_uncovered` and rebuild
with a wider `--max-batch`.

**Code-switched text sounds rough at boundaries.** A model limitation: Latin is
in the vocab but code-switching was not explicitly trained.

---

## 12. Layout

```
flowtts/dhvaani/
├── config.py            all settings + profiles
├── types.py             dataclasses and protocols
├── server.py            entrypoint: engine + WS + REST + control
├── model/
│   ├── loader.py        gated snapshot -> ZipVoice + tokenizer + Vocos
│   ├── ops.py           vectorised, sync-free conditioning (tested vs upstream)
│   ├── text_encoder.py  token ids -> frame-rate conditions
│   ├── vocoder.py       batched Vocos with length-exact trimming
│   ├── triton_kernels.py fused ODE glue (OpenAI Triton) + torch fallbacks
│   └── export_patch.py  shape-dynamic rel-pos shift for ONNX/TRT
├── engine/
│   ├── scheduler.py     continuous-batching Euler ODE   <- the core
│   ├── arena.py         pre-allocated bucket slots
│   ├── memory.py        VRAM watchdog, admission control, OOM recovery
│   ├── vocode.py        micro-batched vocoder stage
│   ├── stitch.py        overlap-add across spans
│   └── engine.py        facade
├── text/                lang.py, normalizer.py, chunker.py
├── voices/              clone.py, store.py, registry.py
├── api/                 ws.py, rest.py, voices_api.py, app.py, control.py
├── monitoring/metrics.py
├── setup/               fetch_model.py, build_trt.py, build_triton_repo.py
└── test/                pytest suite + smoke.py, bench.py, loadtest.py
```

---

## 13. Credits

DhVaani-0.5 by ARTPARK-IISc, Apache-2.0, built on
[ZipVoice](https://github.com/k2-fsa/ZipVoice) (k2-fsa / Xiaomi). The TensorRT
export recipe and the Triton repository structure follow ZipVoice's
`runtime/nvidia_triton/` (NVIDIA, Apache-2.0). Text normalisation uses
[text_preprocessor_for_TTS](https://github.com/Ajaj-Ali/text_preprocessor_for_TTS)
(MIT). Please respect the licences of the training corpora (IndicTTS, Rasa,
IISc SYSPIN).
