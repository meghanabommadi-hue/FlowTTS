# Fish Audio S2 Pro on H200 — Acceleration Playbook

Target: **~200 RPS**, **TTFB < 200 ms**, realtime streaming, single NVIDIA H200.

Model recap: **fishaudio/s2-pro** is a **Dual-AR** TTS model — a Qwen3-4B "slow AR"
predicts one semantic token per ~21 Hz frame, and a 400M 4-layer "fast AR" fills the
remaining 9 residual codebooks per frame; an **EVA-GAN / RVQ codec (10 codebooks)**
decodes tokens → 24 kHz audio. Because the slow AR is structurally a standard
decoder-only LLM, it is served by **sglang-omni** with the full LLM-inference toolkit.

Fish's own published figures on a **single H200**: **RTF ≈ 0.195**, **TTFA < 100 ms**,
**3000+ acoustic tokens/s** while keeping RTF < 0.5 under concurrency. So the targets
are realistic and are delivered mostly by the **backend**, not the gateway.

Two facts drive everything below:
1. **Throughput comes from sglang's continuous batching + paged KV cache.** Concurrent
   requests are coalesced server-side; you scale RPS by feeding the backend more
   concurrency and giving it more KV-cache memory — not by batching in the gateway.
2. **Latency (TTFB) comes from AR streaming + prefix caching.** The first PCM frame
   ships after the first few frames decode (~100 ms); reusing a voice makes the
   reference prefill nearly free via **RadixAttention** (86–90% hit).

---

## TL;DR — recommended stack (do these, in order)

| # | Technique | Where | Effort | Expected win |
|---|-----------|-------|--------|--------------|
| 1 | **Stream with `response_format=pcm`, `stream=true`** (built-in) | gateway | done | TTFB → first-frame latency (~100 ms) |
| 2 | **Reuse voices → RadixAttention prefix cache** | backend | trivial | skips reference prefill (~86–90% hit) |
| 3 | **Raise `--tts-batch-max-items`** (32 → 48/64) | backend | trivial | throughput multiplier under load |
| 4 | **Give the KV cache more room: `--mem-fraction-static`** | backend | trivial | more concurrent streams before RTF climbs |
| 5 | **CUDA-graph capture + warmup** (sglang default) | backend | done | per-step latency; primed by the gateway warmup |
| 6 | **WAV cache** (built-in) | gateway | done | ∞ on repeated call-centre prompts (bypasses backend) |
| 7 | **`reference_mode=local` on a shared volume** | gateway | done | avoids re-uploading the ref clip every request |

Stack 1–7 is the pragmatic path. Everything else is fine-tuning.

---

## Backend levers (sglang-omni — the throughput/latency engine)

### B1. Batch size — `--tts-batch-max-items`
- **What:** cap on items sglang coalesces into one decode step. Higher ⇒ more
  concurrency absorbed ⇒ higher RPS, until compute/memory-bound.
- **How:** `TTS_BATCH_MAX_ITEMS=48` (or `64`) in `docker-compose.yml`. Sweep and watch
  RTF (want < 0.5 under load) and VRAM.

### B2. KV-cache memory — `--mem-fraction-static`
- **What:** fraction of GPU memory reserved for the paged KV cache. More cache ⇒ more
  concurrent streams before eviction/queueing. On a 141 GB H200 with a ~10 GB model
  there is lots of headroom.
- **How:** `MEM_FRACTION=0.85` in compose. Raise until VRAM is comfortably full.

### B3. Prefix caching — reuse voices
- **What:** the reference audio tokens live in the system prompt; sglang caches their
  KV in the Radix tree. Reusing a `voice_id` across requests **skips the reference
  prefill** (the expensive part of a cloned-voice request).
- **How:** nothing to configure — just reuse aliases. Call-centre workloads (a handful
  of agent voices) hit this constantly. Warmup primes the default voice.

### B4. CUDA graphs + warmup
- **What:** sglang captures CUDA graphs per shape; the first requests are slow while it
  compiles/captures. The gateway fires a warmup synthesis at startup to absorb this.
- **How:** default on. Keep the `start_period: 600s` healthcheck grace so first-run
  weight download + capture don't mark the backend unhealthy.

### B5. Model placement / weights
- **What:** point `FISH_MODEL` at a local weights path (or a licensed/fine-tuned
  checkpoint) instead of the HF repo to skip the download and control the version.
- **How:** `FISH_MODEL=/models/s2-pro` + mount it into the backend.

---

## Gateway levers (this repo)

### G1. Streaming PCM (built-in)
- The WS `streaming` path issues one `stream=true` call and forwards contiguous PCM.
  We do **not** crossfade between chunks (AR stream is continuous — see
  `server.py` `continuous_stream`). Smaller `FLOWTTS_FISH__INITIAL_CODEC_CHUNK_FRAMES`
  lowers TTFB at the cost of more, smaller frames.

### G2. WAV cache (built-in)
- `sha256(text).wav` per-voice cache dirs bypass the backend entirely on repeated
  prompts. Biggest real-world win for scripted IVR. See `_VOICE_CACHE_MAP` in
  `server.py` and `wav_cache_dir` in config.

### G3. Reference passing — `reference_mode`
- `local` (default): send the clip's path on the shared `voices/` volume — the backend
  reads it once, then the prefix cache takes over. Lowest overhead.
- `base64`: inline the clip as a data-URI — no shared volume needed, but re-uploads the
  clip each request (only the first same-voice call re-prefills, thanks to B3).
- If the backend can't read the shared volume (path mismatch), either set
  `FLOWTTS_FISH__BACKEND_VOICES_DIR` to the backend's mount point, or flip to `base64`.

### G4. Output sample rate (telephony)
- `FLOWTTS_OUTPUT__SAMPLE_RATE=8000|16000` resamples from native 24 kHz before
  encoding. Per-chunk linear resampling can add minor boundary artifacts on a
  streamed signal; default 24 kHz avoids it. Validate telephony output by ear.

### G5. Horizontal scale
- The gateway is CPU-cheap; the bottleneck is the single backend GPU. To exceed a
  single H200's ceiling, run more backend replicas behind a load balancer and point
  `FLOWTTS_FISH__BACKEND_URL` at it.

---

## Config → lever quick map

| Setting | Enables |
|---|---|
| `TTS_BATCH_MAX_ITEMS` (backend env) | B1 |
| `MEM_FRACTION` → `--mem-fraction-static` (backend env) | B2 |
| reuse `voice_id` | B3 (prefix cache) |
| `FISH_MODEL` (backend env) | B5 |
| `FLOWTTS_FISH__INITIAL_CODEC_CHUNK_FRAMES` | G1 (TTFB) |
| `wav_cache_dir` / `_VOICE_CACHE_MAP` | G2 |
| `FLOWTTS_FISH__REFERENCE_MODE`, `__BACKEND_VOICES_DIR` | G3 |
| `FLOWTTS_OUTPUT__SAMPLE_RATE` | G4 |
| `FLOWTTS_FISH__BACKEND_URL` | G5 |

## Suggested tuning order on the box
1. Backend up, one voice, `stream=true`: record TTFB p95 + single-stream RTF.
2. Load test with `test_pipeline --requests 200 --streaming`, one shared voice
   (exercises the prefix cache). Watch RPS, TTFB p95, `/metrics` RTF, `nvidia-smi`.
3. Raise `TTS_BATCH_MAX_ITEMS` (48/64) and `MEM_FRACTION` (0.85+); re-measure.
4. Shrink `INITIAL_CODEC_CHUNK_FRAMES` until TTFB < 200 ms without hurting prosody.
5. Lean on the WAV cache for repeated prompts; add backend replicas only if a single
   H200 saturates before 200 RPS.

## Honesty note
Fish's RTF/TTFA/tokens-per-second figures are published for S2 Pro on an H200, but the
exact **RPS** you get depends on utterance length, voice reuse, output rate, and the
sglang build. **Benchmark on your hardware** and tune from there.

## License
`fishaudio/s2-pro` weights are under the **Fish Audio Research License** — research /
non-commercial use is free; **commercial use requires a separate license** from Fish
Audio (`business@fish.audio`). Keep `FISH_MODEL` configurable to point at a licensed or
self-hosted checkpoint.

### Sources
- Fish Audio S2 technical report · sglang-omni fishaudio_s2_pro README + `docs/basic_usage/tts.md`
- SGLang: continuous batching, paged KV cache, CUDA graphs, RadixAttention prefix caching
