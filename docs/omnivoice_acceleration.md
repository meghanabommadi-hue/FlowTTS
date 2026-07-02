# OmniVoice on H200 — Acceleration Playbook

Target: **~200 RPS**, **TTFB < 200 ms**, realtime streaming, single NVIDIA H200.

Model recap: **k2-fsa/OmniVoice** is a non-autoregressive **masked/discrete-diffusion LM**
(Qwen3-0.6B bidirectional backbone) that iteratively **unmasks** an (8-codebook, T)
acoustic-token grid over `num_step` passes, using **classifier-free guidance (CFG)**
(`guidance_scale`), then the **Higgs-Audio-v2** neural codec decodes tokens → 24 kHz.
`generate(list[str])` is natively batched. Reported RTF ≈ 0.03 @ 16 steps, ≈ 0.022 batched.

Two facts drive everything below:
1. **Per-request cost ≈ `num_step × (1 + CFG) × backbone_forward(B, T) + codec_decode(B, T)`.**
   So the biggest levers are *fewer steps*, *killing the CFG 2× tax*, *bigger batches*, and
   *making each forward cheaper* (compile / FP8).
2. **It's a 0.6B model** — at batch=1 the H200 is massively underutilized. Throughput comes
   almost entirely from **batching**; latency comes from **steps × per-step cost**.

---

## TL;DR — recommended stack (do these, in order)

| # | Technique | Effort | Expected win | Toggle |
|---|-----------|--------|--------------|--------|
| 1 | **Lower `num_step` 16 → 8–12** | trivial | 1.3–2× latency & throughput | `FLOWTTS_OMNIVOICE__NUM_STEP` |
| 2 | **Dynamic length-bucketed batching** (already built) | done | the throughput multiplier (10–30×+ vs bs=1) | `MAX_BATCH`, `BATCH_TIMEOUT_MS`, `length_buckets` |
| 3 | **`torch.compile` backbone + codec** (already wired) | low | 1.3–2× per-forward | `--compile` / `COMPILE_MODEL` |
| 4 | **Short first chunk + pipelined streaming** (already built) | done | TTFB < 200 ms | `streaming.first_chunk_max_chars` |
| 5 | **Lower `guidance_scale` / early-steps-only CFG** | low–med | up to ~2× (kills the CFG tax) | `GUIDANCE_SCALE` (+ code) |
| 6 | **FP8 (torchao float8) on Hopper, with compile** | med | 1.3–1.6× per-forward | new `quantize` hook |
| — | **WAV cache** (already built) | done | ∞ on repeated call-centre prompts | `wav_cache_dir` |

Stack 1–4 + the WAV cache is the pragmatic path to the targets. 5–6 are the next
tier if the sweep falls short. Everything below #6 needs training or is higher-risk.

---

## Tier A — highest impact-per-effort (config only, ship first)

### A1. Fewer denoising steps (`num_step`)
- **What:** latency is ~linear in `num_step`; each step is one full backbone forward over the grid.
- **Win:** 16→12 ≈ 1.33×, 16→8 ≈ 2×, on both latency and throughput.
- **Effort/risk:** trivial / low — it's a runtime knob. **Quality:** degrades as steps drop; TTS
  tolerates fewer steps than images. Find the floor empirically (listen at 12, 10, 8).
- **How:** `FLOWTTS_OMNIVOICE__NUM_STEP=10` or `run.sh --num-step 10`.
- **Stage:** diffusion backbone.

### A2. Dynamic length-bucketed batching  ✅ already implemented
- **What:** coalesce concurrent requests **and** streamed chunks into one `generate([...])`;
  sort/bucket by length to minimize padding across the (8,T) grid. This is *the* throughput lever
  for a 0.6B model (bs=1 wastes the H200).
- **Win:** 10–30×+ throughput vs bs=1 (sublinear per-step cost growth until compute-bound).
- **Effort:** done — [`omnivoice_engine.py`](../flowtts/synthesis/omnivoice_engine.py) `_batch_loop`/`_generate_group`.
- **How to tune:** `FLOWTTS_OMNIVOICE__MAX_BATCH` (start 32; sweep 48/64/96),
  `FLOWTTS_OMNIVOICE__BATCH_TIMEOUT_MS` (5–10 ms), and `length_buckets`.
- **Stage:** serving. **Note:** bucketing also keeps the set of compiled/captured shapes small (helps A3).

### A3. `torch.compile` backbone + codec  ✅ already wired (opt-in)
- **What:** operator fusion + (with `reduce-overhead`) **CUDA-graph capture** per shape; `max-autotune`
  tunes kernels harder. Fixed-ish shapes from bucketing → graphs get reused.
- **Win:** ~1.3–2× per-forward on diffusion transformers ([PyTorch](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)); stacks with FP8.
- **Effort/risk:** low — `_maybe_compile()` already compiles the backbone + `audio_tokenizer` defensively.
  First run is slow (compile); warmup already triggers it. Dynamic shapes disable CUDA graphs, so keep
  buckets few. **Quality:** none.
- **How:** `run.sh --compile` (`FLOWTTS_OMNIVOICE__COMPILE_MODEL=true`), `COMPILE_MODE=reduce-overhead|max-autotune`.
- **Stage:** backbone + codec decode.

### A4. Short first chunk + pipelined streaming  ✅ already built
- **What:** split text so chunk-0 is tiny → its batched `generate()` returns in tens of ms →
  first PCM out < 200 ms; later chunks generate while chunk-0 plays. Overlap-fade at boundaries.
- **Win:** TTFB from (whole-utterance) → (first-clause) latency. Matches the async-chunk pattern used by
  production streaming TTS ([LMSYS MOSS-TTS](https://www.lmsys.org/blog/2026-06-17-moss-tts-local-v15),
  [block-wise streaming](https://arxiv.org/pdf/2604.12438)).
- **Effort:** done — [`text_chunker.py`](../flowtts/synthesis/text_chunker.py) + [`models.py`](../flowtts/synthesis/models.py) `synthesize_stream`.
- **How:** `FLOWTTS_STREAMING__FIRST_CHUNK_MAX_CHARS` (smaller = lower TTFB, more boundaries).
- **Stage:** serving.

---

## Tier B — strong wins, a little code

### B1. Cut the classifier-free-guidance 2× tax
CFG runs **conditional + unconditional** each step (≈2× backbone compute). Options, cheapest first:
- **Lower `guidance_scale`** (e.g. 2.0 → 1.3–1.5): CFG is still batched cond+uncond, so this
  doesn't remove the 2× by itself, but weaker guidance lets you drop steps further (compounds with A1).
  `FLOWTTS_OMNIVOICE__GUIDANCE_SCALE=1.3`. **Quality:** less prompt adherence; test.
- **CFG only on early steps** (interval guidance): apply CFG for the first ~40% of steps, skip it after
  → approaches 1× compute on the tail. Needs a small change in OmniVoice's unmask loop (guard the
  uncond pass by step index). Medium effort; no training.
- **Guidance distillation / Adapter Guidance Distillation (AGD)** — single forward pass reproduces CFG,
  ~2× sampling speed, trains only ~2% params ([arXiv:2503.07274](https://arxiv.org/abs/2503.07274)).
  Best end-state but **requires a distillation run**.
- **Win:** up to ~2×. **Stage:** backbone. **Where:** `generate(guidance_scale=...)` already plumbed via
  `OmniVoiceGenerationConfig`; interval-CFG/AGD would live in the OmniVoice generator, not our server.

### B2. Confidence-based parallel unmasking → fewer effective steps
- **What:** masked-diffusion LMs can unmask **many positions per step** by confidence, instead of a
  fixed schedule — fewer network evaluations for the same quality. See LLaDA low-confidence remasking,
  **adaptive parallel decoding** ([UCLA](https://starai.cs.ucla.edu/papers/IsraelNeurIPS25.pdf)),
  **KLASS** ([arXiv:2511.05664](https://arxiv.org/pdf/2511.05664)), confidence-aware calibration
  ([arXiv:2512.07173](https://arxiv.org/pdf/2512.07173)).
- **Win:** 1.5–3× fewer steps at equal quality (task-dependent).
- **Effort/risk:** medium-high — depends on whether OmniVoice exposes/uses a confidence schedule
  (`position_temperature`, `layer_penalty_factor` already influence unmask order). Start by tuning those;
  a custom schedule means forking the generator loop. **Quality:** neutral-to-better if tuned.
- **Stage:** backbone. **Where:** tune `POSITION_TEMPERATURE` / `LAYER_PENALTY_FACTOR` first (free);
  custom schedule = upstream generator change.

### B3. FP8 (torchao float8) on Hopper
- **What:** dynamic float8 matmuls on H200 (CC 9.0). **Must** pair with `torch.compile` or FP8 is *slower*.
- **Win:** ~1.3–1.6× per-forward ([torchao](https://github.com/pytorch/ao),
  [PyTorch LLM inference](https://pytorch.org/blog/accelerating-llm-inference/)); stacks with A3.
- **Effort/risk:** medium — add a quantize step at load. **Quality:** small; validate MOS/similarity.
- **How (sketch), in `_maybe_compile()`/after load:**
  ```python
  from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
  quantize_(self.model.model, Float8DynamicActivationFloat8WeightConfig())  # backbone only
  # then torch.compile(...) as already done
  ```
  Add a `FLOWTTS_OMNIVOICE__QUANTIZE=float8` toggle guarding it. **Stage:** backbone (try codec too).

### B4. Codec-decode overlap on a separate CUDA stream  ⚠ partially wired
- **What:** run the Higgs codec decode on its own stream so it overlaps the *next* batch's diffusion;
  capture short chunks with CUDA graphs; keep codec state buffers at stable addresses for graph replay
  (the pattern behind 50 ms-latency streaming TTS,
  [dev.to Qwen3-TTS](https://dev.to/jayanthkumarmorem/i-made-a-single-cuda-kernel-speak-streaming-qwen3-tts-at-50ms-latency-on-an-rtx-5090-53if)).
- **Win:** hides codec latency behind compute; smoother streaming, lower TTFB tail.
- **Effort/risk:** medium — `overlap_codec_decode` flag exists, but `generate()` fuses decode internally,
  so true overlap needs calling the codec decode separately from the unmask loop (upstream hook) or
  running codec on a `torch.cuda.Stream()`. **Quality:** none.
- **Stage:** codec decode. **Where:** `FLOWTTS_OMNIVOICE__OVERLAP_CODEC_DECODE` + engine change.

---

## Tier C — biggest ceilings, but need training or upstream work

### C1. Few-step distillation (consistency / distribution-matching / MeanFlow)
- **What:** distill the diffusion sampler to **1–4 steps**. E1-TTS (1-step DMD),
  [IntMeanFlow](https://arxiv.org/pdf/2510.07979) (1–3 NFE, ~10× RTF), StyleTTS-ZS (~90% faster),
  [NVIDIA TTS distillation](https://developer.nvidia.com/blog/speeding-up-text-to-speech-diffusion-models-by-distillation/)
  (progressive + CFG distill, 5× no quality regression).
- **Win:** the single largest lever — up to ~10× — but **requires a distillation training run** on OmniVoice.
- **Effort/risk:** high. **Quality:** small if done well. **Stage:** backbone. Track as a project, not a toggle.

### C2. TensorRT / TRT-LLM export
- **What:** export the Qwen3-0.6B backbone (and/or codec) to TensorRT.
- **Win:** ~2× on diffusion inference ([NVIDIA Torch-TensorRT](https://developer.nvidia.com/blog/double-pytorch-inference-speed-for-diffusion-models-using-torch-tensorrt/)),
  can exceed torch.compile.
- **Effort/risk:** high — dynamic shapes (variable T, batch) are painful; build per-bucket engines.
  Do only after A3/B3 plateau. **Stage:** backbone + codec.

### C3. Feature caching across steps (TeaCache-style)  ⚠ verify applicability
- **What:** skip redundant compute when consecutive-timestep activations are similar; 1.5–2× on DiTs,
  training-free ([TeaCache](https://liewfeng.github.io/TeaCache/)). **vllm-omni ships TeaCache**
  ([docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_acceleration/teacache/)).
- **Caveat:** designed for continuous-noise DiTs; on **masked**-diffusion LMs the token grid changes
  discretely each unmask step, so temporal-coherence caching may not transfer cleanly — **validate before trusting.**
- **Effort/risk:** medium/uncertain. **Quality:** small if it applies. **Stage:** backbone.

### C4. Multi-replica / MPS (horizontal scale)
- **What:** H200 has ~140 GB HBM3e; a 0.6B fp16 model is ~1.5 GB. Run **N engine replicas** (separate
  processes, or one process + multiple CUDA streams) behind the existing nginx to multiply throughput
  if a single batched engine saturates before 200 RPS.
- **Win:** ~linear in replicas until memory-bandwidth/SM-bound. **Effort:** low-med (process mgmt).
- **Risk:** replicas contend for the same SMs — batching within one engine is usually better first.
- **Stage:** serving. Tensor parallelism is **not** worth it for 0.6B.

---

## Config → technique quick map

| Setting (`FLOWTTS_OMNIVOICE__…` / file) | Enables |
|---|---|
| `NUM_STEP` | A1 |
| `MAX_BATCH`, `BATCH_TIMEOUT_MS`, `length_buckets` | A2 |
| `COMPILE_MODEL`, `COMPILE_MODE` | A3 |
| `streaming.first_chunk_max_chars`, `chunk_max_chars` | A4 |
| `GUIDANCE_SCALE` (+ interval-CFG code) | B1 |
| `POSITION_TEMPERATURE`, `LAYER_PENALTY_FACTOR` | B2 (tuning) |
| new `QUANTIZE=float8` hook in `_maybe_compile()` | B3 |
| `OVERLAP_CODEC_DECODE` (+ engine change) | B4 |
| `wav_cache_dir` | WAV cache |

## Suggested tuning order on the box
1. Baseline: `num_step=16`, `max_batch=32`, no compile → record RPS + TTFB p95.
2. Add `--compile` (A3); re-measure after warmup.
3. Sweep `num_step` 12 → 10 → 8 (A1); listen for quality floor.
4. Sweep `max_batch` 48/64/96 and `batch_timeout_ms` 5–10 (A2).
5. Shrink `first_chunk_max_chars` until TTFB < 200 ms without hurting prosody (A4).
6. If short of 200 RPS: lower `guidance_scale` (B1), add FP8 (B3), then replicas (C4).
7. Lean on the WAV cache for the (many) repeated call-centre prompts.

## Honesty note
No public absolute RPS/TTFB numbers exist for OmniVoice specifically; the above speedup ranges are
from adjacent diffusion/TTS/LLM work and must be **validated on your H200**. Tiers A + WAV cache are
low-risk and should be measured first; Tier C items are project-sized (training/engine builds).

---

### Sources
- Few-step / distillation TTS: [IntMeanFlow](https://arxiv.org/pdf/2510.07979) · [Single-stage masked TTS](https://arxiv.org/pdf/2409.11003) · [StyleTTS-ZS](https://arxiv.org/html/2409.10058v1) · [NVIDIA TTS distillation](https://developer.nvidia.com/blog/speeding-up-text-to-speech-diffusion-models-by-distillation/) · [Infinite Mask Diffusion few-step](https://arxiv.org/pdf/2605.10518)
- CFG cost: [Adapter Guidance Distillation (AGD)](https://arxiv.org/abs/2503.07274) · [Diffusion without CFG](https://arxiv.org/html/2502.12154v1)
- torch.compile / CUDA graphs: [PyTorch diffusers guide](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/) · [Torch-TensorRT 2× diffusion](https://developer.nvidia.com/blog/double-pytorch-inference-speed-for-diffusion-models-using-torch-tensorrt/)
- FP8 / torchao: [pytorch/ao](https://github.com/pytorch/ao) · [Accelerating LLM inference](https://pytorch.org/blog/accelerating-llm-inference/) · [Transformer Engine on H100/H200](https://www.spheron.network/blog/nvidia-transformer-engine-h100-h200-fp8/)
- Parallel unmasking: [LLaDA](https://mlhonk.substack.com/p/10-llada-large-language-diffusion) · [Adaptive Parallel Decoding](https://starai.cs.ucla.edu/papers/IsraelNeurIPS25.pdf) · [KLASS](https://arxiv.org/pdf/2511.05664) · [Confidence-aware calibration](https://arxiv.org/pdf/2512.07173)
- Caching: [TeaCache](https://liewfeng.github.io/TeaCache/) · [vllm-omni TeaCache](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_acceleration/teacache/)
- Streaming/codec overlap: [Block-wise streaming synthesis](https://arxiv.org/pdf/2604.12438) · [MOSS-TTS on SGLang-Omni](https://www.lmsys.org/blog/2026-06-17-moss-tts-local-v15) · [Qwen3-TTS 50 ms streaming](https://dev.to/jayanthkumarmorem/i-made-a-single-cuda-kernel-speak-streaming-qwen3-tts-at-50ms-latency-on-an-rtx-5090-53if)
</content>
