# FlowTTS — Agent Context & Benchmark Analysis

## What FlowTTS Is Doing

FlowTTS is a **streaming Hindi TTS (Text-to-Speech) inference server** running on WebSocket port `8765`. It is used in a voice-bot context (Bajaj Finance IVR / collection calls) to synthesize Hindi speech from text in real time.

Each request lifecycle:
1. **RECV** — raw text arrives over WebSocket (truncated preview logged)
2. **IN** — full normalized text accepted for synthesis
3. **stream_done** — audio generation complete; metrics emitted

The server processes requests **concurrently** (many requests in flight simultaneously, as seen from burst arrivals at `17:07:01.813–17:07:01.841` with 70+ requests queued within 30 ms).

---

## Inference Pipeline (per request)

```
text → [LLM / flow model] → [acoustic decoder] → [WAV encoder] → streamed audio chunks
         llm_ms                 decode_ms            wav_enc_ms
```

- **LLM step**: the flow/language model generates token sequences (the bulk of latency)
- **Decoder**: converts tokens to mel/waveform frames
- **WAV encoder**: encodes to raw PCM/WAV bytes
- **total** = LLM step time (pipeline bottleneck; decoder runs in parallel/overlap)
- **chunks**: number of streaming audio chunks sent per request (proxy for text length)

---

## RTF — What It Means

**RTF (Real-Time Factor)** = `synthesis_time / audio_duration`

- RTF **< 1.0** → faster than real-time (good; audio is ready before it would finish playing)
- RTF **= 1.0** → exactly real-time
- RTF **> 1.0** → slower than real-time (system is falling behind)

---

## RTF Observations from `llm.log`

### Per-request RTF (300 stream_done events captured in live log)

| Metric | Value |
|--------|-------|
| Min RTF | 0.336 |
| Max RTF | 1.439 |
| Average RTF | **0.688** |
| Requests < 1.0 (real-time) | 86% (129/150) |
| Requests ≥ 1.0 (over budget) | 14% (21/150) |

### Running avg_rtf over time

The `avg_rtf` field is a cumulative rolling average logged per completion:

| Phase | avg_rtf |
|-------|---------|
| Early (req ~0–50) | 0.746 |
| Mid run (req ~100) | ~0.749–0.754 |
| End of run | **0.722** |

The average improved (decreased) over the course of the run, likely due to the batch size growing and GPU utilization becoming more efficient.

---

## Batching Observations

Requests are batched dynamically. The `chunks` field in `stream_done` lines reflects how many decode chunks were needed (correlates with text length and internal batch grouping).

### Chunk/batch distribution (150 completed requests):

| chunks | count | % of requests |
|--------|-------|---------------|
| 1 | 3 | 2% |
| 2 | 21 | 14% |
| 3 | 17 | 11% |
| 4 | 31 | 21% |
| 5 | 45 | 30% |
| 6 | 33 | 22% |

- Average chunks per request: **4.29**
- Most requests (73%) fall in the 4–6 chunk range, corresponding to medium-to-long Hindi sentences

### Batching behavior over time

At `17:07:01`, ~70 requests arrived nearly simultaneously (1 request every ~1–2 ms). These were processed together in batched inference waves, visible from the synchronized `stream_done` timestamps:

- `chunks=1` requests completed at `17:07:52.159–52.263` (~3 requests, ~100 ms spread)
- `chunks=2` requests completed at `17:07:52.337–52.345` (~20 requests, tight ~10 ms cluster)
- `chunks=3` requests completed at `17:07:53.350–53.713` (~15 requests, ~360 ms cluster)
- `chunks=4` requests completed at `17:07:54.425–54.604` (~19 requests, ~180 ms cluster)
- `chunks=5–6` requests completed at `17:07:55.x` (later wave)

This confirms the server batches by approximate token/chunk count — shorter texts complete together, longer texts complete in later synchronized waves.

---

## End-to-End Latency Summary (from benchmark footer)

```
total latency : min=0.576s  avg=3.032s  max=4.148s
time-to-first : min=0.524s  avg=0.594s  max=0.658s  (first audio chunk)
llm           : min=0.517s  avg=2.960s  max=4.073s
decoder       : min=0.212s  avg=2.134s  max=3.432s
llm - decode  : min=0.172s  avg=0.826s  max=1.377s  (net inference overhead)
```

**All 150 requests passed** (`✓ ALL PASSED`).

Key observations:
- **TTFF (time-to-first-audio) is very consistent**: 524–658 ms (avg 594 ms), meaning the first audio chunk starts streaming quickly regardless of total text length
- **Total latency scales with text length**: short texts (1 chunk) finish in ~0.6–1.1 s; long texts (6 chunks) take ~3.9–4.1 s
- **LLM step dominates**: accounts for ~98% of total latency (`total ≈ llm`); the decoder runs largely in parallel

---

## Request Profile

- **Mode**: `synth` (synthesis, not streaming ASR or other)
- **Server port**: 8765
- **Total requests in benchmark**: 150 (all passed)
- **Language**: Hindi (`hi`) — mix of Devanagari script and Hinglish (Hindi + English code-switch)
- **Use case**: Bajaj Finance IVR voice bot — sentences include EMI reminders, account numbers, loan notifications, KYC alerts

---

## What to Watch

| Signal | Threshold | Action |
|--------|-----------|--------|
| `rtf` per request | > 1.0 frequently (>20%) | Increase batch size or reduce concurrency |
| `avg_rtf` trending up | > 0.80 sustained | GPU may be saturating; profile batch queue depth |
| `ttff` | > 700 ms | First-chunk latency degrading; check LLM step |
| `llm` latency spike | > 4.5 s | Long-tail texts or GPU stall; check max token cap |
| `chunks=6` RTF | often > 1.0 | Long sentences need chunking/split upstream |
