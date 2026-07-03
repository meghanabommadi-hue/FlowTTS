# FlowTTS — Full Design & Data Flow

> **Note (current architecture):** the primary production path is the single-process
> gateway ([`flowtts/server.py`](flowtts/server.py)) proxying to a **Fish Audio S2 Pro**
> model served by **sglang-omni** — see [README.md](README.md). The Redis-backed,
> decoder-split flow documented below is the **secondary** multi-process path; where it
> says "Worker runs the TTS model / audio tokens", read it as "gateway/worker calls the
> sglang backend, which returns decoded PCM" (S2 Pro streams decoded audio — there is no
> separate token→PCM decoder stage to run).


**High-level flow:**  
**Client** → **Gateway (FastAPI + WebSocket)** → **Redis** → **Speech Synthesizer Worker** → **Redis** (same instance) → **Decoder (per-call, GPU-bound)** → **Gateway** → **Client**

Text (and optional controls) enters via the Gateway; synthesis runs on a separate worker and produces **audio encoded tokens**; a per-call decoder instance (on a GPU distinct from the TTS model GPU) consumes tokens from Redis, decodes them to audio, performs post-processing/resampling, and the Gateway streams audio back to the client.

---

## 1. End-to-end data flow (Mermaid)

```mermaid
flowchart LR
    subgraph Client["🖥️ Client"]
        WS_CLIENT[WebSocket Client]
    end

    subgraph Gateway["📡 Gateway (FastAPI + WebSocket)"]
        WS_EP["/ws/{call_id}"]
        CONN[ConnectionManager]
        PUB_REQ[publish_synthesis_request]
        LISTEN[_listen_for_audio]
        SEND_AUDIO[send_audio_to_client]
    end

    subgraph Redis["📦 Redis"]
        TTS_QUEUE["tts_queue (list)"]
        AUDIO_CH["audio:{call_id} (pub/sub)"]
    end

    subgraph Worker["🔊 Speech Synthesizer Worker (model GPU)"]
        BLPOP[BLPOP tts_queue]
        JOB[process_synthesis_job]
        SYNTH[TTS Engine / Model → audio tokens]
        PUB_AUDIO[publish_audio_token_chunks]
    end

    subgraph DecoderSide["📥 Decoder (per call_id, decoder GPU)"]
        SUB[SUBSCRIBE audio:{call_id}]
        BUF[TokenBufferManager]
        DECODE[Decoder + post-processing/resampling]
        PUSH_GW[push decoded audio to Gateway]
    end

    WS_CLIENT <-->|"text, voice, options"| WS_EP
    WS_EP --> CONN
    CONN --> PUB_REQ
    PUB_REQ --> TTS_QUEUE

    TTS_QUEUE --> BLPOP
    BLPOP --> JOB
    JOB --> SYNTH
    SYNTH --> PUB_AUDIO
    PUB_AUDIO --> AUDIO_CH

    AUDIO_CH --> SUB
    SUB --> BUF
    BUF --> DECODE
    DECODE --> PUSH_GW
    PUSH_GW --> LISTEN
    LISTEN --> SEND_AUDIO
    SEND_AUDIO --> WS_CLIENT
```

**Vertical variant (same flow, top-to-bottom):**

```mermaid
flowchart TB
    subgraph Client["🖥️ Client"]
        WS_CLIENT[WebSocket Client]
    end

    subgraph Gateway["📡 Gateway"]
        WS_EP["WebSocket /ws/{call_id}"]
        CONN[ConnectionManager]
        PUB_REQ[Publish synthesis job]
        LISTEN[Subscribe to audio channel]
        SEND_AUDIO[Send audio frames to client]
    end

    subgraph Redis["📦 Redis"]
        TTS_QUEUE["tts_queue"]
        AUDIO_CH["audio:{call_id}"]
    end

    subgraph Worker["🔊 Speech Synthesizer Worker (model GPU)"]
        BLPOP[BLPOP tts_queue]
        JOB[Process job: text → audio tokens]
        SYNTH[TTS Model → tokens]
        PUB_AUDIO[PUBLISH token chunks]
    end

    subgraph Decoder["📥 Decoder (per call_id, decoder GPU)"]
        SUB[SUBSCRIBE audio:{call_id}]
        BUF[TokenBuffer / reorder chunks]
        DEC[Decode tokens → PCM + resample + crossfade]
        TO_GW[Deliver decoded audio to Gateway]
    end

    WS_CLIENT <--> WS_EP
    WS_EP --> CONN
    CONN --> PUB_REQ --> TTS_QUEUE
    BLPOP --> JOB --> SYNTH --> PUB_AUDIO --> AUDIO_CH
    AUDIO_CH --> SUB --> BUF --> DEC --> TO_GW --> LISTEN --> SEND_AUDIO --> WS_CLIENT
```

---

## 2. Flow by stage

### 2.1 Client → Gateway

- Client connects to **WebSocket** e.g. `ws://host/ws/{call_id}` (optional: `?voice=...&lang=...`).
- **ConnectionManager** (Gateway):
  - Accepts connection and binds it to a **single `call_id`** (one live call, one WebSocket).
  - Starts a **result listener** that continuously receives audio results for this call (subscribes to Redis channel `audio:{call_id}`), just like the transcriber.
- Over the lifetime of a call, the same WebSocket can carry **multiple synthesis requests** (multiple `text_id`s) for that `call_id`:
  - Each `text_id` maps to a separate request/response via Redis, but all reuse the same socket.
- Client sends messages, e.g.:
  - `{ "type": "synthesize", "text": "...", "voice_id": "...", "language": "..." }`
  - Or streamed text chunks; Gateway may batch or forward per chunk.

### 2.2 Gateway → Redis (enqueue)

- For each synthesis request (or batched request), Gateway builds a **job**:
  - `call_id`, `text_id`, `text`, `voice_id`, `language`, `options`, `published_at`, etc.
- Gateway **RPUSH**es the job to Redis list **tts_queue** (or `flowtts:tts_queue`).
- Same Gateway process (or a dedicated listener) is **subscribed** to `audio:{call_id}` so it can receive audio back.

### 2.3 Redis → Speech Synthesizer Worker

- One or more **Speech Synthesizer Worker** processes run a loop:
  - **BLPOP** on **tts_queue** (with timeout).
  - On job received: parse JSON, run **TTS engine** (e.g. neural TTS model) on `text` with `voice_id` / `language`.
  - TTS engine produces **audio encoded tokens only** (e.g. neural codec / vocoder tokens), never raw PCM.
  - Worker continuously streams **multiple audio token chunks** for each `text_id` as they are generated (low-latency streaming).

### 2.4 Worker → Redis (publish audio tokens)

- Worker **PUBLISH**es to Redis channel **audio:{call_id}**:
  - Payload (aligned with LITranscriber-style IDs and diagnostics) can include:
    - `type`: `"transcript"`
    - `call_id`, `text_id`, `chunk_id`
    - `text`, `confidence`, `start_time`, `end_time`, `rtf`, `diagnostics`, `is_final`
    - `audio_tokens`: encoded audio token payload for this chunk
  - Multiple messages per `text_id` if streaming; `is_final` marks end of that logical utterance.

### 2.5 Redis → Decoder (required) + processing

- A **Decoder instance is always required** for each `call_id`:
  - It **SUBSCRIBE**s to **`audio:{call_id}`** (one subscription per active call).
  - Receives streaming audio token messages, **buffers** and **reorders** by `text_id` / `chunk_id` if needed.
  - Uses the decoder/vocoder to convert **audio encoded tokens → PCM**.
  - Runs the `processing/` pipeline (resampling, crossfading, other output post‑processing).
  - Delivers ordered PCM (or final target format) to the Gateway for that call (in‑process queue, shared memory, or internal API).

### 2.6 Decoder → Gateway → Client

- Gateway holds the WebSocket per `call_id`. It receives decoded, processed audio from the per‑call decoder instance and **sends it to the client** over the WebSocket (binary frames or base64 PCM).
- When the client disconnects for a `call_id`, the WebSocket and associated resources (including the decoder instance) are cleaned up so no ports/GPUs remain in use.
---

## 3. Component summary

| Component | Role |
|-----------|------|
| **Client** | Opens WebSocket, sends text (and options), receives audio frames. |
| **Gateway** | FastAPI app; WebSocket endpoint; ConnectionManager; publish jobs to Redis; subscribe to `audio:{call_id}`; buffer/decoder delivers audio → Gateway sends to client. |
| **Redis** | **List** `tts_queue`: job queue (Gateway → Worker). **Pub/Sub** `audio:{call_id}`: Worker → Buffer/Decoder (or Gateway). |
| **Speech Synthesizer Worker** | BLPOP jobs, run TTS model, PUBLISH audio chunks to `audio:{call_id}`. |
| **Buffer Manager** | Subscribe to `audio:{call_id}`; buffer and reorder chunks; optional decode. |
| **Decoder** | Convert encoded audio to PCM (or target format) before handing to Gateway. |

---

## 4. Suggested directory layout (FlowTTS, with `processing/` and decoder GPU split)

```
flowtts/
├── __init__.py
├── main.py                  # FastAPI app, lifespan, mount routers
├── core/
│   ├── __init__.py
│   └── config.py            # Redis, queue/channel names, model GPUs vs decoder GPUs, TTS settings
├── api/
│   ├── __init__.py
│   ├── models.py            # Pydantic: SynthesizeRequest, TtsChunkMessage (mirrors TranscriptMessage + audio_tokens)
│   └── websockets.py        # WebSocket endpoint, ConnectionManager, enqueue jobs, stream PCM back by call_id
├── synthesis/               # Text → audio encoded tokens (model GPUs)
│   ├── __init__.py
│   ├── engine.py            # TTS engine wrapper: synthesize(text, voice) → audio_tokens
│   └── models.py            # Model-specific code (e.g. Qwen, Mira, etc.)
├── decoder/                 # Tokens → PCM + teardown logic (decoder GPUs)
│   ├── __init__.py
│   ├── manager.py           # Per-call DecoderInstance lifecycle, GPU id assignment, kill on WS disconnect
│   └── decoder.py           # Decode audio_tokens → PCM (vocoder), call processing pipeline
├── processing/              # Output audio processing
│   ├── __init__.py
│   ├── resample.py          # Resampling utilities
│   ├── crossfade.py         # Crossfading between chunks
│   └── pipeline.py          # Orchestrate resample + crossfade + other effects
├── worker.py                # BLPOP tts_queue → synthesis.engine → publish token chunks to audio:{call_id}
└── monitoring/
    ├── __init__.py
    ├── logging.py
    └── metrics.py
```

- **Gateway**: `main.py` + `api/websockets.py`; holds WebSockets per `call_id`, and uses `decoder.manager` outputs to stream PCM to clients.
- **Speech Synthesizer Worker**: `worker.py` + `synthesis/engine.py` on **model GPUs**; only produces **audio encoded tokens**.
- **Decoder instances**: `decoder/` + `processing/` on **decoder GPUs**; per-`call_id` instances subscribe to `audio:{call_id}`, decode tokens, run resampling/crossfading, and are **killed when the WebSocket for that `call_id` disconnects**.

---

## 5. Terminal-style one-page flow

```
Client
  │
  │ WebSocket: { "type": "synthesize", "text": "...", "voice_id": "...", ... }
  ▼
Gateway ─────────────────────────────────────────────────────────────────
  │
  │ ConnectionManager
  │   • Accept WebSocket for call_id
  │   • Subscribe to audio:{call_id} (listen for audio from Redis)
  │   • On "synthesize" message → build job → RPUSH tts_queue
  │
  ▼
Redis
  │ tts_queue (list) ◄── job { call_id, text_id, text, voice_id, ... }
  ▼
Speech Synthesizer Worker ───────────────────────────────────────────────
  │ BLPOP tts_queue
  │ process_synthesis_job():
  │   • TTS engine: text (+ voice, lang) → audio (PCM or encoded)
  │   • PUBLISH to audio:{call_id} (chunk(s) + metadata, is_final)
  ▼
Redis audio:{call_id}
  │ audio:{call_id} (pub/sub) ──► audio chunks (Worker → Buffer/Decoder)
  ▼
Buffer Manager + Decoder ────────────────────────────────────────────────
  │ SUBSCRIBE audio:{call_id}
  │ Buffer: reorder chunks by text_id / chunk_index
  │ Decoder: optional decode to PCM
  │ Deliver ordered audio to Gateway (in-process or internal channel)
  ▼
Gateway
  │ Receives audio from Buffer/Decoder for call_id
  │ send_audio_to_client(call_id, audio) → WebSocket
  ▼
Client
```

---

## 6. Message shapes (suggested)

**Client → Gateway (WebSocket JSON)**

```json
{
  "type": "synthesize",
  "text_id": "req-uuid",
  "text": "Hello, this is FlowTTS.",
  "voice_id": "en-in-female-1",
  "language": "en-IN"
}
```

**Gateway → Redis (job on tts_queue)**

```json
{
  "call_id": "sess-uuid",
  "text_id": "req-uuid",
  "text": "Hello, this is FlowTTS.",
  "voice_id": "en-in-female-1",
  "language": "en-IN",
  "options": {},
  "published_at": 1234567890.123
}
```

**Worker → Redis (PUBLISH to audio:{call_id})**

```json
{
  "text_id": "req-uuid",
  "chunk_index": 0,
  "audio_base64": "...",
  "sample_rate": 24000,
  "format": "pcm_s16le",
  "is_final": true
}
```

**Gateway → Client (WebSocket)**

- Binary: raw PCM frames; or
- JSON: `{ "type": "audio", "text_id": "...", "audio": "<base64>", "is_final": true }`

---

## 7. Design notes

- **Same Redis**: Both **tts_queue** (list) and **audio:{call_id}** (pub/sub) use the same Redis instance; only key/channel names differ.
- **Buffer Manager + Decoder**: Can be implemented inside the Gateway (same process subscribes to `audio:{call_id}`, buffers, decodes, and sends on the WebSocket) to avoid an extra network hop.
- **Scalability**: Multiple workers can BLPOP from the same **tts_queue**; each session’s audio is isolated by **audio:{call_id}** so only the Gateway (or decoder) for that session needs to subscribe.
- **Backpressure**: If the client is slow, Gateway or Buffer can apply backpressure (e.g. pause consuming from Redis or drop chunks) to avoid unbounded memory.

This document is the single reference for the FlowTTS data flow and can be extended with deployment, config, and API details as needed.
