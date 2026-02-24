"""FlowTTS — Telugu TTS pipeline.

Full pipeline (Redis-backed, multi-process):
  Client
    │  WebSocket (text)
    ▼
  Gateway  (flowtts/main.py or flowtts/server.py)
    │  rpush → Redis TTS queue
    ▼
  Worker   (flowtts/worker.py)
    │  sglang Engine → audio token string
    │  publish → Redis Pub/Sub channel
    ▼
  Gateway  (flowtts/api/websockets.py)
    │  optional ncodec decode → PCM/WAV
    ▼
  Client   (audio_tokens / audio_base64)

Single-process shortcut (flowtts/server.py):
  Client → server.py → sglang → audio_tokens → Client
  (no Redis, no worker, model loaded once in-process)

Package layout:
  core/        — Pydantic settings (env-vars, model paths, Redis config)
  api/         — WebSocket gateway: connection manager + message models
  synthesis/   — Text → audio tokens (sglang Engine + ncodec TTSCodec)
  worker.py    — Redis queue consumer, one job at a time or concurrent
  decoder/     — Audio tokens → PCM/WAV (ncodec TTSCodec.decode)
  processing/  — Post-decode audio: resample, crossfade
  monitoring/  — Structured logging (structlog) + in-process latency metrics
  test/        — Benchmark, load, and integration test scripts
"""
