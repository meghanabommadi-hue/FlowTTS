"""FlowTTS — OmniVoice streaming TTS server.

Model: k2-fsa/OmniVoice — a non-autoregressive discrete-diffusion TTS language
model (Qwen3-0.6B backbone + Higgs-Audio-v2 neural codec, 24 kHz). It replaced
the previous sglang + ncodec MiraTTS stack; the serving framework (WebSocket
gateway, dynamic batching, metrics, WAV cache, streaming protocol) was kept.

Primary path (single process, no Redis — flowtts/server.py):
  Client ──WS(text)──▶ server.py (OmniVoice loaded once, N ports)
    → synthesizer splits text into chunks (short first chunk = low TTFB)
    → OmniVoiceEngine dynamic batch queue coalesces chunks/requests → generate()
    → 24 kHz waveform → (resample) → int16 PCM ──▶ Client (audio_chunk … audio_done)

Secondary path (Redis-backed multi-process — flowtts/main.py + worker.py):
  Gateway rpush → Redis queue → worker generate() → base64 WAV → Redis pubsub → client.

Package layout:
  core/        — Pydantic settings (model, output, streaming, batching, accel)
  synthesis/   — OmniVoice engine + dynamic batcher, synthesizer facade, text chunker
  voices/      — voice-clone npz registry + offline builder (alias → VoiceClonePrompt)
  decoder/     — waveform → PCM/WAV helpers (+ Redis-path lifecycle bookkeeping)
  processing/  — post-decode audio: resample, crossfade, fades
  api/         — WebSocket gateway (Redis path) + request/response models
  worker.py    — Redis queue consumer (secondary path)
  monitoring/  — structlog logging + in-process + Prometheus metrics
  test/        — benchmark, load, and unit test scripts
"""
