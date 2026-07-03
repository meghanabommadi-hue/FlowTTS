"""FlowTTS — Fish Audio S2 Pro streaming TTS gateway.

Model / serving: **Fish Audio S2 Pro** (`fishaudio/s2-pro`) — a Dual-AR (Qwen3-4B
slow-AR + 400M fast-AR) TTS model with an EVA-GAN / RVQ codec (24 kHz), served
out-of-process by **sglang-omni** (`sgl-omni serve`, OpenAI-compatible
`POST /v1/audio/speech`). All GPU work — AR decoding, codec decode, continuous
batching, RadixAttention prefix caching — lives in the sglang backend. This gateway
is a CPU-only WebSocket proxy that preserves FlowTTS's streaming protocol, voice
registry, WAV cache, metrics, and control API. It replaced the previous in-process
k2-fsa/OmniVoice diffusion stack.

Primary path (single process — flowtts/server.py):
  Client ──WS(text)──▶ server.py (N ports, one asyncio loop)
    → FishSpeechEngine.synthesize_stream() → POST sglang /v1/audio/speech (pcm)
    → contiguous 24 kHz PCM → (resample) → int16 PCM ──▶ Client (audio_chunk … audio_done)

Secondary path (Redis-backed multi-process — flowtts/main.py + worker.py):
  Gateway rpush → Redis queue → worker synth → base64 WAV → Redis pubsub → client.

Package layout:
  core/        — Pydantic settings (backend URL, generation, output, streaming)
  synthesis/   — Fish backend client (fish_engine), synthesizer facade, text chunker
  voices/      — reference-clip registry + manifest store + offline builder (alias → reference)
  decoder/     — waveform → PCM/WAV helpers (+ Redis-path lifecycle bookkeeping)
  processing/  — post-decode audio: resample, crossfade, fades
  api/         — WebSocket gateway (Redis path) + request/response models
  worker.py    — Redis queue consumer (secondary path)
  monitoring/  — structlog logging + in-process + Prometheus metrics
  test/        — benchmark, load, and unit test scripts
"""
