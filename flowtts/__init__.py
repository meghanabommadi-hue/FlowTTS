"""
FlowTTS package.

This package provides a minimal, modular TTS stack built around:
- a text-to-audio-token engine (powered by sglang),
- an audio-token decoder (powered by ncodec.TTSCodec),
- a FastAPI + WebSocket gateway.

The layout and coding style intentionally mirror LITranscriber to keep
integration and maintenance simple.
"""

