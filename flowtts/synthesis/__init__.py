"""Synthesis: text → 24 kHz waveform (OmniVoice engine + dynamic batcher).

Import submodules directly to avoid pulling torch/omnivoice at package import:
  from flowtts.synthesis.engine import synthesis_service          # heavy (GPU)
  from flowtts.synthesis.text_chunker import split_for_streaming  # light (stdlib)
"""
