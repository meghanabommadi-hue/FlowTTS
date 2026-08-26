"""TRT-LLM network definition for OmniVoice's Qwen3 backbone.

Vendored from github.com/tlitech/omnivoice-trtllm (`patch/omnivoice/`) with no
behavioural changes — the graph these two modules build must stay bit-compatible
with the upstream engine, so they are kept verbatim rather than refactored.

This package is copied into ``tensorrt_llm/models/omnivoice`` and registered in
that package's MODEL_MAP by ``flowtts.trt.build_trtllm``; the relative imports
below (``from ..._utils``) resolve only once it sits there. It is therefore NOT
importable from this location, and nothing at serve time imports it.
"""
