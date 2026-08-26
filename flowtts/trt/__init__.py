"""TensorRT / TensorRT-LLM acceleration for OmniVoice's Qwen3 backbone.

Implements github.com/tlitech/omnivoice-trtllm's approach — leave every part of
OmniVoice alone except ``llm.forward``, and route that one call through a
compiled engine — with three interchangeable backends behind one contract:

    backbone(hidden_states[B,S,H], input_lengths[B]) -> [B,S,H]

    tensorrt  engines/…/backbone.plan   built by `python -m flowtts.trt.build_trt`
    trtllm    engines/…/rank0.engine    built by `python -m flowtts.trt.build_trtllm`
    torch     no engine                 the PyTorch mirror, optionally compiled

Serve-time entry point::

    from flowtts.trt import patch_model
    result = patch_model(model, settings.omnivoice)   # never raises
"""

from flowtts.trt.backbone import BackboneConfig, Qwen3Backbone, precompute_rope
from flowtts.trt.patcher import BackbonePatch, PatchResult, patch_model

__all__ = [
    "BackboneConfig",
    "BackbonePatch",
    "PatchResult",
    "Qwen3Backbone",
    "patch_model",
    "precompute_rope",
]
