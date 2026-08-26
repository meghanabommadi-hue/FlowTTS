"""Pipeline position: ACCELERATION — standalone Qwen3 backbone (torch, exportable).

Role in pipeline:
  OmniVoice's generation loop calls ``self.llm(inputs_embeds=…, attention_mask=…)``
  once per denoise step. Everything else — embeddings, audio heads, the unmasking
  loop, the codec — stays in PyTorch. So the whole acceleration story is: replace
  that one call with something faster that returns the same tensor.

      OmniVoice.generate()
        └─ _generate_iterative()          [unchanged upstream code]
             └─ self.llm(inputs_embeds=…) ← patched   (flowtts.trt.patcher)
                  └─ Qwen3Backbone / TensorRT engine / TRT-LLM engine

  This module is the PyTorch half of that: a faithful, self-contained
  re-implementation of the 28 Qwen3 layers + final RMSNorm, holding the *same
  weight tensors* as ``model.llm`` (no copy — the parameters are shared).

Why re-implement instead of exporting ``model.llm`` directly:
  transformers' Qwen3Model builds causal masks and cache objects inside forward,
  neither of which survives ONNX export or CUDA-graph capture cleanly. This
  module takes exactly the four tensors the upstream TRT-LLM engine takes
  (github.com/tlitech/omnivoice-trtllm, ``patch/omnivoice/model.py``) —
  hidden_states, rope_cos, rope_sin, input_lengths — so one runtime contract
  covers all three backends and they are interchangeable at run time.

  Correctness is not assumed: ``validate_against_llm`` compares this module's
  output to the real ``model.llm`` and reports cosine similarity, the same check
  upstream's ``_validate_trt_vs_pytorch`` performs. The patcher runs it at
  startup and refuses to swap in a backbone that does not match.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class BackboneConfig:
    """The subset of OmniVoice's llm_config the backbone needs."""

    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 4096

    @classmethod
    def from_hf(cls, llm_config) -> "BackboneConfig":
        """Read the config off a loaded transformers Qwen3 config object or dict."""
        get = (llm_config.get if isinstance(llm_config, dict)
               else lambda k, d=None: getattr(llm_config, k, d))

        rope = get("rope_parameters", None) or {}
        rope_theta = (rope.get("rope_theta") if isinstance(rope, dict)
                      else getattr(rope, "rope_theta", None))
        if rope_theta is None:
            rope_theta = get("rope_theta", 1_000_000.0)

        hidden = int(get("hidden_size", 1024))
        heads = int(get("num_attention_heads", 16))
        return cls(
            hidden_size=hidden,
            num_hidden_layers=int(get("num_hidden_layers", 28)),
            num_attention_heads=heads,
            num_key_value_heads=int(get("num_key_value_heads", 8)),
            head_dim=int(get("head_dim", None) or hidden // heads),
            intermediate_size=int(get("intermediate_size", 3072)),
            rms_norm_eps=float(get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope_theta),
            max_position_embeddings=int(get("max_position_embeddings", 4096)),
        )


class RMSNorm(nn.Module):
    """Qwen3 RMSNorm: normalize in fp32, scale, cast back."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32.to(dtype)) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[-x2, x1] where x = [x1, x2] split at the head-dim midpoint."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings. q/k are [B, H, S, D]; cos/sin are [B, S, D]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Qwen3Attention(nn.Module):
    """Grouped-query attention with per-head QK RMSNorm (the Qwen3 variant)."""

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_kv_groups = cfg.num_attention_heads // cfg.num_key_value_heads
        self.scale = cfg.head_dim ** -0.5

        q_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        # Flipped by Qwen3Backbone.set_export_mode() around torch.onnx.export.
        self.export_mode = False

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor | None) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim)

        # QK norm is per head and comes BEFORE rope — that ordering is what
        # makes this Qwen3 rather than Qwen2, and getting it wrong still
        # produces plausible-looking audio, just in the wrong voice.
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Bidirectional, not causal: OmniVoice is a discrete-diffusion model and
        # every position sees every other valid position at each denoise step.
        if self.export_mode:
            out = self._attention_explicit(q, k, v, attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                 is_causal=False)
        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)

    def _attention_explicit(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            attn_mask: torch.Tensor | None) -> torch.Tensor:
        """Attention written out, for ONNX export.

        ``F.scaled_dot_product_attention`` is faster in PyTorch, but it exports
        to a subgraph that derives its scale through Shape -> Slice -> Cast ->
        Sqrt and applies the mask as Where(mask, 0, -inf). A strongly-typed
        TensorRT build finds no implementation for that mixed-dtype scale chain
        and refuses to build; the -inf also risks NaN in fp16 softmax.

        This form is mathematically identical but has a constant scale, a finite
        additive mask, and an fp32 softmax — a graph TensorRT compiles cleanly.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)).float() * self.scale
        if attn_mask is not None:
            # A large finite negative rather than -inf: -inf survives softmax as
            # NaN whenever a row is fully masked, and NaN poisons every layer
            # after it.
            scores = scores.masked_fill(~attn_mask, -1.0e30)
        weights = torch.softmax(scores, dim=-1).to(v.dtype)
        return torch.matmul(weights, v)


class Qwen3MLP(nn.Module):
    """SwiGLU feed-forward."""

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Block(nn.Module):
    """Pre-norm attention → residual → post-norm MLP → residual."""

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_mask)
        return x + self.mlp(self.post_attention_layernorm(x))


# How the validity mask gets its position indices matters more than it looks.
# Two obvious formulations both produce a TensorRT engine that silently returns
# all zeros, while being perfectly correct in PyTorch and in onnxruntime:
#
#   torch.arange(hidden_states.shape[1])   -> ONNX `Range` over a dynamic dim
#   self.positions[:seq]                    -> ONNX `Slice` of a constant by a
#                                              shape-derived bound
#
# In both, TensorRT folds the index tensor at build time and the `Less` against
# input_lengths comes out uniformly false, so the final `x * mask` zeroes the
# whole output. Nothing errors — the engine just returns silence, which is why
# the patcher validates against the real module before installing anything.
#
# The formulation below derives the positions from a cumulative sum over a real
# input tensor instead. It is data-driven rather than shape-driven, uses only
# elementwise ops and CumSum, and produces a correct engine.
DEFAULT_MAX_POSITIONS = 4096


class Qwen3Backbone(nn.Module):
    """The 28 Qwen3 layers + final norm, with the upstream engine's I/O contract.

    forward(hidden_states[B,S,H], rope_cos[B,S,D], rope_sin[B,S,D],
            input_lengths[B]) -> [B,S,H]
    """

    def __init__(self, cfg: BackboneConfig, max_positions: int = DEFAULT_MAX_POSITIONS) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_positions = max_positions
        self.layers = nn.ModuleList(Qwen3Block(cfg) for _ in range(cfg.num_hidden_layers))
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.register_buffer(
            "positions",
            torch.arange(max_positions, dtype=torch.int32),
            persistent=False,
        )

    # ------------------------------------------------------------------ build
    @classmethod
    def from_llm(cls, llm: nn.Module, cfg: BackboneConfig | None = None,
                 max_positions: int = DEFAULT_MAX_POSITIONS) -> "Qwen3Backbone":
        """Build a backbone that SHARES ``llm``'s parameter tensors (no copy).

        Sharing rather than copying means no extra VRAM and no chance of the two
        drifting apart. ``llm`` stays loaded because OmniVoice still reaches into
        ``llm.model.embed_tokens`` for text embeddings.
        """
        cfg = cfg or BackboneConfig.from_hf(getattr(llm, "config", {}))
        self = cls(cfg, max_positions=max_positions)

        src_layers = getattr(llm, "layers", None)
        if src_layers is None:                       # Qwen3ForCausalLM wrapper
            src_layers = llm.model.layers
        src_norm = getattr(llm, "norm", None)
        if src_norm is None:
            src_norm = llm.model.norm

        if len(src_layers) != cfg.num_hidden_layers:
            raise ValueError(
                f"backbone expects {cfg.num_hidden_layers} layers, llm has {len(src_layers)}"
            )

        for dst, src in zip(self.layers, src_layers):
            dst.input_layernorm.weight = src.input_layernorm.weight
            dst.post_attention_layernorm.weight = src.post_attention_layernorm.weight
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(dst.self_attn, name).weight = getattr(src.self_attn, name).weight
            for name in ("q_norm", "k_norm"):
                getattr(dst.self_attn, name).weight = getattr(src.self_attn, name).weight
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(dst.mlp, name).weight = getattr(src.mlp, name).weight
        self.final_norm.weight = src_norm.weight

        # from_llm shares the source tensors rather than copying them, so this
        # module is never .to(device)'d and the position buffer registered in
        # __init__ would still be on the CPU. Put it where the weights are.
        device = src_norm.weight.device
        self.positions = self.positions.to(device)

        return self.eval()

    def set_export_mode(self, enabled: bool) -> None:
        """Switch every attention block to the ONNX-friendly explicit form."""
        for layer in self.layers:
            layer.self_attn.export_mode = enabled

    # ------------------------------------------------------------------ run
    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # Positions 0..S-1 per row, derived from the input itself rather than
        # from its shape — see the note on DEFAULT_MAX_POSITIONS for why the
        # shape-derived forms produce an all-zero TensorRT engine.
        ones = torch.ones_like(hidden_states[:, :, 0], dtype=torch.float32)   # [B,S]
        positions = torch.cumsum(ones, dim=1) - 1.0                            # [B,S]
        # [B,1,1,S]: broadcast over query positions. Queries beyond a row's
        # length still compute, but their outputs are masked to zero below and
        # OmniVoice never reads them.
        valid = positions < input_lengths.unsqueeze(1).to(torch.float32)
        attn_mask = valid[:, None, None, :]

        x = hidden_states
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, attn_mask)
        x = self.final_norm(x)
        return x * valid.unsqueeze(-1).to(x.dtype)


def precompute_rope(
    cfg: BackboneConfig,
    device: torch.device | str,
    dtype: torch.dtype = torch.float16,
    max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE cos/sin tables of shape [max_seq_len, head_dim].

    Computed exactly as upstream's ``OmniVoiceTRTLLM._precompute_rope`` does, so
    a PyTorch backbone and a TRT engine consume identical tables.
    """
    max_seq_len = max_seq_len or cfg.max_position_embeddings
    inv_freq = 1.0 / (
        cfg.rope_theta
        ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim)
    )
    freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(device=device, dtype=dtype), emb.sin().to(device=device, dtype=dtype)


@torch.no_grad()
def validate_against_llm(
    backbone: "Qwen3Backbone",
    llm: nn.Module,
    *,
    seq_len: int = 64,
    batch_size: int = 2,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> dict:
    """Compare the backbone to the real ``llm`` on random input.

    Uses a full-length (unpadded) input so both paths see identical masking —
    padded positions are legitimately allowed to differ, since neither the
    engine nor OmniVoice ever reads them.

    Returns ``{"cosine": float, "max_abs_diff": float, "rel_error": float}``.
    """
    cfg = backbone.cfg
    embeds = torch.randn(batch_size, seq_len, cfg.hidden_size, device=device, dtype=dtype)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=device)

    reference = llm(inputs_embeds=embeds, attention_mask=mask, return_dict=True)[0].float()

    cos, sin = precompute_rope(cfg, device, dtype, max_seq_len=max(seq_len, 16))
    got = backbone(
        embeds,
        cos[:seq_len].unsqueeze(0).expand(batch_size, -1, -1).contiguous(),
        sin[:seq_len].unsqueeze(0).expand(batch_size, -1, -1).contiguous(),
        lengths,
    ).float()

    cosine = F.cosine_similarity(reference.flatten().unsqueeze(0),
                                 got.flatten().unsqueeze(0)).item()
    max_abs = (reference - got).abs().max().item()
    denom = reference.abs().mean().item() or 1.0
    return {
        "cosine": cosine,
        "max_abs_diff": max_abs,
        "rel_error": (reference - got).abs().mean().item() / denom,
    }
