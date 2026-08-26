"""Pipeline position: ACCELERATION — swap OmniVoice's LLM forward for a fast backend.

Role in pipeline:
  This is the whole integration surface of the TRT work, and it is deliberately
  one function. Following github.com/tlitech/omnivoice-trtllm, nothing about
  OmniVoice's generation is reimplemented: the embeddings, the audio heads, the
  iterative unmasking loop, the CFG scoring and the Higgs codec all stay exactly
  as upstream wrote them. Only ``model.llm.forward`` is replaced.

      OmniVoice.generate()
        └─ _generate_iterative()             [upstream, untouched]
             └─ self(input_ids, audio_mask, attention_mask)
                  └─ _prepare_embed_inputs() [upstream, untouched]
                  └─ self.llm(inputs_embeds=…, attention_mask=…)   ← PATCHED
                  └─ self.audio_heads(…)     [upstream, untouched]

Safety rails, because "quality must not be hampered" is the requirement:
  • the replacement is validated against the real ``llm`` before it is installed
    (cosine similarity on random input, the same check upstream's debug mode
    runs), and a backend below ``min_cosine`` is rejected — the server then
    serves on plain PyTorch rather than serving degraded audio;
  • ``restore()`` puts the original forward back, so a failure at any later point
    is recoverable in-process;
  • anything the fast path cannot represent (an explicit non-default
    ``position_ids``) falls through to the original forward rather than being
    silently ignored.

What is deliberately NOT patched: ``_generate_iterative``. Its per-item Python
loop and float32 logits copy do cost real time at large batch, but rewriting it
is the one change that could alter the audio, and the backbone is ~98% of the
arithmetic anyway (the audio-head GEMM is 16.8 MFLOP/token against the
backbone's 880).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from flowtts.trt.backbone import BackboneConfig, validate_against_llm
from flowtts.trt.runtime import TorchBackbone, TRTBackbone, TRTLLMBackbone

logger = logging.getLogger(__name__)

# Below this the backend is not the same function as the model it replaces.
# Upstream warns at 0.9; we refuse, because a warning in a log does not stop
# bad audio reaching a live call.
DEFAULT_MIN_COSINE = 0.99


@dataclass
class PatchResult:
    """What actually got installed, for /v1/stats and the startup log."""

    backend: str
    validated: bool
    cosine: float | None
    info: dict
    fell_back: bool = False
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "validated": self.validated,
            "cosine": round(self.cosine, 6) if self.cosine is not None else None,
            "fell_back": self.fell_back,
            "reason": self.reason,
            **self.info,
        }


class BackbonePatch:
    """Installs a fast backbone into a loaded OmniVoice model, reversibly."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._original: Optional[Callable] = None
        self.backend = None
        self.result: PatchResult | None = None

    # ------------------------------------------------------------------ build
    def _make_backend(self, kind: str, cfg: BackboneConfig, settings_ov) -> Any:
        device = settings_ov.device
        engine_dir = settings_ov.trt_engine_dir

        if kind == "tensorrt":
            return TRTBackbone(engine_dir, cfg, device=device)
        if kind == "trtllm":
            return TRTLLMBackbone(settings_ov.trtllm_engine_dir or engine_dir, cfg,
                                  device=device)
        if kind == "torch":
            return TorchBackbone(
                self.model.llm,
                compile_model=settings_ov.compile_model,
                compile_mode=settings_ov.compile_mode,
            )
        raise ValueError(f"unknown backbone backend: {kind!r}")

    # ------------------------------------------------------------------ apply
    def apply(
        self,
        kind: str,
        settings_ov,
        *,
        min_cosine: float = DEFAULT_MIN_COSINE,
        validate: bool = True,
    ) -> PatchResult:
        """Install backend *kind*, validating it first. Never raises.

        Returns a :class:`PatchResult` describing what is actually running — on
        any failure that is the untouched PyTorch path, not a half-applied one.
        """
        if kind in (None, "", "none", "pytorch"):
            self.result = PatchResult("pytorch", validated=True, cosine=None, info={})
            return self.result

        cfg = BackboneConfig.from_hf(self.model.llm.config)

        try:
            backend = self._make_backend(kind, cfg, settings_ov)
        except Exception as exc:  # noqa: BLE001 — a missing engine must not stop the server
            logger.warning("backbone backend %s unavailable (%s); serving on PyTorch",
                           kind, exc)
            self.result = PatchResult("pytorch", validated=True, cosine=None, info={},
                                      fell_back=True, reason=f"{type(exc).__name__}: {exc}")
            return self.result

        cosine = None
        if validate:
            try:
                report = self._validate(backend, cfg)
                cosine = report["cosine"]
                if cosine < min_cosine:
                    logger.error(
                        "backbone backend %s failed validation: cosine=%.6f < %.6f "
                        "(max_abs_diff=%.4g); serving on PyTorch",
                        kind, cosine, min_cosine, report["max_abs_diff"],
                    )
                    self.result = PatchResult(
                        "pytorch", validated=False, cosine=cosine, info=backend.info(),
                        fell_back=True, reason=f"cosine {cosine:.6f} < {min_cosine}",
                    )
                    return self.result
                logger.info("backbone backend %s validated: cosine=%.6f max_abs_diff=%.4g",
                            kind, cosine, report["max_abs_diff"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("backbone validation failed for %s (%s); serving on PyTorch",
                               kind, exc)
                self.result = PatchResult("pytorch", validated=False, cosine=None,
                                          info={}, fell_back=True,
                                          reason=f"validation error: {exc}")
                return self.result

        self._install(backend)
        self.result = PatchResult(kind, validated=validate, cosine=cosine,
                                  info=backend.info())
        return self.result

    def _validate(self, backend, cfg: BackboneConfig) -> dict:
        """Compare *backend* against the untouched ``llm`` on random input."""
        device = next(self.model.llm.parameters()).device
        dtype = next(self.model.llm.parameters()).dtype
        seq_len, batch = 64, 2

        embeds = torch.randn(batch, seq_len, cfg.hidden_size, device=device, dtype=dtype)
        lengths = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
        mask = torch.ones(batch, 1, seq_len, seq_len, dtype=torch.bool, device=device)

        with torch.no_grad():
            reference = self.model.llm(inputs_embeds=embeds, attention_mask=mask,
                                       return_dict=True)[0].float()
            got = backend(embeds, lengths).float()

        cosine = torch.nn.functional.cosine_similarity(
            reference.flatten().unsqueeze(0), got.flatten().unsqueeze(0)
        ).item()
        return {"cosine": cosine, "max_abs_diff": (reference - got).abs().max().item()}

    def _install(self, backend) -> None:
        model = self.model
        self.backend = backend
        self._original = model.llm.forward

        original = self._original

        def _patched_forward(inputs_embeds=None, attention_mask=None,
                             position_ids=None, **kwargs):
            # The fast path assumes positions 0..S-1, which is what OmniVoice
            # uses (it passes position_ids=None). Anything else goes to the
            # original rather than being silently mis-rotated.
            if inputs_embeds is None or position_ids is not None:
                return original(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                position_ids=position_ids, **kwargs)

            batch, seq, _ = inputs_embeds.shape
            if attention_mask is not None and attention_mask.dim() == 4:
                # Row 0 of each item's mask is its valid-prefix indicator: True
                # for :len. The diagonal-only entries OmniVoice writes for the
                # unconditional half's padding sit at rows >= len, so they do
                # not disturb this sum.
                input_lengths = attention_mask[:, 0, 0, :].sum(dim=-1).to(torch.int32)
            elif attention_mask is not None and attention_mask.dim() == 2:
                input_lengths = attention_mask.sum(dim=-1).to(torch.int32)
            else:
                input_lengths = torch.full((batch,), seq, dtype=torch.int32,
                                           device=inputs_embeds.device)

            hidden = backend(inputs_embeds, input_lengths)

            from transformers.modeling_outputs import BaseModelOutputWithPast
            return BaseModelOutputWithPast(last_hidden_state=hidden.to(inputs_embeds.dtype))

        model.llm.forward = _patched_forward

    def restore(self) -> None:
        """Put the original ``llm.forward`` back."""
        if self._original is not None:
            self.model.llm.forward = self._original
            self._original = None
            self.backend = None


def patch_model(model: Any, settings_ov) -> PatchResult:
    """Install the configured backbone backend into *model*; return what ran.

    ``settings_ov`` is ``settings.omnivoice``. Reads ``backbone_backend``
    ("auto" | "tensorrt" | "trtllm" | "torch" | "pytorch"). "auto" prefers a
    built TensorRT engine, then TRT-LLM, then compiled torch, then plain
    PyTorch — the first one whose prerequisites are actually present.
    """
    kind = (settings_ov.backbone_backend or "auto").lower()
    patch = BackbonePatch(model)

    if kind != "auto":
        result = patch.apply(kind, settings_ov,
                             min_cosine=settings_ov.backbone_min_cosine,
                             validate=settings_ov.backbone_validate)
        model._flowtts_backbone_patch = patch
        return result

    for candidate in _auto_order(settings_ov):
        result = patch.apply(candidate, settings_ov,
                             min_cosine=settings_ov.backbone_min_cosine,
                             validate=settings_ov.backbone_validate)
        if not result.fell_back and result.backend != "pytorch":
            model._flowtts_backbone_patch = patch
            return result

    model._flowtts_backbone_patch = patch
    return patch.apply("pytorch", settings_ov)


def _auto_order(settings_ov) -> list[str]:
    """Backends worth trying, best first, filtered to those that could work."""
    order: list[str] = []

    engine_dir = Path(settings_ov.trt_engine_dir or "")
    if (engine_dir / "backbone.plan").exists():
        order.append("tensorrt")

    trtllm_dir = Path(settings_ov.trtllm_engine_dir or settings_ov.trt_engine_dir or "")
    if (trtllm_dir / "rank0.engine").exists():
        order.append("trtllm")

    if settings_ov.compile_model:
        order.append("torch")

    return order
