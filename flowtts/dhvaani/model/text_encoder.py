"""Pipeline position: TEXT CONDITIONING — token ids -> frame-rate conditions.

Role in pipeline:
  Sits between tokenisation and the ODE scheduler. For a batch of newly admitted
  spans it produces the three tensors the flow decoder consumes for every one of
  its `num_step` iterations:

      token ids (prompt + span)
        -> embed -> text_encoder (Zipformer, 4 layers, width 192, out 100)
        -> upsample to frame rate by average token duration
        -> text_condition   (B, T, 100)
      prompt mel
        -> right-pad, zero past the prompt
        -> speech_condition (B, T, 100)
      lengths
        -> padding_mask     (B, T)

  Because these are computed ONCE per span and then reused by every ODE step,
  this stage is amortised ~8-16x and is never the bottleneck. What matters here
  is that it is (a) batched across all spans admitted in the same tick and
  (b) free of host synchronisation.

DhVaani's duration model:
  There is no duration predictor. `predict_feature_lens` simply says the
  generated audio has the same frames-per-character rate as the prompt clip.
  That is why a voice's reference transcript must actually match its audio --
  a transcript that is too short makes every clone of that voice speak slowly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import structlog
import torch

from flowtts.dhvaani.config import N_MELS, dhv_settings
from flowtts.dhvaani.model import ops
from flowtts.dhvaani.model.loader import LoadedModel

logger = structlog.get_logger(__name__)


@dataclass
class Conditions:
    """Per-span conditioning, padded to a single bucket width."""

    text_condition: torch.Tensor    # (B, num_frames, 100)
    speech_condition: torch.Tensor  # (B, num_frames, 100)
    padding_mask: torch.Tensor      # (B, num_frames) bool, True = padded
    total_lens: torch.Tensor        # (B,) int64  prompt + generated frames
    gen_lens: torch.Tensor          # (B,) int64  generated frames only
    prompt_lens: torch.Tensor       # (B,) int64


class TextEncoder:
    """Batched text conditioning for the flow decoder."""

    def __init__(self, loaded: LoadedModel, settings=None):
        self._s = settings or dhv_settings
        self._m = loaded
        self._fp32 = self._s.model.text_encoder_fp32
        self._enc_dtype = torch.float32 if self._fp32 else loaded.dtype
        self._out_dtype = loaded.dtype
        self._device = loaded.device

    # -- stage 1: encode -----------------------------------------------------
    @torch.inference_mode()
    def encode_batch(
        self, cat_token_ids: Sequence[Sequence[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode `prompt_tokens + span_tokens` per item.

        Returns `(embed (B, S, 100), tokens_lens (B,))` where `tokens_lens` is
        the TRUE token count, excluding the single pad slot `pad_token_ids`
        appends. That slot exists so the trailing-remainder frames have
        somewhere to point (see ops.token_frame_index).
        """
        model = self._m.zipvoice
        padded, tokens_lens = ops.pad_token_ids(
            list(cat_token_ids), self._m.pad_id, self._device
        )
        emb = model.embed(padded)
        if emb.dtype != self._enc_dtype:
            emb = emb.to(self._enc_dtype)

        # Upstream forward_text_embed masks against the TRUE length, so the
        # appended pad slot is treated as padded during encoding. Reproduce that
        # exactly -- the encoder's output at that slot is what the remainder
        # frames read.
        tok_mask = ops.make_pad_mask(tokens_lens, emb.shape[1])
        out = model.text_encoder(x=emb, t=None, padding_mask=tok_mask)
        return out, tokens_lens

    # -- stage 2: upsample to frame rate ------------------------------------
    @torch.inference_mode()
    def build_conditions(
        self,
        cat_token_ids: Sequence[Sequence[int]],
        prompt_token_lens: Sequence[int],
        prompt_mels: Sequence[torch.Tensor],
        prompt_mel_lens: Sequence[int],
        speeds: Sequence[float],
        num_frames: int,
    ) -> Conditions:
        """Build all three conditions padded to `num_frames` (the bucket width).

        Note `num_frames` is the BUCKET width, not `max(total_lens)`. Padding to
        the bucket is what lets the scheduler write straight into a
        pre-allocated arena and lets TensorRT/CUDA-graphs specialise on a small
        set of shapes.
        """
        B = len(cat_token_ids)
        dev = self._device

        embed, tokens_lens = self.encode_batch(cat_token_ids)

        prompt_tok = torch.tensor(list(prompt_token_lens), dtype=torch.int64, device=dev)
        prompt_frames = torch.tensor(list(prompt_mel_lens), dtype=torch.int64, device=dev)
        # tokens_lens counts prompt + span; the span-only count drives duration.
        span_tok = tokens_lens - prompt_tok

        speed_t = torch.tensor(list(speeds), dtype=torch.float32, device=dev)
        total_lens = ops.predict_feature_lens(prompt_frames, prompt_tok, span_tok, speed_t)
        total_lens = torch.clamp(total_lens, max=num_frames)
        gen_lens = torch.clamp(total_lens - prompt_frames, min=0)

        text_condition, padding_mask = ops.build_text_condition(
            embed, tokens_lens, total_lens, num_frames
        )

        # Stack the ragged prompt mels into one padded tensor.
        max_p = max(int(x) for x in prompt_mel_lens) if B else 0
        stacked = torch.zeros((B, max(max_p, 1), N_MELS), dtype=self._out_dtype, device=dev)
        for i, mel in enumerate(prompt_mels):
            n = min(mel.shape[0], stacked.shape[1])
            stacked[i, :n] = mel[:n].to(self._out_dtype)

        speech_condition = ops.build_speech_condition(stacked, prompt_frames, num_frames)

        return Conditions(
            text_condition=text_condition.to(self._out_dtype),
            speech_condition=speech_condition.to(self._out_dtype),
            padding_mask=padding_mask,
            total_lens=total_lens,
            gen_lens=gen_lens,
            prompt_lens=prompt_frames,
        )

    # -- planning helper -----------------------------------------------------
    def estimate_total_frames(
        self, prompt_frames: int, prompt_tokens: int, span_tokens: int, speed: float
    ) -> int:
        """Host-side duration estimate, used to pick a bucket BEFORE any GPU work.

        Same formula as `ops.predict_feature_lens`, in plain Python so the
        scheduler can size its arena slot without a device round-trip.
        """
        import math

        if prompt_tokens <= 0:
            return prompt_frames
        gen = math.ceil(prompt_frames / prompt_tokens * span_tokens / max(speed, 1e-6))
        return int(prompt_frames + gen)
