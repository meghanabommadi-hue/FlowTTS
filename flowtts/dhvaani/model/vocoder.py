"""Pipeline position: VOCODER — mel spectrogram -> 24 kHz waveform.

Role in pipeline:
  Terminal GPU stage. Consumes the mel the flow scheduler produced (already
  divided by `flow.feat_scale`) and returns float32 PCM.

      scheduler -> mel (B, 100, T) -> Vocos -> waveform (B, T*256) -> CPU

Vocos is a fully convolutional ConvNeXt backbone followed by an inverse STFT, so
it is cheap relative to the flow decoder (roughly 2-4% of total GPU time at
num_step=8) and batches trivially. Two details matter:

  1. Padding bleed. Batching items of different true lengths means the ISTFT
     window at the boundary sees padded frames. Cutting each output at exactly
     `lengths[i] * HOP_LENGTH` removes the affected tail, so the audible result
     is identical to decoding the item alone.
  2. fp16 overflow. The ISTFT accumulates across overlapping windows and can
     saturate fp16 on loud material. We check finiteness with a single cheap
     reduction over the whole batch and re-run only the offending items in
     fp32 -- so the common case pays one extra kernel, not a precision penalty.
"""

from __future__ import annotations

from typing import Sequence

import structlog
import torch

from flowtts.dhvaani.config import HOP_LENGTH, MODEL_SAMPLE_RATE, dhv_settings
from flowtts.dhvaani.model.loader import LoadedModel

logger = structlog.get_logger(__name__)


class VocosVocoder:
    """Batched Vocos decode with length-exact trimming."""

    sample_rate: int = MODEL_SAMPLE_RATE

    def __init__(self, loaded: LoadedModel, settings=None):
        self._s = settings or dhv_settings
        self._m = loaded
        self._device = loaded.device
        self._dtype = loaded.dtype
        self._autocast = self._device.type == "cuda" and self._dtype != torch.float32
        self._fp16_overflows = 0
        self._calls = 0
        self._items = 0

    @torch.inference_mode()
    def _decode_raw(self, mel: torch.Tensor) -> torch.Tensor:
        """mel (B, 100, T) -> waveform (B, S).

        Runs under autocast rather than casting the Vocos weights: autocast puts
        the ConvNeXt convolutions in fp16 (where the time is) while keeping the
        inverse STFT in fp32 (where fp16 overflows on loud material). Casting the
        module instead would force the whole thing to one precision and also
        break on the fp32 biases the released checkpoint ships.
        """
        voc = self._m.vocoder
        mel = mel.float()
        if not self._autocast:
            return voc.decode(mel).squeeze(1).clamp(-1.0, 1.0)

        with torch.autocast("cuda", dtype=self._dtype, cache_enabled=False):
            out = voc.decode(mel).squeeze(1)
        out = out.float().clamp(-1.0, 1.0)

        # One reduction for the whole batch; only pay the fp32 path if it
        # actually went non-finite.
        if not bool(torch.isfinite(out).all()):
            self._fp16_overflows += 1
            bad = ~torch.isfinite(out).all(dim=-1)
            fixed = voc.decode(mel[bad]).squeeze(1).float().clamp(-1.0, 1.0)
            out[bad] = fixed
            logger.warning("vocoder_fp16_overflow", n_items=int(bad.sum()))
        return out

    @torch.inference_mode()
    def decode(
        self, mel: torch.Tensor, lengths: Sequence[int]
    ) -> list[torch.Tensor]:
        """Decode a padded batch.

        Args:
            mel: `(B, 100, T)`, already divided by `flow.feat_scale`.
            lengths: true mel frame count per item.

        Returns:
            One 1-D float32 tensor per item, ON THE DEVICE, cut to
            `lengths[i] * HOP_LENGTH` samples.

        Waveforms stay on the GPU deliberately. The caller (`engine/vocode.py`)
        may still need to resample -- 24 kHz to 8 kHz for telephony -- and doing
        that before the device-to-host copy is both faster and cuts the PCIe
        transfer by 3x. The copy happens once, at the end of that stage.
        """
        self._calls += 1
        self._items += mel.shape[0]

        # Group by identical true length. When a batch is length-homogeneous
        # (the common case, because the scheduler retires whole buckets) this is
        # a single group and costs nothing; when it is not, grouping avoids
        # padding a 1 s span up to a 6 s one.
        groups: dict[int, list[int]] = {}
        for i, L in enumerate(lengths):
            groups.setdefault(int(L), []).append(i)

        out: list[torch.Tensor | None] = [None] * mel.shape[0]
        for L, idxs in groups.items():
            if L <= 0:
                for i in idxs:
                    out[i] = torch.zeros(0, dtype=torch.float32, device=mel.device)
                continue
            sub = mel[idxs, :, :L]
            wav = self._decode_raw(sub)
            keep = L * HOP_LENGTH
            wav = wav[:, :keep]
            for j, i in enumerate(idxs):
                out[i] = wav[j]

        empty = torch.zeros(0, dtype=torch.float32, device=mel.device)
        return [o if o is not None else empty for o in out]

    def warmup(self, buckets: Sequence[int], batch_sizes: Sequence[int]) -> None:
        """Force cuDNN algorithm selection for the shapes we will actually see."""
        if self._device.type != "cuda":
            return
        for T in buckets:
            for B in batch_sizes:
                mel = torch.zeros((B, 100, T), device=self._device, dtype=self._dtype)
                try:
                    self.decode(mel, [T] * B)
                except Exception as e:  # pragma: no cover
                    logger.warning("vocoder_warmup_failed", frames=T, batch=B, error=str(e))
        torch.cuda.synchronize(self._device)
        logger.info("vocoder_warm", buckets=list(buckets), batches=list(batch_sizes))

    def stats(self) -> dict:
        return {
            "calls": self._calls,
            "items": self._items,
            "fp16_overflows": self._fp16_overflows,
        }
