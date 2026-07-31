"""
Batched-codec-decode patch for OmniVoice.

Verified from the model source (omnivoice/models/omnivoice.py) that:
  - The LLM / iterative-unmasking core (`_generate_iterative`,
    `_generate_chunked`) is ALREADY batched: it builds one padded
    `[2B, C, L]` tensor per denoising step and runs a single forward call
    across the whole batch. No changes needed there.
  - `HiggsAudioV2TokenizerModel.decode()` (the neural codec / vocoder) is
    batch-capable at the model level (`(batch, num_quantizers, T)` in,
    `(batch, channels, T)` out) and is pure-conv (LayerNorm/GroupNorm only,
    no BatchNorm) so padding one sample cannot leak into another's output.
  - `OmniVoice._decode_and_post_process` calls `decode()` once per sample
    (and once per chunk for long-form items) with `.unsqueeze(0)`, i.e.
    batch size 1 each time. That per-item Python loop is the only unbatched
    stage downstream of generation, and this patch batches it.

This intentionally does NOT batch the reference-audio *encode* path
(`create_voice_clone_prompt`): the codec's HuBERT-based semantic feature
extractor doesn't accept an attention_mask in `_extract_semantic_features`,
so batching variable-length ref audio with zero-padding would let HuBERT's
self-attention attend over padded silence and subtly perturb the real
frames near each clip's boundary. Fixing that safely requires plumbing an
attention mask through a third-party HF model call, which is out of scope
for a decode-only, quality-preserving change.
"""

from typing import List, Optional, Union

import numpy as np
import torch

from omnivoice.models.omnivoice import (
    OmniVoice,
    OmniVoiceGenerationConfig,
)
from omnivoice.utils.audio import cross_fade_chunks


def _batched_decode_and_post_process(
    self: OmniVoice,
    all_tokens: List[Union[torch.Tensor, List[torch.Tensor]]],
    all_rms: List[Optional[float]],
    gen_config: OmniVoiceGenerationConfig,
) -> List[np.ndarray]:
    """Decode every sample's (and every chunk's) audio tokens in one batched
    codec call instead of one `decode()` call per sample/chunk.

    Args:
        all_tokens: One entry per sample, in the same layout `generate()`
            already produces: a single `(C, T)` tensor for non-chunked
            samples, or a `List[(C, T)]` for chunked (long-form) samples.
        all_rms: Reference RMS per sample, for volume post-processing.
        gen_config: Generation config (post-processing options).
    Returns:
        List of 1-D `np.ndarray` waveforms, one per sample.
    """
    tokenizer_device = self.audio_tokenizer.device

    # Flatten to a single list of (C, T) tensors, remembering how to
    # reassemble them back into per-sample (possibly multi-chunk) groups.
    flat_tokens: List[torch.Tensor] = []
    owner: List[int] = []  # sample index each flat entry belongs to
    for sample_idx, tokens in enumerate(all_tokens):
        if isinstance(tokens, list):
            for chunk in tokens:
                flat_tokens.append(chunk)
                owner.append(sample_idx)
        else:
            flat_tokens.append(tokens)
            owner.append(sample_idx)

    lengths = [t.size(-1) for t in flat_tokens]
    max_len = max(lengths)
    num_codebooks = flat_tokens[0].size(0)
    # NOTE: pad with 0, a valid codec codebook index (codec vocab is
    # 0..codebook_size-1). `self.config.audio_mask_id` (1024) is the LLM's
    # own mask sentinel, one past the codec's valid range — using it here
    # would index out of bounds in the codec's embedding table. The padded
    # region is discarded below via `true_len` slicing, so its value has no
    # effect on the output as long as it's in-range.
    pad_id = 0

    batch = torch.full(
        (len(flat_tokens), num_codebooks, max_len),
        pad_id,
        dtype=flat_tokens[0].dtype,
        device=tokenizer_device,
    )
    for i, t in enumerate(flat_tokens):
        batch[i, :, : t.size(-1)] = t.to(tokenizer_device)

    # Pure-conv codec (no attention/BatchNorm) — padding a shorter sample
    # cannot influence another sample's output, so a single padded batch
    # call is numerically equivalent to per-sample calls for the valid
    # (non-padded) region, which we slice out immediately below.
    decoded = self.audio_tokenizer.decode(batch).audio_values  # [N, 1, T']

    # Map codec time-steps back from padded token length.
    hop = decoded.shape[-1] / max_len
    flat_audios = []
    for i, tok_len in enumerate(lengths):
        true_len = int(round(tok_len * hop))
        flat_audios.append(decoded[i, :, :true_len].cpu().numpy())

    # Reassemble per-sample groups (cross-fade multi-chunk long-form items).
    per_sample_chunks: List[List[np.ndarray]] = [[] for _ in all_tokens]
    for flat_idx, sample_idx in enumerate(owner):
        per_sample_chunks[sample_idx].append(flat_audios[flat_idx])

    results = []
    for sample_idx, chunks in enumerate(per_sample_chunks):
        if len(chunks) > 1:
            audio_waveform = cross_fade_chunks(chunks, self.sampling_rate)
        else:
            audio_waveform = chunks[0]

        audio_waveform = self._post_process_audio(
            audio_waveform,
            ref_rms=all_rms[sample_idx],
            gen_config=gen_config,
        )
        results.append(audio_waveform.squeeze(0))

    return results


def enable_batched_decode(model: OmniVoice) -> OmniVoice:
    """Patch a loaded OmniVoice instance to decode all samples in one
    batched codec call instead of one call per sample/chunk.

    Only the final decode step is changed; generation (tokenizer, LLM,
    iterative unmasking) is untouched since it was already batched.
    """
    model.generate = _generate_with_batched_decode.__get__(model, OmniVoice)
    return model


@torch.inference_mode()
def _generate_with_batched_decode(
    self: OmniVoice,
    text: Union[str, List[str]],
    language: Union[str, List[str], None] = None,
    ref_text: Union[str, List[str], None] = None,
    ref_audio=None,
    voice_clone_prompt=None,
    instruct: Union[str, List[str], None] = None,
    duration: Union[float, List[Optional[float]], None] = None,
    speed: Union[float, List[Optional[float]], None] = None,
    generation_config: Optional[OmniVoiceGenerationConfig] = None,
    **kwargs,
) -> List[np.ndarray]:
    """Drop-in replacement for `OmniVoice.generate` (same signature) that
    batches the final codec-decode step. See module docstring for why this
    is the only stage that needed changing.
    """
    if self.audio_tokenizer is None or self.text_tokenizer is None:
        raise RuntimeError(
            "Model is not loaded with audio/text tokenizers. Make sure you "
            "loaded the model with OmniVoice.from_pretrained()."
        )
    gen_config = (
        generation_config
        if generation_config is not None
        else OmniVoiceGenerationConfig.from_dict(kwargs)
    )

    self.eval()

    full_task = self._preprocess_all(
        text=text,
        language=language,
        ref_text=ref_text,
        ref_audio=ref_audio,
        voice_clone_prompt=voice_clone_prompt,
        instruct=instruct,
        preprocess_prompt=gen_config.preprocess_prompt,
        speed=speed,
        duration=duration,
    )

    short_idx, long_idx = full_task.get_indices(
        gen_config, self.audio_tokenizer.config.frame_rate
    )

    results = [None] * full_task.batch_size

    if short_idx:
        short_task = full_task.slice_task(short_idx)
        short_results = self._generate_iterative(short_task, gen_config)
        for idx, res in zip(short_idx, short_results):
            results[idx] = res

    if long_idx:
        long_task = full_task.slice_task(long_idx)
        long_results = self._generate_chunked(long_task, gen_config)
        for idx, res in zip(long_idx, long_results):
            results[idx] = res

    for i in range(full_task.batch_size):
        assert results[i] is not None, f"Result {i} was not generated"

    return _batched_decode_and_post_process(
        self, results, full_task.ref_rms, gen_config
    )
