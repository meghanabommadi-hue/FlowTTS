"""Pipeline position: MODEL OPS — vectorised, sync-free replacements for the
per-request tensor plumbing ZipVoice does in Python loops.

Why this module exists
----------------------
Upstream ``zipvoice.utils.common`` builds the text conditioning with nested
Python loops on the CPU::

    def get_tokens_index(durations, num_frames):
        ans = torch.zeros(B, num_frames, dtype=torch.int64)
        for b in range(B):
            cur = 0
            for i, d in enumerate(durations[b]):
                ans[b, cur:cur + d] = i
                cur += d
        return ans

That is O(B * num_tokens) Python-level slice assignments per batch, on the CPU,
followed by an H2D copy -- several milliseconds per request at 100+ tokens, and
it holds the GIL. ``make_pad_mask`` additionally calls ``lengths.max()`` and
compares it to a Python int, forcing a device synchronisation on every call.

Every function here is a drop-in, numerically identical replacement that runs
entirely on the GPU with no host synchronisation. Equivalence against the
upstream implementations is asserted by ``flowtts/dhvaani/test/test_ops.py``.

Consumers
---------
model/text_encoder.py  -> build_text_condition, predict_feature_lens
engine/scheduler.py    -> make_pad_mask, cfg_split_inputs, euler_update
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Padding masks
# ---------------------------------------------------------------------------
def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """``(B, max_len)`` bool mask, True at padded positions.

    Unlike ``zipvoice.utils.common.make_pad_mask`` this takes ``max_len`` as a
    required Python int, so it never calls ``lengths.max()`` and never
    synchronises the device. Callers always know the padded width -- it is the
    bucket size.
    """
    assert lengths.ndim == 1, lengths.shape
    rng = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
    return rng.unsqueeze(0) >= lengths.unsqueeze(1)


def pad_token_ids(
    token_ids: Sequence[Sequence[int]],
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a ragged list of id sequences to ``(B, S)``.

    Mirrors ``zipvoice.utils.common.pad_labels`` exactly, including the extra
    single pad token appended to every sequence. That extra slot is load-bearing:
    :func:`token_frame_index` addresses index ``len(tokens)`` for the trailing
    "remainder" frames, which would be out of bounds without it.

    Returns:
        (padded_ids ``(B, S)`` int64, true_lens ``(B,)`` int64) where
        ``true_lens[i] == len(token_ids[i])`` *without* the appended pad.
    """
    lens = [len(t) for t in token_ids]
    width = max(lens) + 1  # +1 for the appended pad, matching upstream
    out = torch.full((len(token_ids), width), pad_id, dtype=torch.int64)
    for i, ids in enumerate(token_ids):
        if ids:
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.int64)
    return (
        out.to(device, non_blocking=True),
        torch.tensor(lens, dtype=torch.int64, device=device),
    )


# ---------------------------------------------------------------------------
# Duration prediction and token -> frame alignment
# ---------------------------------------------------------------------------
def predict_feature_lens(
    prompt_features_lens: torch.Tensor,
    prompt_tokens_lens: torch.Tensor,
    tokens_lens: torch.Tensor,
    speed: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Total frame count (prompt + generated) for each item.

    Identical to the ratio-duration branch of
    ``ZipVoice.forward_text_inference_ratio_duration``::

        features_lens = prompt_features_lens + ceil(
            prompt_features_lens / prompt_tokens_lens * tokens_lens / speed
        )

    The duration model is nothing more than "the generated audio has the same
    frames-per-character rate as the prompt", which is why a voice's prompt
    length and transcript length fully determine its speaking rate.
    """
    pt = prompt_tokens_lens.clamp(min=1).to(torch.float32)
    rate = prompt_features_lens.to(torch.float32) / pt          # frames per token
    gen = torch.ceil(rate * tokens_lens.to(torch.float32) / speed)
    return prompt_features_lens + gen.to(torch.int64)


def token_frame_index(
    features_lens: torch.Tensor,
    tokens_lens: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    """``(B, num_frames)`` int64 index: which token each frame reads from.

    Vectorised equivalent of
    ``get_tokens_index(prepare_avg_tokens_durations(features_lens, tokens_lens))``.

    Upstream gives every token the same duration ``avg = features_lens //
    tokens_lens`` and assigns any leftover frames to a synthetic trailing entry
    at index ``tokens_lens`` (which lands on the pad slot appended by
    :func:`pad_token_ids`). So::

        index[b, f] = min(f // avg[b], L[b])        when avg[b] > 0
        index[b, f] = L[b]                          when avg[b] == 0

    The ``avg == 0`` case happens when an item has more tokens than frames --
    upstream produces all-zero durations there, so every frame falls into the
    trailing entry. We reproduce that rather than silently diverging.
    """
    device = features_lens.device
    L = tokens_lens.clamp(min=1)
    avg = torch.div(features_lens, L, rounding_mode="floor")     # (B,)

    frames = torch.arange(num_frames, device=device, dtype=torch.int64)  # (T,)
    safe_avg = avg.clamp(min=1).unsqueeze(1)                     # (B, 1)
    idx = torch.div(frames.unsqueeze(0), safe_avg, rounding_mode="floor")
    idx = torch.minimum(idx, tokens_lens.unsqueeze(1))
    # avg == 0 -> every frame reads the trailing entry
    idx = torch.where(avg.unsqueeze(1) > 0, idx, tokens_lens.unsqueeze(1))
    return idx


def build_text_condition(
    embed: torch.Tensor,
    tokens_lens: torch.Tensor,
    features_lens: torch.Tensor,
    num_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Upsample encoded text ``(B, S, C)`` to frame rate ``(B, num_frames, C)``.

    Returns ``(text_condition, padding_mask)``.
    """
    idx = token_frame_index(features_lens, tokens_lens, num_frames)   # (B, T)
    gather_idx = idx.unsqueeze(-1).expand(embed.size(0), num_frames, embed.size(-1))
    text_condition = torch.gather(embed, dim=1, index=gather_idx)
    padding_mask = make_pad_mask(features_lens, num_frames)
    return text_condition, padding_mask


def build_speech_condition(
    prompt_features: torch.Tensor,
    prompt_features_lens: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    """Right-pad prompt mel to ``num_frames`` and zero everything past the prompt.

    Mirrors the ``speech_condition`` construction in ``ZipVoice.sample``: the
    generated region is exactly the zeroed tail, which is what tells the flow
    decoder where to infill.
    """
    B, T_p, C = prompt_features.shape
    out = prompt_features.new_zeros((B, num_frames, C))
    keep = min(T_p, num_frames)
    out[:, :keep] = prompt_features[:, :keep]
    # Zero any position at or beyond that item's true prompt length.
    beyond = make_pad_mask(prompt_features_lens, num_frames)          # (B, T)
    out = out.masked_fill(beyond.unsqueeze(-1), 0.0)
    return out


# ---------------------------------------------------------------------------
# Classifier-free guidance
# ---------------------------------------------------------------------------
def cfg_expand(
    x: torch.Tensor,
    text_condition: torch.Tensor,
    speech_condition: torch.Tensor,
    padding_mask: torch.Tensor,
    t: torch.Tensor,
    guidance_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the doubled ``[uncond; cond]`` batch for classifier-free guidance.

    Generalises ``solver.DiffusionModel.forward`` to *per-sample* timesteps.
    Upstream asserts ``t.dim() == 0`` because it only ever runs a homogeneous
    batch; our scheduler deliberately runs samples at different ODE steps in one
    forward pass, so the ``t > 0.5`` branch becomes an elementwise select:

        t > 0.5  ->  drop the speech condition in the uncond branch
        t <= 0.5 ->  keep it, and double the guidance scale instead

    Returns ``(x2, text2, speech2, mask2, t2, gs_eff)`` where the first half of
    the batch is unconditional and the second half is conditional.
    """
    B = x.shape[0]
    hi = (t > 0.5).view(B, 1, 1)

    x2 = torch.cat([x, x], dim=0)
    mask2 = torch.cat([padding_mask, padding_mask], dim=0)
    t2 = torch.cat([t, t], dim=0)

    text2 = torch.cat([torch.zeros_like(text_condition), text_condition], dim=0)

    speech_uncond = torch.where(hi, torch.zeros_like(speech_condition), speech_condition)
    speech2 = torch.cat([speech_uncond, speech_condition], dim=0)

    gs = guidance_scale.view(B)
    gs_eff = torch.where(hi.view(B), gs, gs * 2.0)
    return x2, text2, speech2, mask2, t2, gs_eff


def cfg_combine(
    v2: torch.Tensor, guidance_scale: torch.Tensor
) -> torch.Tensor:
    """``(1 + w) * v_cond - w * v_uncond`` from the doubled output."""
    v_uncond, v_cond = v2.chunk(2, dim=0)
    w = guidance_scale.view(-1, 1, 1).to(v2.dtype)
    return (1.0 + w) * v_cond - w * v_uncond


# ---------------------------------------------------------------------------
# Euler ODE stepping
# ---------------------------------------------------------------------------
def get_time_steps(
    num_step: int,
    t_shift: float = 1.0,
    t_start: float = 0.0,
    t_end: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``(num_step + 1,)`` timestep grid, identical to ``solver.get_time_steps``.

    ``t_shift < 1`` bunches the grid toward small t, where the trajectory is
    least certain and therefore benefits most from resolution.
    """
    ts = torch.linspace(t_start, t_end, num_step + 1, device=device, dtype=dtype)
    return t_shift * ts / (1 + (t_shift - 1) * ts)


def euler_update(
    x: torch.Tensor, v: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    """``x + v * dt`` with a per-sample ``dt`` of shape ``(B,)``."""
    return x + v * dt.view(-1, 1, 1).to(v.dtype)


# ---------------------------------------------------------------------------
# Reference (slow) implementations -- used only by the equivalence tests
# ---------------------------------------------------------------------------
def _ref_prepare_avg_tokens_durations(features_lens, tokens_lens) -> List[List[int]]:
    out = []
    for i in range(len(features_lens)):
        avg = int(features_lens[i]) // int(tokens_lens[i])
        out.append([avg] * int(tokens_lens[i]))
    return out


def _ref_get_tokens_index(durations: List[List[int]], num_frames: int) -> torch.Tensor:
    durations = [x + [num_frames - sum(x)] for x in durations]
    ans = torch.zeros(len(durations), num_frames, dtype=torch.int64)
    for b, this_dur in enumerate(durations):
        cur = 0
        for i, d in enumerate(this_dur):
            ans[b, cur : cur + d] = i
            cur += d
        assert cur == num_frames, (cur, num_frames)
    return ans


def _ref_make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    max_len = max(max_len, int(lengths.max()))
    n = lengths.size(0)
    rng = torch.arange(0, max_len, device=lengths.device)
    return rng.unsqueeze(0).expand(n, max_len) >= lengths.unsqueeze(-1)
