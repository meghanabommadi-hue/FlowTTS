"""Equivalence tests for `flowtts/dhvaani/model/ops.py`.

The vectorised ops replace Python-loop implementations from upstream ZipVoice
that ran per request on the CPU. "Faster" is only useful if it is also
*identical*, so every function is checked against the upstream reference that
ops.py ships alongside it (`_ref_*`), over randomised shapes including the
degenerate cases.

All of this runs on CPU; no GPU is required.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from flowtts.dhvaani.model import ops  # noqa: E402


def _rand_case(rng):
    B = rng.randint(1, 6)
    tokens_lens = torch.tensor([rng.randint(1, 60) for _ in range(B)])
    # Mix normal lengths with tiny ones so the avg==0 branch is exercised.
    features_lens = torch.tensor(
        [
            rng.choice([rng.randint(1, 5), rng.randint(1, 400)])
            for _ in range(B)
        ]
    )
    return features_lens, tokens_lens


def test_make_pad_mask_matches_upstream():
    rng = random.Random(0)
    for _ in range(200):
        lens = torch.tensor([rng.randint(1, 100) for _ in range(rng.randint(1, 8))])
        max_len = int(lens.max()) + rng.randint(0, 50)
        assert torch.equal(ops.make_pad_mask(lens, max_len),
                           ops._ref_make_pad_mask(lens, max_len))


def test_make_pad_mask_does_not_read_lengths():
    """The fast version must take max_len as a Python int -- reading it off the
    tensor is what forced a device sync in the upstream implementation."""
    lens = torch.tensor([3, 5])
    m = ops.make_pad_mask(lens, 6)
    assert m.shape == (2, 6)
    assert m[0].tolist() == [False, False, False, True, True, True]


def test_token_frame_index_matches_upstream():
    rng = random.Random(1)
    checked = 0
    for _ in range(400):
        features_lens, tokens_lens = _rand_case(rng)
        num_frames = int(features_lens.max())
        ref_dur = ops._ref_prepare_avg_tokens_durations(features_lens, tokens_lens)
        ref = ops._ref_get_tokens_index(ref_dur, num_frames)
        got = ops.token_frame_index(features_lens, tokens_lens, num_frames)
        assert torch.equal(ref, got), (features_lens, tokens_lens, num_frames)
        checked += 1
    assert checked == 400


def test_token_frame_index_degenerate_more_tokens_than_frames():
    """avg == 0: upstream assigns every frame to the trailing remainder slot."""
    features_lens = torch.tensor([3])
    tokens_lens = torch.tensor([10])
    idx = ops.token_frame_index(features_lens, tokens_lens, 3)
    assert idx.tolist() == [[10, 10, 10]]


def test_predict_feature_lens_matches_expression():
    rng = random.Random(2)
    for _ in range(300):
        B = rng.randint(1, 5)
        pfl = torch.tensor([rng.randint(20, 400) for _ in range(B)])
        ptl = torch.tensor([rng.randint(1, 80) for _ in range(B)])
        tl = torch.tensor([rng.randint(1, 200) for _ in range(B)])
        speed = rng.choice([0.75, 1.0, 1.25, 2.0])
        ref = pfl + torch.ceil(pfl / ptl * tl / speed).to(torch.int64)
        assert torch.equal(ops.predict_feature_lens(pfl, ptl, tl, speed), ref)


def test_pad_token_ids_appends_one_pad():
    ids = [[1, 2, 3], [4]]
    padded, lens = ops.pad_token_ids(ids, pad_id=0, device=torch.device("cpu"))
    # +1 pad on every sequence, then padded to the max -- the extra slot is what
    # token_frame_index's trailing remainder points at.
    assert padded.shape == (2, 4)
    assert lens.tolist() == [3, 1]
    assert padded[0].tolist() == [1, 2, 3, 0]
    assert padded[1].tolist() == [4, 0, 0, 0]


def test_build_text_condition_shapes_and_masking():
    B, S, C, T = 3, 12, 100, 64
    embed = torch.randn(B, S, C)
    tokens_lens = torch.tensor([5, 8, 11])
    features_lens = torch.tensor([40, 64, 20])
    tc, mask = ops.build_text_condition(embed, tokens_lens, features_lens, T)
    assert tc.shape == (B, T, C)
    assert mask.shape == (B, T)
    assert mask[0, 39].item() is False and mask[0, 40].item() is True
    assert not mask[1].any()


def test_build_speech_condition_zeroes_past_prompt():
    B, Tp, C, T = 2, 10, 100, 32
    prompt = torch.randn(B, Tp, C)
    lens = torch.tensor([10, 4])
    sc = ops.build_speech_condition(prompt, lens, T)
    assert sc.shape == (B, T, C)
    assert torch.allclose(sc[0, :10], prompt[0, :10])
    assert torch.all(sc[0, 10:] == 0)
    assert torch.allclose(sc[1, :4], prompt[1, :4])
    assert torch.all(sc[1, 4:] == 0)


def _ref_cfg(model_fn, t_scalar, x, text_c, speech_c, mask, gs):
    """Direct transcription of solver.DiffusionModel.forward (scalar t only)."""
    x2 = torch.cat([x, x], dim=0)
    mask2 = torch.cat([mask, mask], dim=0)
    text2 = torch.cat([torch.zeros_like(text_c), text_c], dim=0)
    if t_scalar > 0.5:
        speech2 = torch.cat([torch.zeros_like(speech_c), speech_c], dim=0)
        gs_eff = gs
    else:
        gs_eff = gs * 2
        speech2 = torch.cat([speech_c, speech_c], dim=0)
    v = model_fn(x2, text2, speech2, mask2)
    uncond, cond = v.chunk(2, dim=0)
    return (1 + gs_eff) * cond - gs_eff * uncond


@pytest.mark.parametrize("t_val", [0.2, 0.5, 0.51, 0.9])
def test_cfg_expand_combine_matches_upstream_for_scalar_t(t_val):
    """Our per-sample CFG must reduce exactly to upstream's scalar-t behaviour."""
    torch.manual_seed(0)
    B, T, C = 3, 16, 100
    x = torch.randn(B, T, C)
    text_c = torch.randn(B, T, C)
    speech_c = torch.randn(B, T, C)
    mask = torch.zeros(B, T, dtype=torch.bool)
    gs_val = 1.0

    def model_fn(xx, tt, ss, mm):
        # Any deterministic function of the inputs exercises the algebra.
        return xx * 0.5 + tt * 0.25 + ss * 0.125

    t = torch.full((B,), t_val)
    gs = torch.full((B,), gs_val)
    x2, text2, speech2, mask2, t2, gs_eff = ops.cfg_expand(x, text_c, speech_c, mask, t, gs)
    v2 = model_fn(x2, text2, speech2, mask2)
    got = ops.cfg_combine(v2, gs_eff)

    ref = _ref_cfg(model_fn, t_val, x, text_c, speech_c, mask, gs_val)
    assert torch.allclose(got, ref, atol=1e-5)


def test_cfg_expand_handles_mixed_timesteps():
    """The whole point of the generalisation: rows either side of t=0.5 in one
    batch, which upstream's `assert t.dim() == 0` forbids."""
    torch.manual_seed(1)
    B, T, C = 4, 8, 100
    x = torch.randn(B, T, C)
    text_c = torch.randn(B, T, C)
    speech_c = torch.randn(B, T, C)
    mask = torch.zeros(B, T, dtype=torch.bool)
    t = torch.tensor([0.1, 0.6, 0.4, 0.95])
    gs = torch.full((B,), 1.0)

    x2, text2, speech2, mask2, t2, gs_eff = ops.cfg_expand(x, text_c, speech_c, mask, t, gs)
    assert x2.shape[0] == 2 * B
    # t <= 0.5 -> guidance doubled and speech condition kept in both branches
    assert gs_eff.tolist() == [2.0, 1.0, 2.0, 1.0]
    assert torch.allclose(speech2[0], speech_c[0])   # low t: kept
    assert torch.all(speech2[1] == 0)                # high t: zeroed
    assert torch.allclose(speech2[2], speech_c[2])
    assert torch.all(speech2[3] == 0)


def test_get_time_steps_matches_upstream_formula():
    for num_step in (4, 8, 16):
        for shift in (0.5, 1.0):
            ts = ops.get_time_steps(num_step, shift)
            ref = torch.linspace(0.0, 1.0, num_step + 1)
            ref = shift * ref / (1 + (shift - 1) * ref)
            assert torch.allclose(ts, ref, atol=1e-6)
            assert ts.numel() == num_step + 1
            assert float(ts[0]) == 0.0 and abs(float(ts[-1]) - 1.0) < 1e-6


def test_euler_update_per_sample_dt():
    x = torch.zeros(3, 4, 100)
    v = torch.ones(3, 4, 100)
    dt = torch.tensor([0.1, 0.2, 0.3])
    out = ops.euler_update(x, v, dt)
    assert torch.allclose(out[0], torch.full((4, 100), 0.1))
    assert torch.allclose(out[1], torch.full((4, 100), 0.2))
    assert torch.allclose(out[2], torch.full((4, 100), 0.3))
