#!/usr/bin/env python3
"""Isolate which stage of the accelerated backbone diverges from the real model.

The chain is  transformers Qwen3  ->  Qwen3Backbone (torch)  ->  ONNX  ->  TRT
engine, and a mismatch at the end says nothing about where it started. This
checks each hop against the one before it and prints tensor statistics, so a
failure localizes to a single stage instead of "the engine is wrong".

    python -m flowtts.test.diagnose_backbone
    python -m flowtts.test.diagnose_backbone --seq 128 --batch 2 --pad
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F


def stats(name: str, tensor: torch.Tensor) -> None:
    flat = tensor.detach().float().flatten()
    print(f"  {name:34} shape={tuple(tensor.shape)} mean={flat.mean():+.5f} "
          f"std={flat.std():.5f} absmax={flat.abs().max():.4f} "
          f"zeros={100.0 * (flat == 0).float().mean():.1f}%")


def compare(name: str, reference: torch.Tensor, got: torch.Tensor) -> float:
    a, b = reference.detach().float().flatten(), got.detach().float().flatten()
    if a.shape != b.shape:
        print(f"  {name:34} SHAPE MISMATCH {tuple(reference.shape)} vs {tuple(got.shape)}")
        return 0.0
    cosine = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    print(f"  {name:34} cosine={cosine:.6f}  max_abs_diff={(a - b).abs().max():.5f}")
    return cosine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--pad", action="store_true",
                    help="make row 1 shorter, to exercise the input_lengths mask")
    ap.add_argument("--engine-dir", default=None)
    args = ap.parse_args()

    from flowtts.core.config import resolve_model_source, settings
    from flowtts.trt.backbone import BackboneConfig, Qwen3Backbone, precompute_rope

    device = settings.omnivoice.device
    dtype = getattr(torch, settings.omnivoice.dtype)

    from omnivoice import OmniVoice
    model = OmniVoice.from_pretrained(
        resolve_model_source(), device_map=device, dtype=dtype,
        load_asr=False, trust_remote_code=True,
    )
    cfg = BackboneConfig.from_hf(model.llm.config)
    print(f"\nconfig: {cfg}\n")

    batch, seq = args.batch, args.seq
    lengths = torch.full((batch,), seq, dtype=torch.int32, device=device)
    if args.pad and batch > 1:
        lengths[1] = seq // 2

    embeds = torch.randn(batch, seq, cfg.hidden_size, device=device, dtype=dtype)
    mask = torch.zeros(batch, 1, seq, seq, dtype=torch.bool, device=device)
    for i in range(batch):
        n = int(lengths[i])
        mask[i, :, :n, :n] = True

    print(f"input: batch={batch} seq={seq} lengths={lengths.tolist()}")
    stats("inputs_embeds", embeds)

    # ---- Stage 1: transformers Qwen3 (the reference) ----
    with torch.no_grad():
        reference = model.llm(inputs_embeds=embeds, attention_mask=mask,
                              return_dict=True)[0]
    print("\n[1] transformers Qwen3Model")
    stats("reference", reference)

    # ---- Stage 2: the PyTorch mirror ----
    backbone = Qwen3Backbone.from_llm(model.llm, cfg)
    cos, sin = precompute_rope(cfg, device, dtype, max_seq_len=max(seq, 16))
    rope_cos = cos[:seq].unsqueeze(0).expand(batch, -1, -1).contiguous()
    rope_sin = sin[:seq].unsqueeze(0).expand(batch, -1, -1).contiguous()
    with torch.no_grad():
        mirrored = backbone(embeds, rope_cos, rope_sin, lengths)
    print("\n[2] Qwen3Backbone (torch mirror)  vs  transformers")
    stats("mirror", mirrored)
    # Compare only the valid region: padded positions are legitimately allowed
    # to differ, since neither path's consumer ever reads them.
    for i in range(batch):
        n = int(lengths[i])
        compare(f"row {i} valid[:{n}]", reference[i, :n], mirrored[i, :n])

    # ---- Stage 3: the TensorRT engine ----
    engine_dir = args.engine_dir or settings.omnivoice.trt_engine_dir
    try:
        from flowtts.trt.runtime import TRTBackbone
        engine = TRTBackbone(engine_dir, cfg, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[3] TensorRT engine unavailable: {type(exc).__name__}: {exc}")
        return

    print(f"\n[3] TensorRT engine  ({engine.info()})")
    with torch.no_grad():
        engine_out = engine(embeds, lengths)
    stats("engine", engine_out)
    for i in range(batch):
        n = int(lengths[i])
        compare(f"row {i} vs mirror[:{n}]", mirrored[i, :n], engine_out[i, :n])
        compare(f"row {i} vs reference[:{n}]", reference[i, :n], engine_out[i, :n])

    # Does the engine react to input_lengths at all? If the mask was folded into
    # a constant at export time it will return identical output for both.
    print("\n[4] does the engine honour input_lengths?")
    with torch.no_grad():
        full = engine(embeds, torch.full((batch,), seq, dtype=torch.int32, device=device))
        half = engine(embeds, torch.full((batch,), seq // 2, dtype=torch.int32, device=device))
    delta = (full - half).abs().max().item()
    print(f"  max|out(len=seq) - out(len=seq/2)| = {delta:.5f} "
          f"{'(reacts — good)' if delta > 1e-3 else '(NO REACTION — mask is baked in)'}")


if __name__ == "__main__":
    main()
