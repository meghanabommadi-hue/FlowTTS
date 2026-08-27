#!/usr/bin/env python3
"""Synthesize the fixed eval set from a checkpoint and log it to TensorBoard.

Run as a SEPARATE PROCESS from the trainer, deliberately:

  * builder.py loads the training model with train=True, which leaves
    text_tokenizer/audio_tokenizer/duration_estimator as None, and generate()
    hard-raises in that state - so a second, inference-mode model is required
    either way.
  * Keeping it out-of-process means a synthesis OOM or a vocoder crash cannot
    take the training run down with it.

Writes into the SAME TensorBoard log dir as training, so audio appears on the
step axis next to the loss curves.
"""
from __future__ import annotations

import argparse, json, os, sys, time, traceback

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-set", required=True, help="json list of eval prompts")
    ap.add_argument("--logdir", required=True, help="tensorboard log dir")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--wav-out", default=None, help="also write wavs here")
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--guidance-scale", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-items", type=int, default=12)
    a = ap.parse_args()

    import torch
    from torch.utils.tensorboard import SummaryWriter
    from omnivoice.models.omnivoice import OmniVoice

    items = json.load(open(a.eval_set))[: a.max_items]
    if not items:
        print("eval set empty, nothing to do")
        return 0

    t0 = time.time()
    print(f"loading {a.checkpoint} on {a.device}", flush=True)
    model = OmniVoice.from_pretrained(a.checkpoint, device_map=a.device,
                                      dtype=torch.float16)
    sr = model.sampling_rate
    writer = SummaryWriter(log_dir=a.logdir)
    if a.wav_out:
        os.makedirs(a.wav_out, exist_ok=True)

    ok = fail = 0
    for it in items:
        tag = f"eval_audio/{it.get('language','xx')}/{it['id']}"
        try:
            with torch.inference_mode():
                audios = model.generate(
                    text=it["text"],
                    language=it.get("language"),
                    ref_audio=it.get("ref_audio"),
                    ref_text=it.get("ref_text"),
                    num_step=a.num_step,
                    guidance_scale=a.guidance_scale,
                )
            wav = np.asarray(audios[0], dtype=np.float32)
            peak = float(np.max(np.abs(wav))) if wav.size else 0.0
            if not np.isfinite(wav).all() or peak < 1e-5:
                # A silent or NaN sample is a real signal about training health,
                # so record it as a metric instead of silently logging noise.
                print(f"  WARN degenerate output for {it['id']} peak={peak:.2e}")
                writer.add_scalar(f"eval_audio_degenerate/{it['id']}", 1.0, a.step)
                fail += 1
                continue
            writer.add_audio(tag, wav / max(peak, 1e-6) * 0.95, a.step, sample_rate=sr)
            writer.add_text(f"eval_text/{it.get('language','xx')}/{it['id']}",
                            it["text"], a.step)
            if a.wav_out:
                import soundfile as sf
                sf.write(os.path.join(a.wav_out, f"step{a.step:07d}_{it['id']}.wav"),
                         wav, sr)
            ok += 1
            print(f"  ok {it['id']} ({len(wav)/sr:.1f}s)", flush=True)
        except Exception as e:
            fail += 1
            print(f"  FAIL {it['id']}: {e!r}", flush=True)
            traceback.print_exc()

    writer.add_scalar("eval_audio/num_ok", ok, a.step)
    writer.add_scalar("eval_audio/num_failed", fail, a.step)
    writer.flush(); writer.close()
    print(f"eval inference: {ok} ok, {fail} failed in {time.time()-t0:.0f}s")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
