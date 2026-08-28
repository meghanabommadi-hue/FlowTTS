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
    ap.add_argument("--asr-url", default=None,
                    help="OpenAI-compatible ASR root; enables WER/CER in TB")
    ap.add_argument("--asr-model", default="Axiveri/NaijaVox-2.0")
    ap.add_argument("--metrics-out", default=None, help="write metrics json here")
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
    gen = []          # (reference_text, audio, language) for metric scoring
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
            gen.append((it["text"], wav, sr, it.get("language", "xx")))
            ok += 1
            print(f"  ok {it['id']} ({len(wav)/sr:.1f}s)", flush=True)
        except Exception as e:
            fail += 1
            print(f"  FAIL {it['id']}: {e!r}", flush=True)
            traceback.print_exc()

    # ---- tts-bench metrics on the same generations -------------------------
    # Reuses tts-bench's own WER/CER implementation so these curves are directly
    # comparable to the benchmark leaderboard, rather than a second opinion.
    scored = {}
    if a.asr_url and gen:
        try:
            scored = _score_with_tts_bench(gen, a.asr_url, a.asr_model)
            for k, v in scored.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"eval_quality/{k}", float(v), a.step)
            print("  tts-bench metrics: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in scored.items()
                              if isinstance(v, (int, float))), flush=True)
        except Exception as e:
            print(f"  WARN tts-bench scoring failed: {e!r}", flush=True)

    writer.add_scalar("eval_audio/num_ok", ok, a.step)
    writer.add_scalar("eval_audio/num_failed", fail, a.step)
    writer.flush(); writer.close()
    if a.metrics_out:
        try:
            with open(a.metrics_out, "w") as f:
                json.dump({"step": a.step, "ok": ok, "failed": fail, **scored}, f)
        except OSError:
            pass
    print(f"eval inference: {ok} ok, {fail} failed in {time.time()-t0:.0f}s")
    return 0 if ok else 3


def _score_with_tts_bench(gen, asr_url, asr_model):
    """WER/CER over the eval generations, using tts-bench's own scorer."""
    from tts_bench.core.audio import AudioBuffer
    from tts_bench.metrics.asr import get_asr

    asr = get_asr("vllm", base_url=asr_url, model=asr_model)
    refs, hyps, langs = [], [], []
    for ref, wav, sr, lang in gen:
        buf = AudioBuffer.from_array(np.asarray(wav, dtype=np.float32), int(sr))
        hyp = asr.transcribe(buf, language=lang, expected_text=ref)
        hyps.append(hyp if isinstance(hyp, str) else (hyp or {}).get("text", ""))
        refs.append(ref)
        langs.append(lang)

    mos_vals = []
    try:
        mos_vals = _predict_mos([(w, sr) for _, w, sr, _ in gen])
    except Exception as e:
        print(f"  MOS unavailable: {e!r}", flush=True)

    import jiwer
    def _norm(t):
        return " ".join(str(t or "").lower().split())
    pairs = [(_norm(r), _norm(h)) for r, h in zip(refs, hyps) if _norm(r)]
    if not pairs:
        return {}
    R = [p[0] for p in pairs]; H = [p[1] for p in pairs]
    out = {"wer": float(jiwer.wer(R, H)), "cer": float(jiwer.cer(R, H))}
    out["sentence_error_rate"] = sum(1 for r, h in pairs if r != h) / len(pairs)
    # per-language too, so a regression in one language is visible
    by = {}
    for (r, h), lg in zip(pairs, langs):
        by.setdefault(lg, [[], []])
        by[lg][0].append(r); by[lg][1].append(h)
    for lg, (rr, hh) in by.items():
        try:
            out[f"wer_{lg}"] = float(jiwer.wer(rr, hh))
        except Exception:
            pass
    if mos_vals:
        out["mos"] = float(sum(mos_vals) / len(mos_vals))
    out.update(_composite(out))
    return out


def _predict_mos(clips):
    """Reference-free naturalness (DistillMOS), same predictor tts-bench uses."""
    import torch
    import distillmos
    import librosa

    model = _predict_mos._m if hasattr(_predict_mos, "_m") else None
    if model is None:
        model = distillmos.ConvTransformerSQAModel()
        model.eval()
        _predict_mos._m = model
    scores = []
    with torch.no_grad():
        for wav, sr in clips:
            x = np.asarray(wav, dtype=np.float32)
            if sr != 16000:
                x = librosa.resample(x, orig_sr=sr, target_sr=16000,
                                     res_type="soxr_hq")
            t = torch.from_numpy(x).float().unsqueeze(0)
            scores.append(float(model(t).squeeze().item()))
    return scores


# Weights for the composite. WER dominates because for TTS the first question
# is whether the words come out right; MOS carries real weight because a model
# can be intelligible and still sound wrong.
COMPOSITE_W = {"wer": 0.40, "cer": 0.20, "sentence_error_rate": 0.15, "mos": 0.25}


def _composite(m):
    """One 0-1 quality score (higher better) from the tts-bench metrics.

    This is NOT a training objective - it is not differentiable and costs
    seconds per sample. It is a model-selection signal: eval loss proved a poor
    proxy for quality (it barely moved while WER halved), so checkpoints are
    chosen on this instead.
    """
    parts, used = {}, 0.0
    if m.get("wer") is not None:
        parts["wer"] = max(0.0, 1.0 - float(m["wer"]))
    if m.get("cer") is not None:
        parts["cer"] = max(0.0, 1.0 - float(m["cer"]))
    if m.get("sentence_error_rate") is not None:
        parts["sentence_error_rate"] = max(0.0, 1.0 - float(m["sentence_error_rate"]))
    if m.get("mos") is not None:
        parts["mos"] = min(1.0, max(0.0, (float(m["mos"]) - 1.0) / 4.0))  # 1-5 -> 0-1
    if not parts:
        return {}
    total_w = sum(COMPOSITE_W[k] for k in parts)
    score = sum(COMPOSITE_W[k] * v for k, v in parts.items()) / total_w
    out = {f"score_{k}": v for k, v in parts.items()}
    out["composite"] = score
    out["composite_coverage"] = total_w      # 1.0 when every component present
    return out


if __name__ == "__main__":
    sys.exit(main())
