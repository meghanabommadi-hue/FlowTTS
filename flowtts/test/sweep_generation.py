#!/usr/bin/env python3
"""Sweep OmniVoice's generation settings for latency against output sanity.

Why this exists: ``num_step`` and ``guidance_scale`` are the two knobs that
decide both cost and quality, and they are not independent. This measures GPU
time per utterance at each combination and, at the same time, checks that the
output is actually audible — because the cheapest settings are cheap precisely
because the model has stopped saying anything.

A healthy OmniVoice utterance peaks around 0.4-0.6. Anything two orders of
magnitude below that is silence wearing the right shape, which is what
``guidance_scale=0`` produces: this model is trained with classifier-free
guidance and cannot be run without it, however tempting halving the batch is.

    python -m flowtts.test.sweep_generation --voice anika
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

# A short IVR turn and a medium collections prompt: the two shapes that dominate
# real traffic, and whose latency budgets differ most.
TEXTS = {
    "short": "नमस्ते, मैं आपकी कैसे मदद कर सकती हूं?",
    "medium": ("आपका बकाया दो हज़ार पाँच सौ रुपये है, "
               "कृपया आज ही भुगतान करें। धन्यवाद।"),
}

AUDIBLE_PEAK = 0.02


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=None)
    ap.add_argument("--language", default="hi")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", default="2,4,6,8,12,16,32")
    ap.add_argument("--guidance", default="0.0,0.5,1.0,1.5,2.0,3.0")
    args = ap.parse_args()

    import torch
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    from flowtts.core.config import resolve_model_source, settings
    from flowtts.voices.registry import VoiceRegistry

    registry = VoiceRegistry(settings.voices.voices_dir, settings.voices.default_voice)
    prompt = registry.prompt(args.voice) if args.voice else None

    model = OmniVoice.from_pretrained(
        resolve_model_source(), device_map=settings.omnivoice.device,
        dtype=getattr(torch, settings.omnivoice.dtype),
        load_asr=False, trust_remote_code=True,
    )

    steps = [int(x) for x in args.steps.split(",")]
    guidances = [float(x) for x in args.guidance.split(",")]

    def run(text: str, **overrides) -> tuple[float, float, float]:
        config = OmniVoiceGenerationConfig(
            postprocess_output=False, pad_duration=0.0, fade_duration=0.0, **overrides
        )
        started = time.perf_counter()
        audios = model.generate(text=[text],
                                voice_clone_prompt=[prompt] if prompt else None,
                                language=[args.language], generation_config=config)
        elapsed = (time.perf_counter() - started) * 1000
        wav = np.asarray(audios[0], dtype=np.float32).reshape(-1)
        peak = float(np.abs(wav).max()) if wav.size else 0.0
        return elapsed, peak, wav.size / model.sampling_rate

    # Prime kernels so the first measured cell is not paying the warmup cost.
    run(TEXTS["short"], num_step=8, guidance_scale=2.0)

    for name, text in TEXTS.items():
        print(f"\n=== {name}: {text[:60]!r} ===")
        print(f"  {'steps':>5} {'cfg':>5} {'ms/utt':>8} {'audio_s':>8} {'rtf':>6} "
              f"{'peak p50':>9} {'peak min':>9}  verdict")
        print("  " + "-" * 74, flush=True)
        for step in steps:
            for guidance in guidances:
                times, peaks, durations = [], [], []
                for _ in range(args.repeats):
                    ms, peak, seconds = run(text, num_step=step, guidance_scale=guidance)
                    times.append(ms)
                    peaks.append(peak)
                    durations.append(seconds)
                median_ms = statistics.median(times)
                audio_s = statistics.median(durations)
                verdict = "ok" if min(peaks) >= AUDIBLE_PEAK else "SILENT"
                print(f"  {step:>5} {guidance:>5.1f} {median_ms:>8.1f} {audio_s:>8.2f} "
                      f"{median_ms / 1000 / audio_s:>6.3f} "
                      f"{statistics.median(peaks):>9.4f} {min(peaks):>9.4f}  {verdict}",
                      flush=True)


if __name__ == "__main__":
    main()
