#!/usr/bin/env python3
"""Measure what batching is actually worth on this model, and what padding costs.

The batcher makes two judgement calls that are only defensible with numbers:

  * how large a batch is worth forming — if ``generate()`` scales nearly
    linearly with batch size there is little to gain and plenty of latency to
    lose, whereas a flat region means batching is close to free throughput;
  * how much length mismatch to tolerate — ``generate()`` pads every item to the
    longest one, so batching a short chunk behind a long one makes the short one
    pay the long one's cost. Whether that still beats running them separately
    depends on how much of the cost is per-call overhead.

This measures both directly, so ``length_bucket_ratio`` and ``max_batch`` are
set from data rather than intuition.

    python -m flowtts.test.bench_batching
"""

from __future__ import annotations

import argparse
import time

UNIFORM = "आपका बकाया दो हज़ार पाँच सौ रुपये है, कृपया आज ही भुगतान करें। धन्यवाद।"
SHORT = "कृपया थोड़ा इंतज़ार करें।"
LONG = " ".join([UNIFORM] * 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="anika")
    ap.add_argument("--language", default="hi")
    ap.add_argument("--num-step", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sizes", default="1,2,4,8,16,24")
    args = ap.parse_args()

    import torch
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    from flowtts.core.config import resolve_model_source, settings
    from flowtts.trt import patch_model
    from flowtts.voices.registry import VoiceRegistry

    registry = VoiceRegistry(settings.voices.voices_dir, settings.voices.default_voice)
    prompt = registry.prompt(args.voice)

    model = OmniVoice.from_pretrained(
        resolve_model_source(), device_map=settings.omnivoice.device,
        dtype=getattr(torch, settings.omnivoice.dtype),
        load_asr=False, trust_remote_code=True,
    )
    result = patch_model(model, settings.omnivoice)
    print(f"backbone: {result.backend} (cosine {result.cosine})")

    config = OmniVoiceGenerationConfig(
        num_step=args.num_step, guidance_scale=2.0,
        postprocess_output=False, pad_duration=0.0, fade_duration=0.0,
    )

    def run(texts: list[str]) -> float:
        """Median wall-clock milliseconds for one generate() over *texts*."""
        kwargs = dict(voice_clone_prompt=[prompt] * len(texts),
                      language=[args.language] * len(texts),
                      generation_config=config)
        model.generate(text=texts, **kwargs)          # warm
        times = []
        for _ in range(args.reps):
            started = time.perf_counter()
            model.generate(text=texts, **kwargs)
            times.append((time.perf_counter() - started) * 1000)
        return sorted(times)[len(times) // 2]

    print("\nUNIFORM-LENGTH BATCHES — the best case for batching")
    print(f"  {'batch':>5} {'total ms':>9} {'ms/item':>8} {'vs sequential':>14}")
    print("  " + "-" * 40)
    base = None
    for size in [int(x) for x in args.sizes.split(",")]:
        ms = run([UNIFORM] * size)
        if base is None:
            base = ms
        print(f"  {size:>5} {ms:>9.1f} {ms / size:>8.1f} {base * size / ms:>13.2f}x")

    print("\nMIXED LENGTHS — what the length bucket currently refuses to batch")
    short_ms = run([SHORT])
    long_ms = run([LONG])
    together = run([SHORT, LONG])
    print(f"  short alone                 {short_ms:>8.1f} ms")
    print(f"  long alone                  {long_ms:>8.1f} ms")
    print(f"  run separately (sum)        {short_ms + long_ms:>8.1f} ms")
    print(f"  batched together            {together:>8.1f} ms"
          f"   → {'batching wins' if together < short_ms + long_ms else 'padding wins'}")

    mixed = [SHORT, UNIFORM, LONG, SHORT, UNIFORM, LONG, SHORT, UNIFORM]
    sequential = sum(run([t]) for t in mixed)
    batched = run(mixed)
    print(f"\n  8 mixed, one at a time      {sequential:>8.1f} ms")
    print(f"  8 mixed, single batch       {batched:>8.1f} ms"
          f"   → {sequential / batched:.2f}x")


if __name__ == "__main__":
    main()
