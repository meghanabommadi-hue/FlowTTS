#!/usr/bin/env python3
"""Trace the language parameter from the request all the way into generate().

The language conditions OmniVoice's phoneme choices, so if it is being dropped
or silently rewritten somewhere in the request path the audio degrades in a way
that is hard to attribute. This wraps ``model.generate`` and prints exactly what
arrives, for each way a caller can specify (or omit) a language.

It also checks that the value actually *changes the audio* — a parameter that is
plumbed through but ignored by the model looks identical to one that works.

    python -m flowtts.test.trace_language --voice anika
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="anika")
    ap.add_argument("--num-step", type=int, default=8)
    args = ap.parse_args()

    from flowtts.synthesis.models import OmniVoiceSynthesizer
    from flowtts.synthesis.omnivoice_engine import GenParams

    seen: list[dict] = []

    async def run() -> None:
        synth = OmniVoiceSynthesizer()
        await synth.initialize()
        model = synth.engine.model

        original = model.generate

        def traced(**kwargs):
            seen.append({
                "text": (kwargs.get("text") or [""])[0][:44],
                "language": kwargs.get("language", "<absent>"),
                "instruct": kwargs.get("instruct", "<absent>"),
                "has_prompt": kwargs.get("voice_clone_prompt") is not None,
            })
            return original(**kwargs)

        model.generate = traced
        params = GenParams.build({"num_step": args.num_step})

        HINDI = "आपका बकाया दो हज़ार रुपये है।"
        MARATHI = "तुमच्या खात्यात दोन हजार रुपये आहेत."
        TAMIL = "உங்கள் கணக்கில் இரண்டு ஆயிரம் ரூபாய் உள்ளது."
        ENGLISH = "Your balance is two thousand rupees."

        cases = [
            ("explicit hi",           HINDI,   "hi"),
            ("explicit mr",           MARATHI, "mr"),
            ("explicit ta",           TAMIL,   "ta"),
            ("explicit en",           ENGLISH, "en"),
            ("language name 'hindi'", HINDI,   "hindi"),
            ("ISO-639-3 remap 'or'",  "ଆପଣଙ୍କ ବାକି ଦୁଇ ହଜାର ଟଙ୍କା ଅଛି।", "or"),
            ("omitted (detected)",    HINDI,   None),
            ("omitted, Tamil text",   TAMIL,   None),
        ]

        print("\nWHAT REACHES model.generate():")
        print(f"  {'case':24} {'requested':12} -> {'delivered':10}  {'text'}")
        print("  " + "-" * 76)
        for label, text, language in cases:
            before = len(seen)
            await synth.synthesize(text, voice_id=args.voice, language=language,
                                   params=params)
            for call in seen[before:]:
                delivered = call["language"]
                if isinstance(delivered, list):
                    delivered = delivered[0]
                print(f"  {label:24} {str(language):12} -> {str(delivered):10}  "
                      f"{call['text']}")

        # A parameter that is plumbed through but ignored looks identical to one
        # that works, so check the audio actually differs.
        print("\nDOES IT CHANGE THE AUDIO?")
        text = HINDI
        outputs = {}
        for language in ("hi", "mr", "ta", "en", None):
            waves = []
            for _ in range(2):
                wav = await synth.synthesize(text, voice_id=args.voice,
                                             language=language, params=params)
                waves.append(np.asarray(wav, dtype=np.float32))
            outputs[str(language)] = waves

        base = outputs["hi"][0]
        print(f"  {'language':10} {'seconds':>8} {'peak':>7}  {'vs hi (len delta)':>18}")
        print("  " + "-" * 52)
        for language, waves in outputs.items():
            wav = waves[0]
            delta = abs(len(wav) - len(base)) / max(1, len(base)) * 100
            print(f"  {language:10} {len(wav) / 24000:>8.2f} "
                  f"{float(np.abs(wav).max()):>7.3f}  {delta:>17.1f}%")

        # Run-to-run variation for the SAME language, as a baseline for the above.
        same = outputs["hi"]
        spread = abs(len(same[0]) - len(same[1])) / max(1, len(same[0])) * 100
        print(f"\n  same language, two runs: {spread:.1f}% length difference "
              f"(the noise floor for the column above)")

    asyncio.run(run())


if __name__ == "__main__":
    main()
