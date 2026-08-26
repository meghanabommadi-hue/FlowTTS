#!/usr/bin/env python3
"""Diagnose what the model actually returns for a given text chunk.

Bypasses the whole server and calls OmniVoice directly, printing the shape,
peak, RMS and duration of the raw waveform under several generation settings.
Used to tell "the service dropped the audio" apart from "the model produced
nothing", which look identical from the outside.

    python -m flowtts.test.diagnose_chunk --voice anika --text "नमस्ते,"
"""

from __future__ import annotations

import argparse

import numpy as np


def describe(name: str, wav) -> None:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        print(f"  {name:38} EMPTY")
        return
    peak = float(np.abs(wav).max())
    rms = float(np.sqrt((wav.astype(np.float64) ** 2).mean()))
    print(f"  {name:38} n={wav.size:>7}  {wav.size / 24000:5.2f}s  "
          f"peak={peak:.5f}  rms={rms:.5f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="नमस्ते,")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--language", default="hi")
    args = ap.parse_args()

    import torch
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    from flowtts.core.config import resolve_model_source, settings
    from flowtts.voices.registry import VoiceRegistry

    registry = VoiceRegistry(settings.voices.voices_dir, settings.voices.default_voice)
    prompt = registry.prompt(args.voice) if args.voice else None
    if prompt is not None:
        print(f"voice={args.voice}  ref_rms={prompt.ref_rms:.6f}  "
              f"ref_tokens={tuple(prompt.ref_audio_tokens.shape)}")

    model = OmniVoice.from_pretrained(
        resolve_model_source(), device_map=settings.omnivoice.device,
        dtype=getattr(torch, settings.omnivoice.dtype),
        load_asr=False, trust_remote_code=True,
    )
    print(f"model loaded: sr={model.sampling_rate} "
          f"frame_rate={model.audio_tokenizer.config.frame_rate}")
    print(f"\ntext={args.text!r} ({len(args.text)} chars)\n")

    cases = [
        ("num_step=8 cfg=2 post=on  pad=0.1", dict(num_step=8, guidance_scale=2.0,
                                                   postprocess_output=True)),
        ("num_step=8 cfg=2 post=off pad=0", dict(num_step=8, guidance_scale=2.0,
                                                 postprocess_output=False,
                                                 pad_duration=0.0, fade_duration=0.0)),
        ("num_step=4 cfg=0 post=off pad=0", dict(num_step=4, guidance_scale=0.0,
                                                 postprocess_output=False,
                                                 pad_duration=0.0, fade_duration=0.0)),
        ("num_step=4 cfg=0 post=on  pad=0.1", dict(num_step=4, guidance_scale=0.0,
                                                   postprocess_output=True)),
        ("num_step=8 cfg=0 postemp=0 post=off", dict(num_step=8, guidance_scale=0.0,
                                                     position_temperature=0.0,
                                                     class_temperature=0.0,
                                                     postprocess_output=False,
                                                     pad_duration=0.0, fade_duration=0.0)),
    ]

    for label, overrides in cases:
        config = OmniVoiceGenerationConfig(**overrides)
        for attempt in range(3):
            audios = model.generate(text=[args.text],
                                    voice_clone_prompt=[prompt] if prompt else None,
                                    language=[args.language],
                                    generation_config=config)
            describe(f"{label} #{attempt}", audios[0])

    # What target length does the estimator give this text?
    task = model._preprocess_all(
        text=[args.text], language=[args.language],
        voice_clone_prompt=[prompt] if prompt else None,
        preprocess_prompt=True,
    )
    print(f"\nestimated target frames: {task.target_lens} "
          f"(= {[t / 25 for t in task.target_lens]} s)")


if __name__ == "__main__":
    main()
