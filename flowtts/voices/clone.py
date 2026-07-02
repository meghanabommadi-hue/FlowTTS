#!/usr/bin/env python3
"""Offline voice-clone builder for OmniVoice.

Creates reusable ``<alias>.npz`` voice-clone artifacts that the running server
loads at startup and addresses by alias. Run this ONCE per voice (it needs the
GPU + the omnivoice package); the server itself never re-encodes reference audio.

Usage:
    # One voice (auto-transcribe the reference with Whisper if --ref-text omitted)
    python -m flowtts.voices.clone --add priya --ref-audio sample_files/priya.wav \
        --ref-text "नमस्ते, मैं प्रिया बोल रही हूँ।"

    # Build an npz for every audio file in sample_files/ (stem = alias)
    python -m flowtts.voices.clone --build-all

    # Build from a manifest that supplies ref_text per alias
    python -m flowtts.voices.clone --build-all --manifest voices/manifest.json

    # List installed voices
    python -m flowtts.voices.clone --list

manifest.json format:
    { "priya": {"ref_audio": "sample_files/priya.wav", "ref_text": "..."},
      "tara":  {"ref_audio": "sample_files/tara.wav"} }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flowtts.core.config import settings
from flowtts.voices.npz_io import load_voice_npz, save_voice_npz

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


# ---------------------------------------------------------------------------
# Model (loaded once, lazily — heavy GPU import kept out of module import time)
# ---------------------------------------------------------------------------
_model = None


def _get_model(load_asr: bool):
    """Load OmniVoice once. Whisper ASR is loaded only when transcription is needed
    (i.e. some voice has no ref_text), so fully-specified manifests avoid that download."""
    global _model
    if _model is not None:
        return _model
    import torch
    from omnivoice import OmniVoice

    from flowtts.core.config import resolve_model_source

    cfg = settings.omnivoice
    dtype = getattr(torch, cfg.dtype)
    model_source = resolve_model_source()
    print(f"[clone] loading {model_source} (load_asr={load_asr})…", flush=True)
    _model = OmniVoice.from_pretrained(
        model_source,
        device_map=cfg.device,
        dtype=dtype,
        load_asr=load_asr,           # only needed to auto-fill missing ref_text
        trust_remote_code=cfg.trust_remote_code,
    )
    print("[clone] model ready", flush=True)
    return _model


def _extract_fields(model, prompt) -> dict:
    """Pull (ref_audio_tokens int16, ref_text, ref_rms, sample_rate, frame_rate) out of a VoiceClonePrompt."""
    tokens = prompt.ref_audio_tokens.detach().cpu().numpy()
    sample_rate = int(getattr(model, "sampling_rate", 24000))
    frame_rate = float(getattr(getattr(model, "audio_tokenizer", None), "config", None).frame_rate) \
        if getattr(getattr(model, "audio_tokenizer", None), "config", None) is not None \
        and getattr(model.audio_tokenizer.config, "frame_rate", None) is not None else 0.0
    return {
        "ref_audio_tokens": tokens,
        "ref_text": str(prompt.ref_text),
        "ref_rms": float(prompt.ref_rms),
        "sample_rate": sample_rate,
        "frame_rate": frame_rate,
    }


def build_one(alias: str, ref_audio: str | Path, ref_text: str | None,
              load_asr: bool | None = None) -> Path:
    """Create voices_dir/<alias>.npz from a reference clip.

    load_asr defaults to True only when ref_text is missing (Whisper transcribes it).
    """
    if load_asr is None:
        load_asr = ref_text is None
    model = _get_model(load_asr)
    ref_audio = str(ref_audio)
    print(f"[clone] '{alias}' ← {ref_audio}"
          + (f"  ref_text={ref_text[:40]!r}" if ref_text else "  (auto-transcribe)"), flush=True)

    prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    fields = _extract_fields(model, prompt)

    out = Path(settings.voices.voices_dir) / f"{alias}.npz"
    save_voice_npz(out, alias=alias, **fields)

    # Verify the round-trip immediately (guards the flagged serialization caveat).
    reloaded = load_voice_npz(out)
    assert reloaded["ref_audio_tokens"].shape == fields["ref_audio_tokens"].shape, "npz round-trip shape mismatch"
    print(f"[clone] wrote {out}  tokens={reloaded['ref_audio_tokens'].shape}"
          f"  frame_rate={reloaded['frame_rate']}  rms={reloaded['ref_rms']:.4f}", flush=True)
    return out


def build_all(manifest_path: str | None) -> None:
    manifest: dict = {}
    if manifest_path and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    jobs: dict[str, tuple[str, str | None]] = {}
    if manifest:
        # A manifest means "build exactly these voices". Skip comment/non-entry keys
        # (e.g. "_comment") and any value that isn't an object with a ref_audio.
        for alias, entry in manifest.items():
            if alias.startswith("_") or not isinstance(entry, dict) or "ref_audio" not in entry:
                continue
            jobs[alias] = (entry["ref_audio"], entry.get("ref_text"))
        print(f"[clone] {len(jobs)} voice(s) from manifest {manifest_path}", flush=True)
    else:
        # No manifest → build one voice per clip in sample_files/ (auto-transcribed).
        sample_dir = Path.home() / "FlowTTS/sample_files"
        if sample_dir.is_dir():
            for f in sorted(sample_dir.iterdir()):
                if f.suffix.lower() in _AUDIO_EXTS:
                    jobs[f.stem] = (str(f), None)
        print(f"[clone] {len(jobs)} voice(s) from {sample_dir}", flush=True)

    if not jobs:
        print("[clone] nothing to build (empty manifest, no sample_files/*)", file=sys.stderr)
        return

    # Load Whisper only if at least one voice needs transcription.
    need_asr = any(rt is None for (_ra, rt) in jobs.values())
    if need_asr:
        print("[clone] some voices lack ref_text → loading Whisper ASR (one-time download).", flush=True)

    ok = 0
    for alias, (ref_audio, ref_text) in jobs.items():
        try:
            build_one(alias, ref_audio, ref_text, load_asr=need_asr)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[clone] FAILED '{alias}': {e}", file=sys.stderr, flush=True)
    print(f"[clone] built {ok}/{len(jobs)} voices → {settings.voices.voices_dir}", flush=True)


def list_voices() -> None:
    vdir = Path(settings.voices.voices_dir)
    npzs = sorted(vdir.glob("*.npz")) if vdir.is_dir() else []
    if not npzs:
        print(f"No voices in {vdir}")
        return
    print(f"Voices in {vdir}:")
    for p in npzs:
        try:
            d = load_voice_npz(p)
            print(f"  {d['alias']:<16} tokens={tuple(d['ref_audio_tokens'].shape)}"
                  f"  sr={d['sample_rate']}  ref_text={d['ref_text'][:50]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  {p.stem:<16} [unreadable: {e}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build OmniVoice voice-clone npz artifacts")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--add", metavar="ALIAS", help="Create one voice npz")
    g.add_argument("--build-all", action="store_true", help="Build from sample_files/ (+ optional manifest)")
    g.add_argument("--list", action="store_true", help="List installed voices")
    ap.add_argument("--ref-audio", help="Reference audio path (with --add)")
    ap.add_argument("--ref-text", default=None, help="Reference transcript (optional; auto-transcribed if omitted)")
    ap.add_argument("--manifest", default=None, help="manifest.json for --build-all")
    args = ap.parse_args()

    if args.list:
        list_voices()
    elif args.build_all:
        build_all(args.manifest)
    else:
        if not args.ref_audio:
            ap.error("--add requires --ref-audio")
        build_one(args.add.strip().lower(), args.ref_audio, args.ref_text)


if __name__ == "__main__":
    main()
