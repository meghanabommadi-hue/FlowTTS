#!/usr/bin/env python3
"""Offline voice-clone builder for Fish Audio S2 Pro.

Creates reusable voice references that the running gateway loads at startup and
addresses by alias. A voice is a reference clip converted to mono WAV plus a JSON
manifest (ref_text + optional language). Unlike the old OmniVoice flow, this needs
NO GPU and NO model — the sglang backend encodes the reference on first use.

ref_text is MANDATORY — there is no ASR/auto-transcription. Provide the transcript
of each reference clip (via --ref-text or the manifest). Voices without a ref_text
are skipped.

Usage:
    # One voice
    python -m flowtts.voices.clone --add priya --ref-audio sample_files/priya.wav \
        --ref-text "नमस्ते, मैं प्रिया बोल रही हूँ।" --lang hi

    # Build every voice defined in a manifest (each entry needs ref_audio + ref_text)
    python -m flowtts.voices.clone --build-all --manifest voices/manifest.json

    # List installed voices
    python -m flowtts.voices.clone --list

manifest.json format (ref_text required for each voice):
    { "priya": {"ref_audio": "sample_files/priya.wav", "ref_text": "...", "language": "hi"} }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flowtts.core.config import settings
from flowtts.voices.store import load_voice, manifest_path, save_voice

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def _to_mono_wav(src: str, dst: str) -> tuple[int, float]:
    """Convert any supported clip (via ffmpeg) to a mono 16-bit WAV. Returns (sr, duration_s)."""
    from pydub import AudioSegment

    seg = AudioSegment.from_file(src).set_channels(1).set_sample_width(2)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    seg.export(dst, format="wav")
    return int(seg.frame_rate), float(len(seg) / 1000.0)


def build_one(alias: str, ref_audio: str | Path, ref_text: str | None,
              language: str | None = None) -> Path:
    """Create voices_dir/<alias>.wav + <alias>.json from a reference clip. ref_text required."""
    if not ref_text or not str(ref_text).strip():
        raise ValueError(
            f"ref_text is required to clone voice '{alias}' "
            f"(pass --ref-text, or add \"ref_text\" to the manifest entry)."
        )
    voices_dir = Path(settings.voices.voices_dir)
    out_wav = voices_dir / f"{alias}.wav"
    print(f"[clone] '{alias}' ← {ref_audio}  ref_text={ref_text[:40]!r}  lang={language or '-'}", flush=True)

    sr, dur = _to_mono_wav(str(ref_audio), str(out_wav))
    save_voice(voices_dir, alias=alias, ref_text=str(ref_text),
               audio_file=out_wav.name, language=language)

    # Verify the manifest round-trips immediately.
    reloaded = load_voice(manifest_path(voices_dir, alias))
    assert reloaded["audio_file"] == out_wav.name, "manifest round-trip mismatch"
    print(f"[clone] wrote {out_wav}  sr={sr}  dur={dur:.2f}s  + {out_wav.stem}.json", flush=True)
    return out_wav


def build_all(manifest_path_arg: str | None) -> None:
    manifest: dict = {}
    if manifest_path_arg and Path(manifest_path_arg).is_file():
        manifest = json.loads(Path(manifest_path_arg).read_text(encoding="utf-8"))

    jobs: dict[str, tuple[str, str | None, str | None]] = {}
    if manifest:
        # A manifest means "build exactly these voices". Skip comment/non-entry keys.
        for alias, entry in manifest.items():
            if alias.startswith("_") or not isinstance(entry, dict) or "ref_audio" not in entry:
                continue
            jobs[alias] = (entry["ref_audio"], entry.get("ref_text"), entry.get("language"))
        print(f"[clone] {len(jobs)} voice(s) from manifest {manifest_path_arg}", flush=True)
    else:
        # No manifest → list sample_files/ clips (skipped below unless a manifest gives ref_text).
        sample_dir = Path.home() / "FlowTTS/sample_files"
        if sample_dir.is_dir():
            for f in sorted(sample_dir.iterdir()):
                if f.suffix.lower() in _AUDIO_EXTS:
                    jobs[f.stem] = (str(f), None, None)
        print(f"[clone] {len(jobs)} voice(s) from {sample_dir}", flush=True)

    # ref_text is mandatory — skip (don't fail) voices that don't provide it.
    missing = [a for a, (_ra, rt, _lg) in jobs.items() if not rt]
    if missing:
        print(f"[clone] skipping (no ref_text — required for cloning): {missing}", flush=True)
        jobs = {a: v for a, v in jobs.items() if v[1]}

    if not jobs:
        print("[clone] nothing to build — add a \"ref_text\" to each manifest entry to clone.", flush=True)
        return

    ok = 0
    for alias, (ref_audio, ref_text, language) in jobs.items():
        try:
            build_one(alias, ref_audio, ref_text, language)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[clone] FAILED '{alias}': {e}", file=sys.stderr, flush=True)
    print(f"[clone] built {ok}/{len(jobs)} voices → {settings.voices.voices_dir}", flush=True)


def list_voices() -> None:
    vdir = Path(settings.voices.voices_dir)
    manifests = sorted(vdir.glob("*.json")) if vdir.is_dir() else []
    if not manifests:
        print(f"No voices in {vdir}")
        return
    print(f"Voices in {vdir}:")
    for p in manifests:
        try:
            d = load_voice(p)
            print(f"  {d['alias']:<16} audio={d['audio_file']:<24} lang={d['language'] or '-':<4}"
                  f"  ref_text={d['ref_text'][:50]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  {p.stem:<16} [unreadable: {e}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Fish S2 Pro voice references (clip + manifest)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--add", metavar="ALIAS", help="Create one voice")
    g.add_argument("--build-all", action="store_true", help="Build from sample_files/ (+ optional manifest)")
    g.add_argument("--list", action="store_true", help="List installed voices")
    ap.add_argument("--ref-audio", help="Reference audio path (with --add)")
    ap.add_argument("--ref-text", default=None, help="Reference transcript (REQUIRED — no ASR/auto-transcribe)")
    ap.add_argument("--lang", "--language", dest="language", default=None, help="Preferred synthesis language")
    ap.add_argument("--manifest", default=None, help="manifest.json for --build-all")
    args = ap.parse_args()

    if args.list:
        list_voices()
    elif args.build_all:
        build_all(args.manifest)
    else:
        if not args.ref_audio:
            ap.error("--add requires --ref-audio")
        if not args.ref_text:
            ap.error("--add requires --ref-text (ref_text is mandatory for voice cloning)")
        build_one(args.add.strip().lower(), args.ref_audio, args.ref_text, args.language)


if __name__ == "__main__":
    main()
