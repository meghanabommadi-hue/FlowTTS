#!/usr/bin/env python3
"""
Add or remove a voice from all places in the FlowTTS codebase.

Usage:
    python3 add_voice.py --add priya
    python3 add_voice.py --remove priya
    python3 add_voice.py --list

The script updates:
  1. flowtts/core/config.py        — VOICE_REF_AUDIO dict
  2. flowtts/server.py             — _VOICE_CACHE_MAP dict
  3. flowtts/test/test_pipeline.py — --voice argparse choices
  4. commands.md                   — voice example + available-voices line
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "sample_files"

FILES = {
    "config":   ROOT / "flowtts/core/config.py",
    "server":   ROOT / "flowtts/server.py",
    "pipeline": ROOT / "flowtts/test/test_pipeline.py",
    "commands": ROOT / "commands.md",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"  updated  {path.relative_to(ROOT)}")


# ── Per-file patch functions ──────────────────────────────────────────────────

def patch_config(text: str, voice: str, add: bool) -> str:
    """Add/remove entry in VOICE_REF_AUDIO dict."""
    entry = f'    "{voice}": f"{{_SAMPLE_FILES_DIR}}/{voice}.wav",'

    if add:
        if f'"{voice}"' in text:
            print(f"  skip     config.py — '{voice}' already present")
            return text
        # Insert before the closing line "}" of VOICE_REF_AUDIO.
        # Use a line-anchored pattern so the f-string braces inside values
        # don't terminate the match prematurely.
        text = re.sub(
            r'(VOICE_REF_AUDIO\s*:\s*dict[^\n]*\n(?:[ \t][^\n]*\n)*)(^\})',
            lambda m: m.group(1).rstrip('\n') + f'\n{entry}\n' + m.group(2),
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = re.sub(rf'\n\s*"{voice}"[^\n]+', '', text)

    return text


def patch_server(text: str, voice: str, add: bool) -> str:
    """Add/remove entry in _VOICE_CACHE_MAP dict."""
    entry = f'    "{voice}": "cached_data_{voice}",'

    if add:
        if f'"{voice}"' in text:
            print(f"  skip     server.py — '{voice}' already present")
            return text
        text = re.sub(
            r'(_VOICE_CACHE_MAP\s*:\s*dict[^\n]*\n(?:[ \t][^\n]*\n)*)(^\})',
            lambda m: m.group(1).rstrip('\n') + f'\n{entry}\n' + m.group(2),
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = re.sub(rf'\n\s*"{voice}"[^\n]+', '', text)

    return text


def patch_pipeline(text: str, voice: str, add: bool) -> str:
    """Add/remove voice from --voice argparse choices list."""
    # Match only the --voice argument line, not --mode or any other choices list.
    pattern = r'(parser\.add_argument\("--voice"[^\n]*choices=\[)([^\]]+)(\])'

    m = re.search(pattern, text)
    if not m:
        print("  warn     test_pipeline.py — --voice choices pattern not found")
        return text

    choices = [c.strip().strip('"') for c in m.group(2).split(',')]
    if add:
        if voice in choices:
            print(f"  skip     test_pipeline.py — '{voice}' already present")
            return text
        choices.append(voice)
    else:
        choices = [c for c in choices if c != voice]

    formatted = ', '.join(f'"{c}"' for c in choices)
    return text[:m.start(2)] + formatted + text[m.end(2):]


def patch_commands(text: str, voice: str, add: bool) -> str:
    """Add/remove voice example block and update available-voices line."""
    example = (
        f"\n# Use {voice} voice  (sample_files/{voice}.wav)\n"
        f"python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75 --voice {voice}"
    )

    # --- example block ---
    if add:
        if f"--voice {voice}" not in text:
            # Insert before the closing ``` of the main voice block
            text = text.replace(
                "\n# Use rani voice",
                f"{example}\n\n# Use rani voice",
            ) if "# Use rani voice" in text else text.replace(
                "\n```\n\n```bash\n# Use british_rose",
                f"{example}\n```\n\n```bash\n# Use british_rose",
            )
            # Fallback: just append before the closing ``` of the last voice block
            if f"--voice {voice}" not in text:
                last_voice_block = re.search(
                    r'(# Use \w+ voice[^\n]*\npython3[^\n]+--voice \w+)(\n```)',
                    text,
                )
                if last_voice_block:
                    text = text[:last_voice_block.end(1)] + f"\n{example}" + text[last_voice_block.end(1):]
        else:
            print(f"  skip     commands.md example — '{voice}' already present")
    else:
        text = re.sub(rf'\n# Use {voice} voice[^\n]*\npython3[^\n]+--voice {voice}', '', text)

    # --- available-voices line ---
    def update_voices_line(m: re.Match) -> str:
        line = m.group(0)
        voices = re.findall(r'`(\w+)`', line)
        if add:
            if voice not in voices:
                voices.append(voice)
        else:
            voices = [v for v in voices if v != voice]
        return "- Available voices: " + ", ".join(f"`{v}`" for v in voices)

    text = re.sub(r'- Available voices:.*', update_voices_line, text)

    return text


# ── Main ─────────────────────────────────────────────────────────────────────

def list_voices() -> None:
    text = read(FILES["config"])
    voices = re.findall(r'"(\w+)":\s*f"\{_SAMPLE_FILES_DIR\}', text)
    print("Registered voices:")
    for v in voices:
        wav = SAMPLE_DIR / f"{v}.wav"
        status = "OK" if wav.exists() else "MISSING WAV"
        print(f"  {v:<16} {status}")

    print("\nWAV files in sample_files/ not yet registered:")
    registered = set(voices)
    for wav in sorted(SAMPLE_DIR.glob("*.wav")):
        if wav.stem not in registered:
            print(f"  {wav.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage FlowTTS voices")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--add",    metavar="VOICE", help="Add a voice")
    group.add_argument("--remove", metavar="VOICE", help="Remove a voice")
    group.add_argument("--list",   action="store_true", help="List registered voices")
    args = parser.parse_args()

    if args.list:
        list_voices()
        return

    voice = (args.add or args.remove).lower().strip()
    add = args.add is not None

    if add:
        wav = SAMPLE_DIR / f"{voice}.wav"
        if not wav.exists():
            print(f"ERROR: {wav} not found — add the WAV file first.", file=sys.stderr)
            sys.exit(1)

    print(f"{'Adding' if add else 'Removing'} voice: {voice}\n")

    patches = [
        ("config",   patch_config),
        ("server",   patch_server),
        ("pipeline", patch_pipeline),
        ("commands", patch_commands),
    ]

    for key, fn in patches:
        path = FILES[key]
        original = read(path)
        updated = fn(original, voice, add)
        if updated != original:
            write(path, updated)
        else:
            print(f"  no-op    {path.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
