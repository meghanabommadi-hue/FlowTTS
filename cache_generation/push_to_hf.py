#!/usr/bin/env python3
"""
Push tara_june and simran_june audio datasets to HuggingFace Hub.

Creates a dataset with two columns:
  - text: the original sentence
  - audio: the audio file bytes (HF Audio feature — playable in dataset viewer)

Each voice is pushed as a separate config (subset) of the same repo.

Usage:
    python push_to_hf.py --repo Shubhangi7/bajaj_tts_june --token YOUR_HF_TOKEN
    python3 push_to_hf.py --repo Shubhangi7/tara_june --token "*" --tara-only
    python push_to_hf.py --repo Shubhangi7/bajaj_tts_june --token YOUR_HF_TOKEN --simran-only
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "diff_voices_and_cache"))
from text_normalize import normalize_text, split_and_expand_sentences  # noqa: E402


SENTENCE_FILES = [
    ROOT / "new_bajaj_sentences.txt",
]


def build_hash_to_text(sentence_files: list[Path]) -> dict[str, str]:
    """Build sha256_hex -> text mapping from all sentence files.

    Mirrors generate_audio_batch.py: each input line is passed through
    split_and_expand_sentences, then each sub-sentence is normalized and hashed.
    The stored text is the normalized sub-sentence (what was actually synthesized).
    """
    mapping: dict[str, str] = {}
    for path in sentence_files:
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for sub in split_and_expand_sentences(line):
                norm = normalize_text(sub)
                if not norm:
                    continue
                h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
                if h not in mapping:
                    mapping[h] = norm  # store the normalized sub-sentence as text
    print(f"Built hash→text mapping: {len(mapping):,} unique sentences")
    return mapping


def build_records(audio_dir: Path, hash_to_text: dict[str, str]) -> list[dict]:
    wav_files = sorted(audio_dir.glob("*.wav"))
    records = []
    missing_text = 0
    for wav in wav_files:
        sha = wav.stem
        text = hash_to_text.get(sha)
        if text is None:
            missing_text += 1
            continue
        records.append({"text": text, "audio_path": str(wav)})
    print(f"  {len(records):,} matched, {missing_text:,} wav files had no text mapping")
    return records


SAMPLE_RATE = 16000


def push_voice(
    records: list[dict],
    repo_id: str,
    config_name: str,
    token: str,
) -> None:
    from datasets import Dataset, Audio, Features, Value

    print(f"\nBuilding HF dataset for config '{config_name}' ({len(records):,} samples)...")

    features = Features(
        {
            "text": Value("string"),
            "audio": Audio(sampling_rate=SAMPLE_RATE),
        }
    )

    # datasets Audio feature accepts a file path string directly — it reads
    # and encodes the bytes automatically when the dataset is pushed.
    hf_records = [{"text": r["text"], "audio": r["audio_path"]} for r in records]

    ds = Dataset.from_list(hf_records, features=features)

    print(f"Pushing to {repo_id} (config={config_name}) ...")
    ds.push_to_hub(
        repo_id=repo_id,
        config_name=config_name,
        token=token,
        private=False,
    )
    print(f"Done: https://huggingface.co/datasets/{repo_id}/viewer/{config_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push TTS audio to HuggingFace Hub")
    parser.add_argument("--repo", required=True, help="HF dataset repo id, e.g. Shubhangi7/bajaj_tts_june")
    parser.add_argument("--token", required=True, help="HuggingFace write token")
    parser.add_argument("--tara-dir", default=str(ROOT / "tara_june"), help="Path to tara_june folder")
    parser.add_argument("--simran-dir", default=str(ROOT / "simran_june"), help="Path to simran_june folder")
    parser.add_argument("--tara-only", action="store_true")
    parser.add_argument("--simran-only", action="store_true")
    args = parser.parse_args()

    print("Building hash→text mapping from sentence files...")
    hash_to_text = build_hash_to_text(SENTENCE_FILES)

    voices = []
    if not args.simran_only:
        voices.append(("tara", Path(args.tara_dir)))
    if not args.tara_only:
        voices.append(("simran", Path(args.simran_dir)))

    for config_name, audio_dir in voices:
        if not audio_dir.exists():
            print(f"[skip] {audio_dir} does not exist")
            continue
        print(f"\nScanning {audio_dir} ...")
        records = build_records(audio_dir, hash_to_text)
        if not records:
            print(f"  No matched records for {config_name}, skipping.")
            continue
        push_voice(records, args.repo, config_name, args.token)

    print("\nAll done.")


if __name__ == "__main__":
    main()
