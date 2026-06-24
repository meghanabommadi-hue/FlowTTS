#!/usr/bin/env python3
"""
Push generated WAV files to HuggingFace in the correct parquet format.

Matches the format of Shubhangi7/simran_cache_full:
  - datasets.Audio with bytes embedded in parquet
  - Shards: data/train-00000-of-NNNNN.parquet
  - Schema: {text: string, audio: Audio(sampling_rate=16000)}

Resume-safe: tracks last pushed shard in <voice>_push_progress.json.
Reads sentence->sha256 mapping from the parquet to populate text column.

Usage:
    python push_wavs_to_hf.py --voice simran --token hf_xxx
    python push_wavs_to_hf.py --voice tara   --token hf_xxx --shard-size 500
    python push_wavs_to_hf.py --voice simran --token hf_xxx --delete-and-restart
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

HF_REPOS = {
    "simran": "Shubhangi7/simran_cache_full",
    "tara":   "Shubhangi7/tara_cache_full",
}

DEFAULT_TOKEN    = os.environ.get("HF_TOKEN", "<your_hf_token>")
DEFAULT_PARQUET  = str(HERE / "normalized_sentences.parquet")
PUSH_PROGRESS_TMPL = "{voice}_push_progress.json"
SHARD_SIZE       = 500
SAMPLE_RATE      = 16000


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_text_map(parquet_path: str, min_freq: int) -> dict[str, str]:
    """sha256_hex -> normalized text, from the sentences parquet."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    filtered = df[df["frequency"] > min_freq]
    result = {sha256_hex(t): t for t in filtered["text"].tolist()}
    print(f"[INFO] Text map: {len(result):,} entries")
    return result


def collect_records(audio_dir: Path, text_map: dict[str, str]) -> list[dict]:
    """Return list of {text, wav_path} for every top-level WAV with a known text."""
    wavs = sorted(audio_dir.glob("*.wav"))  # maxdepth=1 via glob on dir directly
    records = []
    missing_text = 0
    for wav in wavs:
        text = text_map.get(wav.stem)
        if text is None:
            missing_text += 1
            continue
        records.append({"text": text, "wav_path": wav})
    print(f"[INFO] {len(records):,} matched WAVs  ({missing_text:,} had no text mapping)")
    return records


def load_push_progress(path: Path) -> dict:
    if path.exists():
        try:
            d = json.loads(path.read_text())
            print(f"[RESUME] last_shard={d.get('last_shard', 0)}  total_pushed={d.get('total_pushed', 0)}")
            return d
        except Exception:
            pass
    return {"last_shard": 0, "total_pushed": 0, "num_shards": 0}


def save_push_progress(path: Path, last_shard: int, total_pushed: int, num_shards: int) -> None:
    path.write_text(json.dumps({
        "last_shard":   last_shard,
        "total_pushed": total_pushed,
        "num_shards":   num_shards,
        "updated":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        # kept for watch_progress.sh compatibility
        "last_pushed_batch": last_shard,
    }, indent=2))


def push_shard(
    records: list[dict],
    shard_idx: int,
    num_shards: int,
    repo_id: str,
    token: str,
) -> None:
    from datasets import Audio, Dataset, Features, Value

    fname = f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet"
    print(f"  [shard {shard_idx}/{num_shards}] building {len(records)} samples → {fname}", flush=True)

    features = Features({
        "text":  Value("string"),
        "audio": Audio(sampling_rate=SAMPLE_RATE),
    })

    texts, audios = [], []
    for r in records:
        wav_path = Path(r["wav_path"])
        audio_bytes = wav_path.read_bytes()
        texts.append(r["text"])
        audios.append({"bytes": audio_bytes, "path": wav_path.name})

    ds = Dataset.from_dict({"text": texts, "audio": audios}, features=features)

    # Write to a local temp parquet then upload
    tmp = HERE / f"_tmp_shard_{shard_idx:05d}.parquet"
    try:
        ds.to_parquet(str(tmp))
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo=f"data/{fname}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  [HF] pushed {fname} ({len(records)} samples) → {repo_id}", flush=True)
    finally:
        tmp.unlink(missing_ok=True)


def delete_repo_contents(repo_id: str, token: str) -> None:
    """Delete all files in the repo data/ folder to start fresh."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(repo_id, repo_type="dataset")
             if f.startswith("data/")]
    if not files:
        print(f"[INFO] No data/ files to delete in {repo_id}")
        return
    print(f"[INFO] Deleting {len(files)} files from {repo_id}...")
    from huggingface_hub import CommitOperationDelete
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo=f) for f in files],
        commit_message="Reset: clear old batched WAV uploads",
    )
    print(f"[INFO] Deleted {len(files)} files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push WAVs to HF in correct parquet format")
    parser.add_argument("--voice",      choices=["simran", "tara"], required=True)
    parser.add_argument("--token",      default=DEFAULT_TOKEN)
    parser.add_argument("--repo",       default=None, help="Override HF repo id")
    parser.add_argument("--audio-dir",  default=None)
    parser.add_argument("--parquet",    default=DEFAULT_PARQUET)
    parser.add_argument("--min-freq",   type=int, default=5)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--delete-and-restart", action="store_true",
                        help="Delete existing data/ in the HF repo and push from scratch")
    args = parser.parse_args()

    if not args.token:
        print("[ERROR] No HF token. Set HF_TOKEN or pass --token.", file=sys.stderr)
        sys.exit(1)

    repo_id   = args.repo or HF_REPOS[args.voice]
    audio_dir = Path(args.audio_dir) if args.audio_dir else HERE / f"{args.voice}_counseling_audio"
    progress_path = HERE / PUSH_PROGRESS_TMPL.format(voice=args.voice)

    print(f"[INFO] Voice      : {args.voice}")
    print(f"[INFO] Audio dir  : {audio_dir.resolve()}")
    print(f"[INFO] HF repo    : {repo_id}")
    print(f"[INFO] Shard size : {args.shard_size}")

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

    if args.delete_and_restart:
        delete_repo_contents(repo_id, args.token)
        progress_path.unlink(missing_ok=True)
        print("[INFO] Reset complete. Starting fresh.")

    text_map = build_text_map(args.parquet, args.min_freq)
    records  = collect_records(audio_dir, text_map)

    if not records:
        print("[ERROR] No WAV files found. Run generate_wavs.py first.")
        sys.exit(1)

    shards = [records[i:i + args.shard_size]
              for i in range(0, len(records), args.shard_size)]
    num_shards = len(shards)

    progress    = load_push_progress(progress_path)
    last_shard  = progress["last_shard"]
    total_pushed = progress["total_pushed"]

    print(f"[INFO] {len(records):,} records → {num_shards} shards  (resuming from shard {last_shard + 1})\n")

    t_start = time.perf_counter()

    for shard_idx, shard in enumerate(shards, 1):
        if shard_idx <= last_shard:
            print(f"  [shard {shard_idx}/{num_shards}] already pushed — skipping")
            continue
        try:
            push_shard(shard, shard_idx, num_shards, repo_id, args.token)
            total_pushed += len(shard)
            last_shard = shard_idx
            save_push_progress(progress_path, last_shard, total_pushed, num_shards)

            elapsed = time.perf_counter() - t_start
            rate = total_pushed / elapsed if elapsed > 0 else 0
            remaining = sum(len(s) for s in shards[shard_idx:])
            eta_s = int(remaining / rate) if rate > 0 else 0
            print(f"  total_pushed={total_pushed:,}  rate={rate:.1f}/s  "
                  f"ETA={eta_s//3600}h{(eta_s%3600)//60}m\n", flush=True)
        except Exception as exc:
            print(f"[ERROR shard {shard_idx}] {exc}", flush=True)
            print(f"Progress saved to shard {last_shard}. Re-run to resume.", flush=True)
            sys.exit(1)

    elapsed = time.perf_counter() - t_start
    print(f"\n[DONE] total_pushed={total_pushed:,}  time={elapsed:.1f}s")
    print(f"[DONE] https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
