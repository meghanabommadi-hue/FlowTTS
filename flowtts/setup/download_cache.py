#!/usr/bin/env python3
"""
Download WAV cache from HuggingFace for a given voice and store it in
the FlowTTS cache directory (~/FlowTTS/cached_data_<voice>).

Supports two repo layouts:
  - wav:     repo contains .wav files directly (simran, tara)
  - parquet: repo contains parquet files with audio={'bytes':...,'path':...} column (simran_june)

Usage:
    python download_cache.py --voice tara
    python download_cache.py --voice simran
    python download_cache.py --voice simran_june
    python download_cache.py --voice simran --voice tara   # multiple
"""

import argparse
import hashlib
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HF_REPOS = {
    "simran":           ("Shubhangi7/simran",           "wav"),
    "tara":             ("Shubhangi7/tara",             "wav"),
    "simran_june":      ("Shubhangi7/simran_june",      "parquet"),
    "tara_cache_full":  ("Shubhangi7/tara_cache_full",  "parquet"),
}

def resolve_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        for p in [Path.home() / ".cache/huggingface/token",
                  Path.home() / ".huggingface/token"]:
            if p.exists():
                token = p.read_text().strip()
                break
    return token or None


# ---------------------------------------------------------------------------
# WAV-based repos
# ---------------------------------------------------------------------------

def download_wav(voice: str, repo: str, cache_dir: Path, token: str | None):
    print(f"[{voice}] Listing WAV files in {repo} ...", flush=True)
    api = HfApi(token=token)
    for attempt in range(1, 6):
        try:
            all_files = [
                f for f in api.list_repo_files(repo, repo_type="dataset")
                if f.endswith(".wav")
            ]
            break
        except Exception as e:
            if "429" in str(e) and attempt < 5:
                wait = 60 * attempt
                print(f"[{voice}] Rate limited. Retrying in {wait}s... (attempt {attempt}/5)", flush=True)
                time.sleep(wait)
            else:
                raise

    print(f"[{voice}] {len(all_files)} WAVs in repo", flush=True)
    existing = {f.name for f in cache_dir.glob("*.wav")}
    missing = [f for f in all_files if Path(f).name not in existing]
    print(f"[{voice}] {len(existing)} already downloaded, {len(missing)} to go", flush=True)

    if not missing:
        print(f"[{voice}] Already complete.")
        return

    def fetch(filename):
        for attempt in range(1, 6):
            try:
                hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=str(cache_dir),
                    token=token,
                )
                return filename, None
            except Exception as e:
                if "429" in str(e) and attempt < 5:
                    time.sleep(30 * attempt)
                else:
                    return filename, str(e)

    done = errors = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(fetch, f): f for f in missing}
        for fut in as_completed(futures):
            filename, err = fut.result()
            done += 1
            if err:
                errors += 1
                print(f"[{voice}] ERROR {filename}: {err}")
            elif done % 100 == 0 or done == len(missing):
                print(f"[{voice}] {done}/{len(missing)} downloaded ({errors} errors)")

    print(f"[{voice}] Done — {len(list(cache_dir.glob('*.wav')))} WAVs in {cache_dir}")


# ---------------------------------------------------------------------------
# Parquet-based repos (audio column contains {'bytes': ..., 'path': ...})
# ---------------------------------------------------------------------------

def download_parquet(voice: str, repo: str, cache_dir: Path, token: str | None):
    import pandas as pd

    print(f"[{voice}] Listing parquet files in {repo} ...", flush=True)
    api = HfApi(token=token)
    parquet_files = [
        f for f in api.list_repo_files(repo, repo_type="dataset")
        if f.endswith(".parquet")
    ]
    print(f"[{voice}] {len(parquet_files)} parquet shard(s) found", flush=True)

    existing = {f.name for f in cache_dir.glob("*.wav")}
    total_written = 0
    total_skipped = len(existing)

    # Download all shards in parallel first
    def fetch_shard(shard_path):
        for attempt in range(1, 6):
            try:
                return hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=shard_path,
                    token=token,
                )
            except Exception as e:
                if "429" in str(e) and attempt < 5:
                    wait = 60 * attempt
                    print(f"[{voice}] Rate limited on {shard_path}. Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    raise

    print(f"[{voice}] Downloading {len(parquet_files)} shards in parallel...", flush=True)
    local_shards = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_shard, p): p for p in parquet_files}
        for fut in as_completed(futures):
            shard_path = futures[fut]
            local_shards.append((shard_path, fut.result()))
            print(f"[{voice}] Shard downloaded: {shard_path} ({len(local_shards)}/{len(parquet_files)})", flush=True)

    # Extract WAVs from each shard in parallel
    def extract_row(args):
        orig_name, wav_bytes, out_path = args
        out_path.write_bytes(wav_bytes)
        return orig_name

    for shard_path, local in local_shards:
        print(f"[{voice}] Extracting WAVs from {shard_path} ...", flush=True)
        df = pd.read_parquet(local)

        rows_to_write = []
        skipped = 0
        for _, row in df.iterrows():
            audio = row["audio"]
            wav_bytes = audio["bytes"] if isinstance(audio, dict) else audio

            orig_name = None
            if isinstance(audio, dict) and audio.get("path"):
                orig_name = Path(audio["path"]).name
                if not orig_name.endswith(".wav"):
                    orig_name += ".wav"
            if orig_name is None:
                orig_name = hashlib.sha256(wav_bytes).hexdigest() + ".wav"

            if orig_name in existing:
                skipped += 1
                continue

            rows_to_write.append((orig_name, wav_bytes, cache_dir / orig_name))
            existing.add(orig_name)

        written = 0
        with ThreadPoolExecutor(max_workers=32) as ex:
            for orig_name in ex.map(extract_row, rows_to_write):
                written += 1
                if written % 1000 == 0 or written == len(rows_to_write):
                    print(f"[{voice}]   {shard_path}: {written}/{len(rows_to_write)} written", flush=True)

        total_written += written
        total_skipped += skipped
        print(f"[{voice}] Shard done — {written} written, {skipped} skipped", flush=True)

    print(f"[{voice}] Done — {total_written} new WAVs written, {total_skipped} already existed", flush=True)
    print(f"[{voice}] Cache dir: {cache_dir} ({len(list(cache_dir.glob('*.wav')))} total WAVs)", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def download(voice: str, token: str | None):
    if voice not in HF_REPOS:
        print(f"[ERROR] Unknown voice '{voice}'. Choose from: {list(HF_REPOS)}")
        return

    repo, layout = HF_REPOS[voice]
    cache_dir = Path.home() / f"FlowTTS/cached_data_{voice}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if layout == "wav":
        download_wav(voice, repo, cache_dir, token)
    elif layout == "parquet":
        download_parquet(voice, repo, cache_dir, token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voice", action="append", required=True,
        choices=list(HF_REPOS), metavar="VOICE",
        help=f"Voice to download: {' | '.join(HF_REPOS)} (repeat for multiple)",
    )
    args = parser.parse_args()

    token = resolve_token()
    for voice in args.voice:
        download(voice, token)


if __name__ == "__main__":
    main()
