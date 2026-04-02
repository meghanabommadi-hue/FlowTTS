#!/usr/bin/env python3
"""
Download WAV cache from HuggingFace for a given voice and store it in
the FlowTTS cache directory (~/FlowTTS/cached_data_<voice>).

Usage:
    python download_cache.py --voice tara
    python download_cache.py --voice simran
    python download_cache.py --voice tara --voice simran   # both
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HF_REPOS = {
    "simran": "Shubhangi7/simran",
    "tara":   "Shubhangi7/tara",
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

def download(voice: str, token: str | None):
    repo = HF_REPOS.get(voice)
    if not repo:
        print(f"[ERROR] Unknown voice '{voice}'. Choose from: {list(HF_REPOS)}")
        return

    cache_dir = Path.home() / f"FlowTTS/cached_data_{voice}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # List all WAV files in the repo — single API call, retry on 429
    print(f"[{voice}] Listing files in {repo} ...")
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
                print(f"[{voice}] Rate limited listing files. Retrying in {wait}s... (attempt {attempt}/5)")
                time.sleep(wait)
            else:
                raise
    print(f"[{voice}] {len(all_files)} WAVs in repo")

    # Only download files not already present
    existing = {f.name for f in cache_dir.glob("*.wav")}
    missing = [f for f in all_files if Path(f).name not in existing]
    print(f"[{voice}] {len(existing)} already downloaded, {len(missing)} to go")

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

    done = 0
    errors = 0
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

    wavs = list(cache_dir.glob("*.wav"))
    print(f"[{voice}] Done — {len(wavs)} WAVs in {cache_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", action="append", required=True,
                        choices=list(HF_REPOS), metavar="VOICE",
                        help="Voice to download: simran | tara (repeat for multiple)")
    args = parser.parse_args()

    token = resolve_token()
    for voice in args.voice:
        download(voice, token)

if __name__ == "__main__":
    main()
