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
from pathlib import Path
from huggingface_hub import snapshot_download

HF_REPOS = {
    "simran": "Shubhangi7/simran",
    "tara":   "Shubhangi7/tara",
}

def resolve_token():
    import os
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

    print(f"[{voice}] Downloading {repo} → {cache_dir} ...")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(cache_dir),
        token=token,
        ignore_patterns=["*.txt", "*.json", ".gitattributes"],
    )
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
