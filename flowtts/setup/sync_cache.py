#!/usr/bin/env python3
"""
Sync WAV cache from HuggingFace to local cache directories.
Downloads only files not already present. Runs both voices in parallel.

Repos:
  Shubhangi7/tara_cache_full   -> ~/FlowTTS/cached_data_tara_cache_full
  Shubhangi7/simran_cache_full -> ~/FlowTTS/cached_data_simran

Used by cron: 3rd-7th of every month at 2am IST (8:30pm UTC prev day).
"""

import hashlib
import io
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Thread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path.home() / "FlowTTS/sync_cache.log"),
    ],
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

VOICES = {
    "tara": {
        "repo":      "Shubhangi7/tara_cache_full_old",
        "cache_dir": Path.home() / "FlowTTS/cached_data_tara_cache_full",
    },
    "simran": {
        "repo":      "Shubhangi7/simran_cache_full",
        "cache_dir": Path.home() / "FlowTTS/cached_data_simran_june",
    },
}


def _hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=HF_TOKEN)


def _retry_list(api, repo):
    from huggingface_hub import HfApi
    for attempt in range(1, 6):
        try:
            return list(api.list_repo_files(repo, repo_type="dataset"))
        except Exception as e:
            if "404" in str(e):
                raise
            if attempt < 5:
                wait = 60 * attempt
                logging.getLogger(repo).warning(f"Rate limited listing files, retrying in {wait}s (attempt {attempt}/5)")
                time.sleep(wait)
            else:
                raise


def sync_wav(name, repo, cache_dir):
    log = logging.getLogger(name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Listing WAV files in {repo} ...")
    try:
        api = _hf_api()
        all_files = [f for f in _retry_list(api, repo) if f.endswith(".wav")]
    except Exception as e:
        if "404" in str(e):
            log.warning(f"Repo {repo} not found — skipping (will retry next run)")
            return
        raise

    existing = {f.name for f in cache_dir.glob("*.wav")}
    missing = [f for f in all_files if Path(f).name not in existing]
    log.info(f"{len(existing)} already present, {len(missing)} to download out of {len(all_files)} total")

    if not missing:
        log.info("Nothing to download.")
        return

    from huggingface_hub import hf_hub_download

    def fetch(filename):
        for attempt in range(1, 6):
            try:
                hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=str(cache_dir),
                    token=HF_TOKEN,
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
                log.error(f"Failed {filename}: {err}")
            elif done % 200 == 0 or done == len(missing):
                log.info(f"{done}/{len(missing)} downloaded ({errors} errors)")

    log.info(f"Done — {len(list(cache_dir.glob('*.wav')))} total WAVs in {cache_dir}")


def sync_parquet(name, repo, cache_dir):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    log = logging.getLogger(name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Listing parquet files in {repo} ...")
    try:
        api = _hf_api()
        parquet_files = [f for f in _retry_list(api, repo) if f.endswith(".parquet")]
    except Exception as e:
        if "404" in str(e):
            log.warning(f"Repo {repo} not found — skipping (will retry next run)")
            return
        raise

    if not parquet_files:
        log.warning(f"No parquet files found in {repo}")
        return

    log.info(f"{len(parquet_files)} parquet shard(s) found")
    existing = {f.name for f in cache_dir.glob("*.wav")}
    total_written = 0

    for shard_path in parquet_files:
        log.info(f"Processing shard: {shard_path}")
        for attempt in range(1, 6):
            try:
                local = hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=shard_path,
                    token=HF_TOKEN,
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < 5:
                    wait = 60 * attempt
                    log.warning(f"Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        df = pd.read_parquet(local)
        written = skipped = 0

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

            (cache_dir / orig_name).write_bytes(wav_bytes)
            existing.add(orig_name)
            written += 1

        total_written += written
        log.info(f"Shard done — {written} written, {skipped} skipped")

    log.info(f"Done — {total_written} new WAVs written, {len(list(cache_dir.glob('*.wav')))} total in {cache_dir}")


def detect_layout_and_sync(name, repo, cache_dir):
    """Detect repo layout (wav files vs parquet) and run the right sync."""
    log = logging.getLogger(name)
    try:
        api = _hf_api()
        files = _retry_list(api, repo)
    except Exception as e:
        if "404" in str(e):
            log.warning(f"Repo {repo} not found — skipping")
            return
        raise

    has_parquet = any(f.endswith(".parquet") for f in files)
    has_wav = any(f.endswith(".wav") for f in files)

    if has_parquet:
        log.info(f"Detected parquet layout")
        sync_parquet(name, repo, cache_dir)
    elif has_wav:
        log.info(f"Detected WAV layout")
        sync_wav(name, repo, cache_dir)
    else:
        log.warning(f"No .wav or .parquet files found in {repo} — nothing to sync")


def sync_voice(name, cfg):
    detect_layout_and_sync(name, cfg["repo"], cfg["cache_dir"])


def main():
    log = logging.getLogger("sync_cache")
    log.info("=== Cache sync started ===")

    threads = [
        Thread(target=sync_voice, args=(name, cfg), name=name, daemon=False)
        for name, cfg in VOICES.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    log.info("=== Cache sync complete ===")


if __name__ == "__main__":
    main()
