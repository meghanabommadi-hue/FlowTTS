import hashlib
import os
import sys
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset, Features, Value
from huggingface_hub import HfApi
from tqdm import tqdm

TEXT_FILE = "bajaj_3_lakh_sentences.txt"
AUDIO_DIR = "simran_3_lakh"
OUTPUT_DIR = "parquet_output"
SHARD_SIZE = 5000
AUDIO_EXT = ".wav"

HF_REPO_ID = "Shubhangi7/simran_cache_full"  # e.g. "kapture/bajaj-tara-tts"
HF_TOKEN = os.environ.get("HF_TOKEN")           # set via: export HF_TOKEN=hf_...


def sha256_filename(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() + AUDIO_EXT


def main():
    print("Loading texts...")
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"Total texts loaded: {len(texts)}")

    audio_dir = Path(AUDIO_DIR)

    print("\nRunning consistency check...")
    valid_pairs = []
    missing = []

    for text in tqdm(texts, desc="Checking text-audio pairs"):
        audio_path = audio_dir / sha256_filename(text)
        if audio_path.exists():
            valid_pairs.append((text, audio_path))
        else:
            missing.append(text)

    print(f"\nValid pairs  : {len(valid_pairs)}")
    print(f"Missing audio: {len(missing)}")
    if missing:
        print(f"Sample missing texts ({min(5, len(missing))} shown):")
        for t in missing[:5]:
            print(f"  {t[:120]}")

    if not valid_pairs:
        print("No valid pairs found. Exiting.")
        sys.exit(1)

    sampling_rate = sf.info(str(valid_pairs[0][1])).samplerate
    print(f"\nDetected sampling rate: {sampling_rate} Hz")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_shards = (len(valid_pairs) + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"Creating {num_shards} parquet shard(s) → '{OUTPUT_DIR}/'")

    features = Features({
        "text": Value("string"),
        "audio": Audio(sampling_rate=sampling_rate),
    })

    for shard_idx in tqdm(range(num_shards), desc="Writing shards"):
        shard = valid_pairs[shard_idx * SHARD_SIZE : (shard_idx + 1) * SHARD_SIZE]

        texts_shard, audio_shard = [], []
        for text, audio_path in tqdm(shard, desc=f"Shard {shard_idx + 1}/{num_shards}", leave=False):
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            texts_shard.append(text)
            audio_shard.append({"bytes": audio_bytes, "path": audio_path.name})

        ds = Dataset.from_dict(
            {"text": texts_shard, "audio": audio_shard},
            features=features,
        )
        out_file = os.path.join(OUTPUT_DIR, f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet")
        ds.to_parquet(out_file)

    print(f"\nDone. {num_shards} parquet file(s) saved to '{OUTPUT_DIR}/'.")

    # Upload to HuggingFace
    print(f"\nUploading to HuggingFace: {HF_REPO_ID}")
    api = HfApi(token=HF_TOKEN)

    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )

    parquet_files = sorted(Path(OUTPUT_DIR).glob("*.parquet"))
    for parquet_file in tqdm(parquet_files, desc="Uploading parquet shards"):
        api.upload_file(
            path_or_fileobj=str(parquet_file),
            path_in_repo=f"data/{parquet_file.name}",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )

    print(f"\nDataset live at: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()

