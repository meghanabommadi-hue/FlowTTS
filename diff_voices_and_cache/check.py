import os
import hashlib
from tqdm import tqdm

text_file = "/home/ubuntu/FlowTTS/cache_dataset/bajaj_sentences_unique.txt"
audio_dir = "/home/ubuntu/FlowTTS/cache_dataset/cached_audio_files"   # folder containing wav files
audio_ext = ".wav"

missing_audio = []
extra_audio = []
duplicate_hash = set()
hashes_seen = set()

# read text
with open(text_file, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

print("Total text lines:", len(sentences))

# check audio existence
for sentence in tqdm(sentences):
    h = hashlib.sha256(sentence.encode("utf-8")).hexdigest()

    if h in hashes_seen:
        duplicate_hash.add(h)

    hashes_seen.add(h)

    audio_path = os.path.join(audio_dir, h + audio_ext)

    if not os.path.exists(audio_path):
        missing_audio.append((sentence, h))

# check extra audio files
audio_files = set(f.replace(audio_ext, "") for f in os.listdir(audio_dir) if f.endswith(audio_ext))

extra_audio = audio_files - hashes_seen

print("\n----- SUMMARY -----")
print("Text lines:", len(sentences))
print("Expected audio files:", len(hashes_seen))
print("Actual audio files:", len(audio_files))
print("Missing audio:", len(missing_audio))
print("Extra audio:", len(extra_audio))
print("Duplicate sentence hashes:", len(duplicate_hash))

# save reports
with open("missing_audio.txt", "w", encoding="utf-8") as f:
    for sentence, h in missing_audio:
        f.write(f"{h}\t{sentence}\n")

with open("extra_audio.txt", "w") as f:
    for h in extra_audio:
        f.write(h + "\n")