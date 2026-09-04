"""Chaashini (चाशनी, "sugar syrup"): a strict, high-throughput builder of pristine
Indian-language speech data.

Package layout
--------------
config      YAML + environment driven settings (pydantic)
db          SQLite (WAL) state store: videos, chunks, GPU jobs, shards, events, metrics
languages   Registry of target languages, scripts, function words, discovery genres
lid         Regex / Unicode-script language identification with code-mix composition
llm         OpenAI-compatible client used to invent discovery search queries
ytsource    Discovery + download of publicly available long-form audio (yt-dlp)
audio       ffmpeg decode/cut/encode helpers
vad         TenVAD frame-level speech activity -> speech regions
segment     VAD x diarization fusion -> single-speaker chunks (0.5 s .. 30 s)
quality     DNSMOS, music/speech tagging, SNR, loudness, clipping, bandwidth
gpujobs     Pull-style job queue consumed by the GPU box workers
export      Final chunk rendering (24 kHz FLAC) + metadata staging
packer      Parquet shard builder (HF `datasets` compatible)
pusher      Hugging Face upload with retry + dataset card refresh
api         FastAPI: dashboard API, internal GPU job API, static UI
workers/    Long-running worker loops (discover, download, process, publish)
supervisor  Spawns/monitors workers, janitor for stale leases
"""
__version__ = "0.1.0"
