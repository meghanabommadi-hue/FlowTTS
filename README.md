# Chaashini (चाशनी) — pristine Indian-language speech corpus builder

A two-box, always-on pipeline that discovers publicly available long-form spoken-word audio in
Indian languages, strips everything that is not clean single-speaker speech, transcribes and
language-tags what survives, and pushes the result to the Hugging Face Hub every 10 hours of
newly accepted audio — with a live dashboard showing the whole machine.

* Dataset: [`kapturecx/Chaashini`](https://huggingface.co/datasets/kapturecx/Chaashini)
* Dashboard: `http://101.53.139.186/chaashini/`
* Docs: [architecture + diagrams](docs/ARCHITECTURE.md) · [operations runbook](docs/OPERATIONS.md)

## What it does

```
discover (LLM queries, channel crawl) → download original audio track → decode + VAD + source gate
→ diarize (GPU) → single-speaker chunks 0.5–30 s at pauses → DNSMOS + music/noise tagger + SNR/level/clipping
→ borderline chunks: enhance (GPU) + re-score → ASR (GPU) → text sanity → regex LID + code-mix composition
→ 24 kHz FLAC + rich metadata → parquet shards → Hugging Face every 10 h
```

Rejection is the default; every accept has to clear every gate. See the gate table in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#quality-gates-defaults-configschaashiniyaml).

## Components

| piece | where | what |
|---|---|---|
| `chaashini/` | CPU box | orchestrator package: workers, SQLite state, quality scoring, LID, packer, pusher, FastAPI + UI |
| `gpu/` | GPU box | pull-workers: `asr_worker.py` (diarization + ASR), `enhance_worker.py` (speech enhancement) |
| `ui/` | CPU box | single-file dashboard, 5 s polling, audio previews |
| `deploy/` | both | bootstrap scripts, control scripts (`chaashinictl`, `gpuctl`), nginx config, sync |
| `configs/chaashini.yaml` | CPU box | all tunables |

Models (GPU box): a streaming multilingual Indic ASR (transformers custom code), NVIDIA streaming
Sortformer diarization (NeMo), Resemble Enhance (own venv, old torch pins). CPU box: TenVAD, DNSMOS
(ONNX), AudioSet CNN14 tagger, yt-dlp + Deno, ffmpeg.

## Deploy

```bash
# once per box (installs uv, Python 3.11 venvs, models, Deno, libc++)
ssh root@101.53.139.186 'bash -s' < deploy/setup_cpu.sh
ssh root@101.53.138.193 'bash -s' < deploy/gpu/setup_gpu.sh     # needs /opt/chaashini/.hf_token_models (gated ASR repo)

# every code change
deploy/sync.sh all

# CPU box: secrets + nginx + start
ssh root@101.53.139.186
  cp /opt/chaashini/app/deploy/chaashini.env.example /opt/chaashini/chaashini.env   # HF_TOKEN, CHAASHINI_INTERNAL_TOKEN
  /opt/chaashini/app/deploy/install_nginx.sh
  chaashinictl start && chaashinictl status

# GPU box: env + start
ssh root@101.53.138.193
  cp /opt/chaashini/app/deploy/gpu/gpu.env.example /opt/chaashini/gpu.env            # same internal token
  gpuctl start && gpuctl status
```

## Source-site requirements (what you need for volume)

* **Deno** — yt-dlp needs a JavaScript runtime for the source's player challenges; installed by `setup_cpu.sh`.
* **Python ≥ 3.11** for yt-dlp (the venvs use 3.11 via uv).
* **Cookies** (`configs/cookies.txt`, Netscape format from a logged-in browser) — datacenter IPs get
  "confirm you're not a bot" checks under load; cookies from a throw-away account lift most of it.
* **Rotating residential proxy** (`source.proxy`) and/or a **PO-token provider** plugin for real scale.
  The pipeline backs off globally (5 min → 1 h) whenever it is throttled, so it degrades instead of breaking.
* Audio format selection is a Python selector (`ytsource.select_audio_format`): DASH over HLS (HLS fragments
  cannot be seeked reliably), non-DRC over loudness-processed "DRC" variants, the **original** track whenever
  auto-dubbed tracks exist, opus over AAC, then bitrate. Music/gaming/film categories and live streams are
  skipped before download, and the per-channel cap keeps speaker diversity.

## Configuration highlights

* `languages:` list with weights (discovery effort share); Urdu/Kashmiri disabled (unsupported by the ASR).
* `quality.accept` / `quality.enhance` thresholds; `quality.video_music_gate`.
* `vad.*` chunking: 0.5–30 s, split at the quietest pause near 15 s.
* `hf.push_every_hours: 10`, `hf.shard_target_mb: 400`.
* `workers.*` process counts; `storage.min_free_gb` pause threshold.

## Tests

```bash
cd /opt/chaashini/app && ../venv/bin/python -m pytest -q tests
```
