# OmniVoice fine-tuning on Nigerian languages (`ohun`)

Unattended pipeline that fine-tunes [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
on Igbo, Yorùbá, Hausa and Nigerian Pidgin from the `kapturecx/ohun` dataset,
with TensorBoard (including **listenable audio previews**) and automatic
checkpoint pushes to the Hub.

## Why it is shaped this way

Three findings from reading the upstream source drove the design:

1. **Training cannot consume raw audio.** `omnivoice/data/dataset.py` reads
   WebDataset tars of pre-extracted Higgs-Audio-v2 tokens (`<key>.npy`, int16
   `[8, T]`) plus a side-car JSONL. `JsonlDatasetReader` additionally requires
   `audio_path` to be a **local file that exists**. So "stream straight from HF
   into training" is not possible — audio must be materialised and tokenised
   first. That is stages 1 and 3.
2. **The training model cannot synthesize.** `builder.py` loads it with
   `train=True`, which leaves the text/audio tokenizers as `None`, and
   `generate()` raises in that state. Audio previews therefore run in a
   **separate process** against a saved snapshot — which also means a synthesis
   crash or OOM cannot take training down.
3. **There is no callback system.** `OmniTrainer` is a hand-written loop, so
   hooks are added by subclassing `evaluate()` (`train_ohun.py`), not by
   registration. Extra settings arrive via env vars because
   `TrainingConfig.from_json` **silently drops unknown keys**.

## Language codes

`ohun` config names are not OmniVoice language ids. The mapping is applied in
`ohun_prepare.py` and is load-bearing — an unknown id is silently downgraded to
language-agnostic, which would quietly waste the run:

| ohun config | OmniVoice id | language |
|---|---|---|
| `ibo` | `ig` | Igbo |
| `yor` | `yo` | Yorùbá |
| `hau` | `ha` | Hausa |
| `pcm` | `pcm` | Nigerian Pidgin |

All four are in `LANG_IDS` (646 languages). Upstream lists only 13.7 / 15.7 /
17.8 / 11.0 pretraining hours for them respectively, so there is real headroom
for fine-tuning.

## Stages

| # | Stage | What it does |
|---|---|---|
| 0 | wait | Polls the ohun uploader until the dataset push reports `done` |
| 1 | prepare | Streams from the Hub, filters, writes 24 kHz WAVs + JSONL |
| 2 | evalset | Picks fixed eval prompts from the dev split (real transcripts) |
| 3 | tokenize | `omnivoice.scripts.extract_audio_tokens` → WebDataset shards |
| 4 | data config | Writes `data_config.json` pointing at both `data.lst` files |
| 5 | probe | Finds the largest `batch_tokens` that trains without OOM |
| 6 | train | `supervise_train.sh` keeps it alive overnight |

Each stage drops a marker in `run/stage/`, so a restart resumes rather than
redoing hours of work.

## Data filtering

The corpora contain multi-minute rows — one igbo "utterance" is **1906 s**.
Those would blow past `max_sample_tokens` and dominate a packed batch, so
`prepare` keeps only `1.5 s ≤ duration ≤ 25 s`, drops empty transcripts and
near-silent clips, and measures duration from the **decoded audio** (the `yor`
and `pcm` configs have no `duration_seconds` column, and `hau` has no
`language` column).

## Failure handling

`supervise_train.sh` classifies failures rather than blindly retrying:

* **CUDA OOM** → shrink `batch_tokens` 25 % and restart
* **flex_attention problem** → fall back to the `sdpa` padding path
* **anything else** → restart, resuming from the newest `checkpoint-N`

Checkpoints resume by directory name (`load_checkpoint` parses the step out of
the basename), so they must never be renamed. `keep_last_n_checkpoints` is set
to `3` — the upstream default of `-1` disables rotation and would fill the disk.

HF pushes and audio previews are both best-effort: a Hub rate limit or a
synthesis failure logs and continues, it never kills training.

## Usage

```bash
# on the training box
printf %s "$HF_WRITE_TOKEN" > /opt/omnivoice-train/token.write
/opt/omnivoice-train/omnivoice_training/run.sh     # detached, resumable
/opt/omnivoice-train/omnivoice_training/stop.sh
```

TensorBoard: `http://<host>/tb/` — `train/loss`, `eval/loss`, `eval/best_loss`,
and `eval_audio/<lang>/<id>` on the step slider so quality can be listened to
over time. Preview WAVs are also at `http://<host>/wav/`.

Tunables: `HOURS_PER_LANG` (default 25), `MAX_SEC`, `OHUN_HF_REPO`,
`OHUN_EVAL_INFER_EVERY`, `WAIT_FOR_DATA=0` to skip the upload gate.
