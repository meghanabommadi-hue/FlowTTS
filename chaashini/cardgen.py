"""Dataset card (README.md) for the public dataset repository.

Deliberately generic about provenance and tooling: the card describes the corpus, its
schema and its quality gates, and lists per-language statistics that are refreshed on
every push.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .languages import LANGUAGES


def _fmt_h(sec: float) -> str:
    return f"{sec / 3600:.2f}"


def render_card(repo_id: str, name: str, per_lang: dict[str, dict], totals: dict, export_sr: int) -> str:
    langs_sorted = sorted(per_lang.items(), key=lambda kv: -kv[1]["seconds"])
    lang_codes = [l for l, _ in langs_sorted]
    configs_yaml = ["configs:", "- config_name: default", "  data_files:", "  - split: train", "    path: data/*/*.parquet"]
    for l in lang_codes:
        configs_yaml += [f"- config_name: {l}", "  data_files:", "  - split: train", f"    path: data/{l}/*.parquet"]
    language_yaml = "language:\n" + "\n".join(f"- {l}" for l in lang_codes) if lang_codes else "language: []"
    rows = "\n".join(
        f"| `{l}` | {LANGUAGES[l].name if l in LANGUAGES else l} | {v['chunks']:,} | {_fmt_h(v['seconds'])} | {v['seconds'] / max(1, v['chunks']):.1f} s | {v.get('avg_ovrl', 0):.2f} |"
        for l, v in langs_sorted)
    now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M IST")
    size_cat = "n<1K" if totals["chunks"] < 1000 else "1K<n<10K" if totals["chunks"] < 10_000 else "10K<n<100K" if totals["chunks"] < 100_000 else "100K<n<1M" if totals["chunks"] < 1_000_000 else "1M<n<10M"
    return f"""---
license: apache-2.0
pretty_name: {name}
task_categories:
- text-to-speech
- automatic-speech-recognition
- audio-classification
{language_yaml}
multilinguality:
- multilingual
size_categories:
- {size_cat}
tags:
- speech
- indic
- indian-languages
- tts
- multilingual
- clean-speech
{chr(10).join(configs_yaml)}
---

# {name} (चाशनी)

**{name}** — Hindi/Urdu for *sugar syrup* — is a continuously growing corpus of **clean, single-speaker,
studio-grade Indian-language speech** built for training speech models (text-to-speech, speech
recognition, speech language models). Every clip in the corpus has passed a strict multi-stage
quality gate; the aim is purity over volume.

* **Total:** {totals['chunks']:,} clips · **{_fmt_h(totals['seconds'])} hours** · {len(per_lang)} languages
* **Format:** mono {export_sr // 1000} kHz FLAC (`audio` column) with a verbatim transcript and rich per-clip metadata
* **Clip length:** 0.5 s – 30 s, cut at natural pauses
* **Last refreshed:** {now}

The corpus grows automatically: new shards are appended every ~2 hours of newly accepted audio.

## Languages

| code | language | clips | hours | avg clip | avg quality (OVRL) |
|---|---|---:|---:|---:|---:|
{rows}

## What makes a clip "pristine"

Audio is sourced from publicly available spoken-word recordings (talks, interviews, narration,
lectures, podcasts and similar long-form speech). Each recording then passes through:

1. **Source-level screening** – recordings dominated by music, singing, or non-speech content are discarded whole.
2. **Speech activity detection** – frame-accurate speech/non-speech decisions; clips are cut only at genuine pauses.
3. **Speaker purity** – a neural diarizer labels speakers; every clip contains exactly **one** speaker and
   any region where voices overlap (plus a safety margin) is removed. Speaker labels are consistent within a source.
4. **Perceptual quality scoring** – each clip receives non-intrusive MOS-style scores for signal quality,
   background intrusiveness and overall quality, plus tagger probabilities for music / singing / noise,
   an SNR estimate, loudness, clipping and effective bandwidth. Clips outside strict thresholds are rejected;
   borderline clips are passed through a speech enhancer and **re-scored** (never accepted blindly) — such clips are flagged `enhanced=true`.
5. **Transcription** – a multilingual Indic ASR system transcribes each accepted clip; clips with implausible
   character rates (empty, hallucinated or clipped transcripts) or low recogniser confidence are rejected.
6. **Language identification** – script-aware identification on the transcript gives the language, a
   confidence, and the full **language composition** of the clip (code-mixing is common in Indian speech and is
   preserved, not filtered; `language` is the dominant language and `language_mix` holds the shares).

## Schema

| column | type | description |
|---|---|---|
| `id` | string | unique clip id |
| `audio` | Audio | mono {export_sr // 1000} kHz FLAC |
| `text` | string | transcript (native script; borrowed English words may appear in Latin script) |
| `language` | string | dominant language (ISO 639 code) |
| `language_name` | string | human-readable language name |
| `language_confidence` | float | confidence of the language decision, 0–1 |
| `language_mix` | string (JSON) | share of each language/script in the clip, e.g. `{{"hi": 0.82, "en": 0.18}}` |
| `script` | string | dominant writing system of the transcript |
| `code_mixed` | bool | true if a secondary language exceeds 15 % of the tokens |
| `duration_s` | float | clip duration in seconds |
| `sample_rate` | int | {export_sr} |
| `speaker_id` | string | speaker label, consistent within a source recording |
| `source_id` | string | opaque, stable id of the source recording (for grouping / de-duplication) |
| `segment_index` | int | order of the clip within its source |
| `enhanced` | bool | clip was passed through speech enhancement before re-scoring |
| `dnsmos_sig` / `dnsmos_bak` / `dnsmos_ovrl` / `dnsmos_p808` | float | perceptual quality scores (1–5): signal, background, overall, P.808 MOS |
| `music_prob` / `speech_prob` / `noise_prob` | float | tagger probabilities (0–1) |
| `snr_db` | float | estimated signal-to-noise ratio |
| `rms_dbfs` / `peak_dbfs` | float | loudness and peak level |
| `clipping_ratio` | float | fraction of clipped samples |
| `bandwidth_hz` | float | effective audio bandwidth |
| `vad_speech_ratio` | float | fraction of the clip that is active speech |
| `speaker_dominance` | float | fraction of speaker-labelled frames belonging to the clip's speaker |
| `chars_per_sec` | float | transcript characters per second |
| `asr_confidence` | float | mean posterior of the recogniser's emitted tokens (0–1); low values flag hard audio |
| `genre` | string | coarse content genre of the source (talk, narration, interview, …) |
| `created_at` | string | ISO-8601 timestamp when the clip was accepted |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", "hi", split="train", streaming=True)   # one language
row = next(iter(ds))
print(row["text"], row["audio"]["sampling_rate"], row["language_mix"])

all_langs = load_dataset("{repo_id}", "default", split="train", streaming=True)
```

Filtering tips: `dnsmos_ovrl >= 3.2` and `enhanced == False` gives the most conservative subset;
`language_confidence >= 0.8` and `code_mixed == False` gives monolingual clips.

## Licensing and intended use

The corpus is released under Apache-2.0 for research and commercial speech-technology development.
Clips are short excerpts of publicly available spoken-word material processed for machine learning;
no personal identifiers are stored beyond an opaque source id. If you believe content should be
removed, open a discussion on this repository.

## Citation

```
@misc{{{name.lower()}2026,
  title  = {{{name}: a quality-gated multilingual Indian speech corpus}},
  author = {{Kapture CX}},
  year   = {{2026}},
  url    = {{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""
