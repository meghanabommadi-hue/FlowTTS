#!/usr/bin/env python3
"""Stage 1: HuggingFace `ohun` -> local WAVs + JSONL manifests.

OmniVoice's tokenizer reads a JSONL whose `audio_path` must be a real local
file (omnivoice/data/dataset.py:JsonlDatasetReader checks os.path.exists), so
the audio has to be materialised before tokenisation. We stream from the Hub
and write only the utterances that survive filtering, capped at a budget of
hours per language, so disk stays bounded.

Language ids MUST be OmniVoice's canonical codes (omnivoice/utils/lang_map.py):
an unrecognised id is silently downgraded to language-agnostic, which would
quietly waste the whole run.
"""
from __future__ import annotations

import argparse, io, json, os, sys, time
from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_parquet import iter_rows  # noqa: E402

# ohun config name -> OmniVoice canonical language id (verified in LANG_IDS)
LANG_MAP = {"ibo": "ig", "yor": "yo", "hau": "ha", "pcm": "pcm"}

# The upstream corpora that `ohun` repackages. ohun re-registers the SAME
# parquet objects, so these are byte-identical sources - useful when the ohun
# upload is still in flight and the GPU would otherwise sit idle.
SOURCE_REPOS = {
    "ibo": "Africanvoice/African_voices_igbo",
    "yor": "Africanvoice/African_voices_yoruba",
    "hau": "Africanvoice/African_voices_hausa",
    "pcm": "Africanvoice/African_voices_naija",
}
TARGET_SR = 24_000  # HIGGS_INPUT_SAMPLE_RATE (extract_audio_tokens.py:72)


@dataclass
class Stats:
    seen: int = 0
    kept: int = 0
    sec_kept: float = 0.0
    drop_short: int = 0
    drop_long: int = 0
    drop_no_text: int = 0
    drop_decode: int = 0
    drop_silent: int = 0


def find_audio(row):
    """ibo/yor/pcm carry audio in `audio_path`; hau carries it in `audio`."""
    for key in ("audio_path", "audio"):
        v = row.get(key)
        if isinstance(v, dict) and v.get("bytes"):
            return v["bytes"]
        if isinstance(v, dict) and v.get("array") is not None:
            return v
    return None


def decode(blob, target_sr=TARGET_SR):
    """Return float32 mono at target_sr, or None."""
    if isinstance(blob, dict):                      # already decoded by datasets
        wav = np.asarray(blob["array"], dtype=np.float32)
        sr = int(blob["sampling_rate"])
    else:
        wav, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return wav.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="kapturecx/ohun")
    ap.add_argument("--langs", default="ibo,yor,hau,pcm")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True, help="output root")
    ap.add_argument("--hours-per-lang", type=float, default=25.0)
    ap.add_argument("--dev-minutes-per-lang", type=float, default=12.0)
    ap.add_argument("--min-sec", type=float, default=1.5)
    ap.add_argument("--max-sec", type=float, default=25.0)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--status", default=None, help="json status file for the UI")
    ap.add_argument("--read-workers", type=int, default=8,
                    help="parallel parquet shard readers")
    ap.add_argument("--skip-shards", type=int, default=0,
                    help="resume: skip this many shards of the (shuffled) list")
    ap.add_argument("--max-shards", type=int, default=0,
                    help="stop after this many shards (0 = no limit)")
    ap.add_argument("--shuffle-seed", type=int, default=17,
                    help="shard order seed - avoids taking every clip from one speaker")
    ap.add_argument("--from-sources", action="store_true",
                    help="read the upstream Africanvoice corpora instead of the "
                         "merged ohun repo (identical audio, already complete)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    langs = [l.strip() for l in a.langs.split(",") if l.strip()]
    for l in langs:
        if l not in LANG_MAP:
            sys.exit(f"unknown ohun config {l!r}; expected one of {list(LANG_MAP)}")

    overall = {}
    t0 = time.time()
    for cfg in langs:
        lang_id = LANG_MAP[cfg]
        budget_s = a.hours_per_lang * 3600
        dev_s = a.dev_minutes_per_lang * 60
        wav_dir = os.path.join(a.out, "wav", cfg)
        os.makedirs(wav_dir, exist_ok=True)
        train_p = os.path.join(a.out, f"{cfg}_train.jsonl")
        dev_p = os.path.join(a.out, f"{cfg}_dev.jsonl")
        st = Stats()

        print(f"[{cfg}->{lang_id}] streaming {'sources' if a.from_sources else a.repo}:{cfg}/{a.split} "
              f"budget={a.hours_per_lang}h (+{a.dev_minutes_per_lang}m dev)", flush=True)
        repo = SOURCE_REPOS[cfg] if a.from_sources else a.repo
        # Only the columns we need - the audio blob dominates, so pulling the
        # rest costs nothing, but naming them keeps the read explicit.
        cols = ["audio_id", "speaker_id", "transcript", "language",
                "audio_path", "audio"]
        ds = iter_rows(repo, a.split, columns=cols, token=a.token,
                       shuffle_seed=a.shuffle_seed, workers=a.read_workers,
                       skip_shards=a.skip_shards, max_shards=a.max_shards or None)

        ftr = open(train_p, "w", encoding="utf-8")
        fdv = open(dev_p, "w", encoding="utf-8")
        try:
            for row in ds:
                st.seen += 1
                text = (row.get("transcript") or "").strip()
                if not text:
                    st.drop_no_text += 1
                    continue
                blob = find_audio(row)
                if blob is None:
                    st.drop_decode += 1
                    continue
                try:
                    wav = decode(blob)
                except Exception:
                    st.drop_decode += 1
                    continue
                dur = len(wav) / TARGET_SR
                if dur < a.min_sec:
                    st.drop_short += 1
                    continue
                if dur > a.max_sec:
                    # The corpora contain multi-minute "utterances" (one igbo row
                    # is 1906 s). Those would blow past max_sample_tokens and
                    # dominate a packed batch, so they are dropped, not truncated.
                    st.drop_long += 1
                    continue
                peak = float(np.max(np.abs(wav))) if wav.size else 0.0
                if peak < 1e-4:
                    st.drop_silent += 1
                    continue

                uid = row.get("audio_id") or f"{cfg}_{st.kept:08d}"
                uid = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(uid))
                path = os.path.join(wav_dir, f"{uid}.wav")
                sf.write(path, wav, TARGET_SR, subtype="PCM_16")

                rec = {"id": uid, "audio_path": os.path.abspath(path),
                       "text": text, "language_id": lang_id,
                       "speaker_id": row.get("speaker_id"), "duration": round(dur, 3)}
                # fill the dev set first, then train
                (fdv if st.sec_kept < dev_s else ftr).write(
                    json.dumps(rec, ensure_ascii=False) + "\n")
                st.kept += 1
                st.sec_kept += dur

                if st.kept % 500 == 0:
                    print(f"  [{cfg}] kept={st.kept} {st.sec_kept/3600:.2f}h "
                          f"seen={st.seen} ({time.time()-t0:.0f}s)", flush=True)
                    if a.status:
                        _write_status(a.status, cfg, st, overall, t0, a)
                if st.sec_kept >= budget_s + dev_s:
                    break
        finally:
            ftr.close(); fdv.close()

        overall[cfg] = asdict(st) | {"lang_id": lang_id, "hours": st.sec_kept / 3600}
        print(f"[{cfg}] DONE kept={st.kept} ({st.sec_kept/3600:.2f}h) of {st.seen} seen | "
              f"short={st.drop_short} long={st.drop_long} notext={st.drop_no_text} "
              f"decode={st.drop_decode} silent={st.drop_silent}", flush=True)
        if a.status:
            _write_status(a.status, cfg, st, overall, t0, a)

    with open(os.path.join(a.out, "prepare_stats.json"), "w") as f:
        json.dump(overall, f, indent=1)
    tot = sum(v["hours"] for v in overall.values())
    print(f"\nprepared {tot:.2f}h across {len(overall)} languages -> {a.out}")


def _write_status(path, cur, st, overall, t0, a):
    d = {"stage": "prepare", "current_lang": cur, "elapsed_s": round(time.time() - t0),
         "current": asdict(st) | {"hours": st.sec_kept / 3600},
         "done_langs": overall, "target_hours_per_lang": a.hours_per_lang,
         "updated_utc": int(time.time())}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
