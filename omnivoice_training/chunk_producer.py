#!/usr/bin/env python3
"""Continuously turn long recordings into training shards, while training runs.

Only touches audio the trainer currently DROPS (> max_sec), so it strictly adds
data that has never been used. Round-robins across languages to keep the mix
roughly even without stalling on whichever language is slowest.

Each batch: stream long clips -> align (NaijaVox) -> cut to 0.8-30s with
ground-truth text -> write WAV+JSONL -> tokenise -> append shards to the live
data.lst. The trainer picks new shards up on its next restart.

State is a json file so a restart resumes instead of re-aligning hours of audio.
"""
from __future__ import annotations

import argparse, base64, io, json, os, subprocess, sys, time
from collections import defaultdict

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunk_align import chunk_recording           # noqa: E402
from hf_parquet import HttpFile, list_shards      # noqa: E402
import pyarrow.parquet as pq                      # noqa: E402

LANG_MAP = {"ibo": "ig", "yor": "yo", "hau": "ha", "pcm": "pcm"}
WHISPER_LANG = {"ibo": "ig", "yor": "yo", "hau": "ha", "pcm": "pcm"}
TARGET_SR = 24_000
COLS = ["audio_id", "speaker_id", "transcript", "language", "duration_seconds",
        "audio_path", "audio"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state(path):
    try:
        return json.load(open(path))
    except Exception:
        return {"done_ids": [], "shard_cursor": {}, "hours": {}, "chunks": {},
                "rejected": 0, "batches": 0}


def save_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, path)


def decode(blob):
    if not isinstance(blob, dict) or not blob.get("bytes"):
        return None, None
    wav, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def to_24k(wav, sr):
    if sr == TARGET_SR:
        return wav
    import librosa
    return librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR,
                            res_type="soxr_hq").astype(np.float32)


def align(wav, sr, gt, wlang, url, timeout):
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    r = requests.post(f"{url}/align", timeout=timeout, json={
        "audio_b64": base64.b64encode(buf.getvalue()).decode(),
        "transcript": gt, "language": wlang, "sample_rate": sr})
    if r.status_code != 200:
        raise RuntimeError(f"align HTTP {r.status_code}: {r.text[:180]}")
    return r.json()


def iter_long_clips(repo, lang, token, cursor, max_shards, min_len):
    """Yield rows longer than min_len, walking shards from `cursor`."""
    shards = list_shards(repo, "train", subdir=lang, token=token)
    end = min(cursor + max_shards, len(shards))
    for si in range(cursor, end):
        path, size = shards[si]
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
        try:
            pf = pq.ParquetFile(HttpFile(url, size, token))
            cols = [c for c in COLS if c in pf.schema_arrow.names]
            for rg in range(pf.metadata.num_row_groups):
                for row in pf.read_row_group(rg, columns=cols).to_pylist():
                    d = row.get("duration_seconds")
                    blob = row.get("audio") or row.get("audio_path")
                    if d is None and isinstance(blob, dict) and blob.get("bytes"):
                        d = len(blob["bytes"]) / 96000.0     # rough, refined later
                    if d and d > min_len:
                        yield si, row
        except Exception as e:
            log(f"  [{lang}] shard {si} unreadable: {e!r}")
            continue
    return


def tokenize_batch(py, src, jsonl, outdir, min_sec, max_sec, nj, workers, logf):
    os.makedirs(f"{outdir}/audios", exist_ok=True)
    os.makedirs(f"{outdir}/txts", exist_ok=True)
    cmd = [py, "-m", "omnivoice.scripts.extract_audio_tokens",
           "--input_jsonl", jsonl,
           "--tar_output_pattern", f"{outdir}/audios/shard-%06d.tar",
           "--jsonl_output_pattern", f"{outdir}/txts/shard-%06d.jsonl",
           "--tokenizer_path", "eustlb/higgs-audio-v2-tokenizer",
           "--min_length", str(min_sec), "--max_length", str(max_sec),
           "--nj_per_gpu", str(nj), "--loader_workers", str(workers),
           "--skip_errors", "--shuffle", "True", "--min_num_shards", "2"]
    env = dict(os.environ, PYTHONPATH=src + ":" + os.environ.get("PYTHONPATH", ""))
    with open(logf, "a") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=lf, env=env, timeout=7200).returncode
    lst = os.path.join(outdir, "data.lst")
    return (rc == 0 and os.path.exists(lst) and os.path.getsize(lst) > 0), lst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/jovyan/omnivoice-train")
    ap.add_argument("--repo", default="kapturecx/ohun")
    ap.add_argument("--langs", default="hau,ibo,yor,pcm")
    ap.add_argument("--align-url", default="http://127.0.0.1:8899")
    ap.add_argument("--min-len", type=float, default=30.0,
                    help="only recordings LONGER than this (the unused tail)")
    ap.add_argument("--min-sec", type=float, default=0.8)
    ap.add_argument("--max-sec", type=float, default=30.0)
    ap.add_argument("--clips-per-batch", type=int, default=24)
    ap.add_argument("--shards-per-pass", type=int, default=3)
    ap.add_argument("--align-timeout", type=int, default=1800)
    ap.add_argument("--nj-per-gpu", type=int, default=1)
    ap.add_argument("--loader-workers", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--vram-floor-mb", type=int, default=6000,
                    help="pause while the trainer needs the GPU")
    a = ap.parse_args()

    base = a.base
    run = f"{base}/run_prog"
    os.makedirs(f"{run}/tokens/train", exist_ok=True)
    os.makedirs(f"{run}/logs", exist_ok=True)
    os.makedirs(f"{run}/work", exist_ok=True)
    state_p = f"{run}/producer_state.json"
    st = load_state(state_p)
    done = set(st["done_ids"])
    token = open(f"{base}/token.read").read().strip()
    py = f"{base}/.venv/bin/python"
    src = f"{base}/OmniVoice-src"
    langs = [l.strip() for l in a.langs.split(",") if l.strip()]
    master = f"{run}/tokens/train/data.lst"

    log(f"producer: langs={langs} min_len={a.min_len}s -> chunks "
        f"[{a.min_sec}, {a.max_sec}]s | resuming with "
        f"{sum(st['hours'].values()):.2f}h already produced")

    def vram_free_mb():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20).stdout
            return int(out.strip().split("\n")[0])
        except Exception:
            return 99999

    batches = 0
    while True:
        if a.max_batches and batches >= a.max_batches:
            log("max-batches reached"); break
        made_any = False
        # round-robin so no language runs far ahead of the others
        for lang in langs:
            while vram_free_mb() < a.vram_floor_mb:
                log(f"  GPU busy ({vram_free_mb()} MB free) - yielding to trainer")
                time.sleep(60)

            cursor = int(st["shard_cursor"].get(lang, 0))
            wdir = f"{run}/work/{lang}_{batches}"
            os.makedirs(f"{wdir}/wav", exist_ok=True)
            recs, n_clips, last_si = [], 0, cursor
            t0 = time.time()

            for si, row in iter_long_clips(a.repo, lang, token, cursor,
                                           a.shards_per_pass, a.min_len):
                last_si = si
                aid = str(row.get("audio_id") or "")
                if not aid or aid in done:
                    continue
                if not aid.startswith(lang + "_"):
                    log(f"  [{lang}] FATAL wrong-language id {aid!r}"); sys.exit(2)
                gt = (row.get("transcript") or "").strip()
                blob = row.get("audio") or row.get("audio_path")
                wav, sr = decode(blob)
                if wav is None or not gt:
                    done.add(aid); continue
                try:
                    res = align(wav, sr, gt, WHISPER_LANG[lang], a.align_url,
                                a.align_timeout)
                except Exception as e:
                    log(f"  [{lang}] align failed {aid}: {e!r}"); continue

                chunks = chunk_recording(res.get("words") or [], gt,
                                         len(wav) / sr, a.min_sec, a.max_sec)
                done.add(aid)
                if not chunks:
                    st["rejected"] += 1
                    continue
                w24 = to_24k(wav, sr)
                for ci, c in enumerate(chunks):
                    i0, i1 = int(c.start * TARGET_SR), int(c.end * TARGET_SR)
                    seg = w24[max(0, i0):min(len(w24), i1)]
                    if len(seg) < a.min_sec * TARGET_SR:
                        continue
                    cid = f"{aid}__c{ci:04d}"
                    p = f"{wdir}/wav/{cid}.wav"
                    sf.write(p, seg, TARGET_SR, subtype="PCM_16")
                    recs.append({"id": cid, "audio_path": os.path.abspath(p),
                                 "text": c.text, "language_id": LANG_MAP[lang],
                                 "speaker_id": row.get("speaker_id"),
                                 "duration": round(len(seg) / TARGET_SR, 3),
                                 "source_id": aid})
                n_clips += 1
                if n_clips >= a.clips_per_batch:
                    break

            if not recs:
                st["shard_cursor"][lang] = last_si + 1
                save_state(state_p, {**st, "done_ids": sorted(done)})
                continue

            jl = f"{wdir}/batch.jsonl"
            with open(jl, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            hrs = sum(r["duration"] for r in recs) / 3600
            log(f"  [{lang}] {n_clips} long clips -> {len(recs)} chunks "
                f"({hrs:.2f}h) in {time.time()-t0:.0f}s; tokenising")

            tdir = f"{run}/tokens/train/{lang}_b{batches}"
            ok, lst = tokenize_batch(py, src, jl, tdir, a.min_sec, a.max_sec,
                                     a.nj_per_gpu, a.loader_workers,
                                     f"{run}/logs/tokenize.log")
            if ok:
                with open(master, "a") as mf, open(lst) as sfh:
                    mf.write(sfh.read())
                st["hours"][lang] = st["hours"].get(lang, 0.0) + hrs
                st["chunks"][lang] = st["chunks"].get(lang, 0) + len(recs)
                made_any = True
                total = sum(st["hours"].values())
                log(f"  [{lang}] +{hrs:.2f}h -> {st['hours'][lang]:.2f}h for {lang}, "
                    f"{total:.2f}h total ({sum(st['chunks'].values())} chunks)")
            else:
                log(f"  [{lang}] tokenisation FAILED - batch dropped")

            os.system(f"rm -rf {wdir}")      # WAVs have served their purpose
            st["shard_cursor"][lang] = last_si + 1
            st["batches"] = batches
            save_state(state_p, {**st, "done_ids": sorted(done)})

        batches += 1
        if not made_any:
            log("no new data produced this pass - sleeping 120s")
            time.sleep(120)


if __name__ == "__main__":
    main()
