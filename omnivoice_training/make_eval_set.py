#!/usr/bin/env python3
"""Build the fixed eval prompt set used for audio previews.

Sentences are taken verbatim from the prepared dev manifests rather than
hand-written: none of these four languages should have text invented for them,
and real transcripts keep the preview in-domain. Each prompt is paired with a
DIFFERENT utterance from the same speaker as the voice-cloning reference, so
the preview exercises the model's actual zero-shot cloning path.

The set is written once and then held fixed, so TensorBoard's step slider
compares like with like across the whole run.
"""
from __future__ import annotations

import argparse, glob, json, os, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang", type=int, default=3)
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--max-chars", type=int, default=160)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    out = []
    for dev in sorted(glob.glob(os.path.join(a.prep_dir, "*_dev.jsonl"))):
        cfg = os.path.basename(dev).replace("_dev.jsonl", "")
        rows = [json.loads(l) for l in open(dev, encoding="utf-8") if l.strip()]
        if not rows:
            print(f"[{cfg}] dev manifest empty - skipped")
            continue
        lang = rows[0].get("language_id")

        # group by speaker so the reference can come from the same voice
        by_spk = {}
        for r in rows:
            by_spk.setdefault(r.get("speaker_id") or "_", []).append(r)
        usable = [(s, v) for s, v in by_spk.items() if len(v) >= 2]
        usable.sort(key=lambda kv: (-len(kv[1]), str(kv[0])))
        if not usable:
            usable = [(s, v * 2) for s, v in list(by_spk.items())[:a.per_lang]]

        picked = 0
        for spk, items in usable:
            if picked >= a.per_lang:
                break
            cands = [r for r in items
                     if a.min_chars <= len(r["text"]) <= a.max_chars]
            if len(cands) < 2:
                continue
            cands.sort(key=lambda r: r["id"])
            tgt, ref = cands[0], cands[1]
            out.append({
                "id": f"{cfg}_{picked}",
                "language": lang,
                "text": tgt["text"],
                "ref_audio": ref["audio_path"],
                "ref_text": ref["text"],
                "speaker_id": spk,
            })
            picked += 1
        print(f"[{cfg}->{lang}] {picked} eval prompts "
              f"from {len(rows)} dev utts / {len(by_spk)} speakers")

    if not out:
        raise SystemExit("no eval prompts could be built - check the dev manifests")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} eval prompts -> {a.out}")


if __name__ == "__main__":
    main()
