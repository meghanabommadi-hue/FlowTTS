#!/usr/bin/env python3
"""Trim prepared manifests so every language contributes equal audio hours.

The corpus is heavily skewed - pcm has 335k train utterances against hau's 40k -
so training on it as-is would mostly teach Nigerian Pidgin. Each language is
capped at the same number of hours before tokenisation, which also avoids
spending GPU time tokenising audio that would then be dropped.

Operates on the prepare stage's <lang>_<split>.jsonl files, which carry a
per-utterance `duration`.
"""
from __future__ import annotations

import argparse, glob, json, os


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--splits", default="train dev")
    ap.add_argument("--tolerance", type=float, default=1.02)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for split in a.splits.split():
        files = sorted(glob.glob(os.path.join(a.prep_dir, f"*_{split}.jsonl")))
        stats = {}
        for f in files:
            lang = os.path.basename(f).rsplit(f"_{split}.jsonl", 1)[0]
            rows = load(f)
            if not rows:
                continue
            hrs = sum(float(r.get("duration") or 0) for r in rows) / 3600
            stats[lang] = (hrs, f, rows)
        if len(stats) < 2:
            print(f"[{split}] fewer than 2 languages - nothing to balance")
            continue

        lo = min(v[0] for v in stats.values())
        print(f"[{split}] before: "
              + ", ".join(f"{k}={v[0]:.2f}h" for k, v in sorted(stats.items())))
        print(f"[{split}] target per language: {lo:.2f}h")
        for lang, (hrs, f, rows) in sorted(stats.items()):
            if hrs <= lo * a.tolerance:
                print(f"  {lang}: {hrs:.2f}h kept as-is ({len(rows)} utts)")
                continue
            # keep a speaker-diverse subset rather than a prefix: the manifests
            # are grouped by shard, so a prefix would over-represent a few voices
            rows_sorted = sorted(rows, key=lambda r: (str(r.get("speaker_id")), r["id"]))
            keep, acc, i = [], 0.0, 0
            by_spk = {}
            for r in rows_sorted:
                by_spk.setdefault(str(r.get("speaker_id")), []).append(r)
            spk = list(by_spk)
            while acc < lo * 3600 and spk:
                progressed = False
                for s in list(spk):
                    if not by_spk[s]:
                        spk.remove(s); continue
                    r = by_spk[s].pop(0)
                    keep.append(r); acc += float(r.get("duration") or 0)
                    progressed = True
                    if acc >= lo * 3600:
                        break
                if not progressed:
                    break
            if not a.dry_run:
                with open(f, "w", encoding="utf-8") as fh:
                    for r in keep:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {lang}: {hrs:.2f}h -> {acc/3600:.2f}h "
                  f"({len(rows)} -> {len(keep)} utts, {len(by_spk)} speakers)")


if __name__ == "__main__":
    main()
