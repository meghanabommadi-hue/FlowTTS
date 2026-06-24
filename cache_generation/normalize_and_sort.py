#!/usr/bin/env python3
"""
Normalize bot_sentences.txt using text_normalize and sort by frequency.

Reads bot_sentences.txt, normalizes each line via normalize_text(),
counts how many raw lines produced each normalized form (frequency),
then writes normalized_sentences.txt sorted highest → lowest frequency.

Usage:
    python normalize_and_sort.py
    python normalize_and_sort.py --in bot_sentences.txt --out normalized_sentences.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from text_normalize import normalize_text  # noqa: E402

import re

DEFAULT_IN   = str(HERE / "norm1_sentences.txt")
DEFAULT_OUT     = str(HERE / "normalized_sentences.txt")
DEFAULT_TOP     = str(HERE / "top1000_sentences.txt")
DEFAULT_PARQUET = str(HERE / "normalized_sentences.parquet")

# Patterns that indicate noise / non-TTS-suitable text
_TAG_RE      = re.compile(r'<[^>]*>')          # <channel|>, <|channel>, etc.
_PARENS_RE   = re.compile(r'^\(.*\)$')         # purely parenthesised like (Retry)


def is_clean(text: str) -> bool:
    """Return True if the text is suitable for TTS synthesis."""
    # Reject anything that still has tag syntax
    if '<' in text or '|>' in text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # Skip parenthesised-only noise like (Retry), (correction)
    if _PARENS_RE.match(stripped):
        return False
    # Skip bare punctuation / very short fragments
    if len(stripped) < 8:
        return False
    # Skip "thought" artifact lines (with or without leading dashes)
    if re.fullmatch(r'[-\s]*thought[-\s]*', stripped, re.IGNORECASE):
        return False
    # Skip lines that look like "X (X)" digit repetitions e.g. "दो (दो)"
    if re.fullmatch(r'\S+\s+\(\S+\)', stripped):
        return False
    # Skip lines with = sign (malformed number expansions like "3189 = ...")
    if '=' in stripped:
        return False
    # Skip lines ending in "..." or "निवासी..." (incomplete/truncated)
    if stripped.endswith('...') or 'निवासी' in stripped:
        return False
    # Skip lines that start with "Correct:" (debug/correction artifacts)
    if stripped.lower().startswith('correct:'):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize sentences and sort by frequency")
    parser.add_argument("--in",  dest="inp", default=DEFAULT_IN,  help="Input raw sentences file")
    parser.add_argument("--out", dest="out", default=DEFAULT_OUT, help="Output normalized file")
    parser.add_argument("--top",     dest="top",     default=DEFAULT_TOP,     help="Output top-1000 txt file")
    parser.add_argument("--parquet", dest="parquet", default=DEFAULT_PARQUET, help="Output parquet file (text + frequency)")
    args = parser.parse_args()

    inp_path = Path(args.inp)
    out_path = Path(args.out)

    if not inp_path.is_file():
        print(f"[ERROR] Input file not found: {inp_path}", file=sys.stderr)
        sys.exit(1)

    lines = inp_path.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    print(f"Read {total:,} lines from {inp_path}")

    freq: Counter[str] = Counter()
    skipped = 0

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            skipped += 1
            continue
        # Support optional tab-separated "sentence\tcount" format from normalization_1.py
        sentence = line.split("\t")[0].strip() if "\t" in line else line
        if not sentence:
            skipped += 1
            continue
        try:
            norm = normalize_text(sentence)
        except Exception:
            skipped += 1
            continue
        if norm and is_clean(norm):
            freq[norm] += 1
        else:
            skipped += 1

        if i % 100_000 == 0:
            print(f"  processed {i:,}/{total:,} ...")

    print(f"\nUnique normalized sentences (before case-merge) : {len(freq):,}")
    print(f"Skipped (empty/error)                          : {skipped:,}")

    # Merge case-insensitive duplicates: keep the most frequent casing variant
    case_merged: Counter[str] = Counter()
    canonical: dict[str, str] = {}  # lower → chosen canonical form
    for text, count in freq.items():
        key = text.lower()
        if key not in canonical or count > freq[canonical[key]]:
            canonical[key] = text
    for text, count in freq.items():
        case_merged[canonical[text.lower()]] += count

    # Sort by frequency descending, then alphabetically for ties
    sorted_items = sorted(case_merged.items(), key=lambda x: (-x[1], x[0]))

    with out_path.open("w", encoding="utf-8") as f:
        for norm_text, count in sorted_items:
            f.write(f"{norm_text}\n")

    print(f"After case-merge            : {len(case_merged):,}")
    print(f"Saved to                    : {out_path.resolve()}")

    # Write top 1000
    top_path = Path(args.top)
    with top_path.open("w", encoding="utf-8") as f:
        for norm_text, count in sorted_items[:1000]:
            f.write(f"{norm_text}\n")

    print(f"Top 1000 saved to           : {top_path.resolve()}")

    # Save parquet with two columns: text, frequency
    parquet_path = Path(args.parquet)
    try:
        import pandas as pd
        df = pd.DataFrame(sorted_items, columns=["text", "frequency"])
        df.to_parquet(parquet_path, index=False)
        print(f"Parquet saved to            : {parquet_path.resolve()}")
    except ImportError:
        print("[WARN] pandas not available — skipping parquet export")

    print(f"\nTop 10 most frequent sentences:")
    for norm_text, count in sorted_items[:10]:
        print(f"  [{count:>6}x] {norm_text[:100]}")


if __name__ == "__main__":
    main()
