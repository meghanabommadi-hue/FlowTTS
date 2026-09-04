"""Staging of accepted chunks: audio at the export rate (FLAC) + one JSON sidecar each.

Layout: <staging>/<lang>/<chunk_id>.flac + .json.  The packer sweeps this directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .audio import encode_bytes


def stage_chunk(staging_dir: Path, lang: str, chunk_id: str, audio: np.ndarray, sr: int, meta: dict,
                fmt: str = "flac") -> tuple[str, int]:
    d = staging_dir / lang
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{chunk_id}.{fmt}"
    data = encode_bytes(audio, sr, fmt)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, p)
    with open(p.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return str(p), len(data)


def staged_seconds(staging_dir: Path) -> dict[str, float]:
    """Total staged seconds per language (from sidecars)."""
    out: dict[str, float] = {}
    if not staging_dir.exists():
        return out
    for lang_dir in staging_dir.iterdir():
        if not lang_dir.is_dir():
            continue
        tot = 0.0
        for js in lang_dir.glob("*.json"):
            try:
                with open(js, encoding="utf-8") as f:
                    tot += float(json.load(f).get("duration_s", 0.0))
            except Exception:  # noqa: BLE001
                continue
        if tot > 0:
            out[lang_dir.name] = tot
    return out
