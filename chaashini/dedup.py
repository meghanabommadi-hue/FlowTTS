"""Duplicate-work guards.

* id-level: `videos.id` is the primary key and rows persist forever (insert-or-ignore everywhere).
* upload-level: same normalised title and same duration (+-1 s) as an item already taken -> re-upload / mirror.
* content-level: a coarse loudness-envelope fingerprint of the decoded audio; two recordings of (almost)
  the same length whose envelopes agree on >= 90 % of 64 coarse bins are the same audio.
"""
from __future__ import annotations

import sqlite3
import unicodedata

import numpy as np


def normalize_title(t: str | None) -> str:
    """Lower-case; keep letters, combining marks (Indic vowel signs!) and digits; everything else is a space."""
    out = []
    for ch in unicodedata.normalize("NFC", (t or "").lower()):
        cat = unicodedata.category(ch)
        out.append(ch if cat[0] in ("L", "M", "N") else " ")
    return " ".join("".join(out).split())[:200]


def duplicate_upload(conn: sqlite3.Connection, video_id: str, title: str | None, duration_s: float | None) -> str | None:
    """Return the id of an earlier item with the same title and length (already downloaded or beyond), else None."""
    nt = normalize_title(title)
    if not nt or not duration_s:
        return None
    r = conn.execute(
        "SELECT id FROM videos WHERE id != ? AND norm_title = ? AND duration_s BETWEEN ? AND ? "
        "AND status NOT IN ('discovered','downloading','rejected','failed') LIMIT 1",
        (video_id, nt, duration_s - 1.0, duration_s + 1.0)).fetchone()
    return r["id"] if r else None


def fingerprint(x16: np.ndarray, sr: int, bins: int = 64) -> str:
    """64 x 4-bit quantised RMS envelope over the whole recording (duration-normalised) as hex."""
    x = x16.astype(np.float32)
    if x.size < sr:
        return ""
    n = (x.size // bins) * bins
    env = np.sqrt((x[:n].reshape(bins, -1) ** 2).mean(axis=1) + 1e-9)
    env = np.log(env + 1e-6)
    lo, hi = np.percentile(env, 5), np.percentile(env, 95)
    q = np.clip(((env - lo) / max(hi - lo, 1e-6) * 15).round(), 0, 15).astype(np.uint8)
    return "".join(f"{v:x}" for v in q)


def fingerprint_similarity(a: str, b: str) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.frombuffer(bytes.fromhex("".join("0" + c for c in a)), dtype=np.uint8)
    vb = np.frombuffer(bytes.fromhex("".join("0" + c for c in b)), dtype=np.uint8)
    return float((np.abs(va.astype(int) - vb.astype(int)) <= 1).mean())


def duplicate_audio(conn: sqlite3.Connection, video_id: str, fp: str, duration_s: float, threshold: float = 0.9) -> str | None:
    """Return the id of an earlier recording with (almost) the same length and envelope, else None."""
    if not fp or not duration_s:
        return None
    for r in conn.execute("SELECT id, fp FROM videos WHERE id != ? AND fp IS NOT NULL AND duration_s BETWEEN ? AND ?",
                          (video_id, duration_s - 2.0, duration_s + 2.0)):
        if fingerprint_similarity(fp, r["fp"]) >= threshold:
            return r["id"]
    return None


# ----------------------------------------------------------------------------- clip level
CLIP_BINS = 32


def clip_fingerprint(x: np.ndarray, sr: int) -> str:
    """32 x 4-bit quantised RMS envelope of one clip (duration-normalised), as hex."""
    x = np.asarray(x, dtype=np.float32)
    if x.size < sr // 4:
        return ""
    n = (x.size // CLIP_BINS) * CLIP_BINS
    env = np.log(np.sqrt((x[:n].reshape(CLIP_BINS, -1) ** 2).mean(axis=1) + 1e-9) + 1e-6)
    lo, hi = np.percentile(env, 5), np.percentile(env, 95)
    q = np.clip(((env - lo) / max(hi - lo, 1e-6) * 15).round(), 0, 15).astype(np.uint8)
    return "".join(f"{v:x}" for v in q)


def duplicate_clip(conn: sqlite3.Connection, fp: str, dur_ms: int, threshold: float = 0.92, tol_ms: int = 300) -> str | None:
    """Return the id of an already-kept clip (local or on the Hub) with the same length and envelope."""
    if not fp:
        return None
    for r in conn.execute("SELECT chunk_id, fp FROM clip_fps WHERE dur_ms BETWEEN ? AND ?", (dur_ms - tol_ms, dur_ms + tol_ms)):
        if fingerprint_similarity(fp, r["fp"]) >= threshold:
            return r["chunk_id"]
    return None


def remember_clip(conn: sqlite3.Connection, chunk_id: str, fp: str, dur_ms: int, source: str = "local") -> None:
    if fp:
        conn.execute("INSERT OR REPLACE INTO clip_fps(chunk_id, dur_ms, fp, source, created_at) VALUES (?,?,?,?,?)",
                     (chunk_id, dur_ms, fp, source, __import__("time").time()))


def seed_clip_fingerprints_from_hub(cfg) -> int:
    """Download every shard already published and index its clips, so a rebuilt state database still
    refuses to push the same audio twice."""
    import io
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    from . import db as D
    api = HfApi(token=cfg.hf.token)
    files = [f for f in api.list_repo_files(cfg.hf.repo_id, repo_type="dataset") if f.endswith(".parquet")]
    conn = D.connect(cfg.paths.db_path)
    D.init_schema(conn)
    n = 0
    for f in files:
        p = hf_hub_download(cfg.hf.repo_id, f, repo_type="dataset", token=cfg.hf.token)
        t = pq.read_table(p, columns=["id", "audio", "duration_s"])
        with D.tx(conn):
            for row in t.to_pylist():
                if conn.execute("SELECT 1 FROM clip_fps WHERE chunk_id=?", (row["id"],)).fetchone():
                    continue
                x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x[:, 0]
                remember_clip(conn, row["id"], clip_fingerprint(x, sr), int(round(row["duration_s"] * 1000)), source="hub")
                n += 1
    return n
