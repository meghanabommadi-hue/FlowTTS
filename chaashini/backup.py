"""State backup/restore: the SQLite database (the 'already seen' set, chunk decisions, shard/push history)
is copied hourly to a PRIVATE Hub repository, and restored on boot when the local database is missing.
A container re-creation on the orchestrator box then costs at most one hour of bookkeeping."""
from __future__ import annotations

import gzip
import logging
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

log = logging.getLogger("chaashini.backup")
DB_FILE = "chaashini.db.gz"


def _api(token: str):
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import HfApi
    return HfApi(token=token)


def backup_db(db_path: Path, token: str, repo_id: str) -> str:
    """Consistent snapshot via the SQLite backup API, gzipped, uploaded as one commit."""
    from huggingface_hub import CommitOperationAdd
    api = _api(token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chaashini-bak-") as td:
        snap = Path(td) / "snap.db"
        src = sqlite3.connect(str(db_path))
        dst = sqlite3.connect(str(snap))
        with dst:
            src.backup(dst)
        src.close(); dst.close()
        gz = Path(td) / DB_FILE
        with open(snap, "rb") as f, gzip.open(gz, "wb", compresslevel=6) as g:
            shutil.copyfileobj(f, g)
        info = api.create_commit(repo_id=repo_id, repo_type="dataset",
                                 operations=[CommitOperationAdd(path_in_repo=DB_FILE, path_or_fileobj=str(gz))],
                                 commit_message=f"state backup {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
        log.info("state backup uploaded (%.1f MB) -> %s", gz.stat().st_size / 1e6, getattr(info, "commit_url", ""))
        return getattr(info, "commit_url", "")


def restore_db_if_missing(db_path: Path, token: str, repo_id: str) -> bool:
    """If there is no usable local database but a backup exists, restore it. Returns True when restored."""
    if db_path.exists() and db_path.stat().st_size > 0:
        try:
            n = sqlite3.connect(str(db_path)).execute("SELECT COUNT(*) FROM videos").fetchone()[0]
            if n > 0:
                return False
        except Exception:  # noqa: BLE001
            pass
    try:
        from huggingface_hub import hf_hub_download
        gz = hf_hub_download(repo_id, DB_FILE, repo_type="dataset", token=token)
    except Exception as e:  # noqa: BLE001
        log.info("no state backup to restore (%s)", str(e)[:120])
        return False
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".restore")
    with gzip.open(gz, "rb") as g, open(tmp, "wb") as f:
        shutil.copyfileobj(g, f)
    for suffix in ("-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()
    os.replace(tmp, db_path)
    n = sqlite3.connect(str(db_path)).execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    log.warning("state database restored from backup (%d sources known)", n)
    return True


def restore_history_from_hub(cfg, conn) -> dict:
    """Rebuild the accepted-clip and shard/push history from the parquet shards already published,
    so totals, per-language hours and quality distributions reflect the real corpus after a state loss.
    Idempotent: clips and shards already known are skipped."""
    import json
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from . import db as D
    api = _api(cfg.hf.token)
    info = api.dataset_info(cfg.hf.repo_id)
    files = sorted(f for f in api.list_repo_files(cfg.hf.repo_id, repo_type="dataset") if f.endswith(".parquet"))
    n_clips = n_shards = 0
    total_s = 0.0
    for f in files:
        if conn.execute("SELECT 1 FROM shards WHERE hf_path=?", (f,)).fetchone():
            continue
        p = hf_hub_download(cfg.hf.repo_id, f, repo_type="dataset", token=cfg.hf.token)
        t = pq.read_table(p)
        rows = t.drop(["audio"]).to_pylist()
        lang = f.split("/")[1]
        dur = sum(r["duration_s"] for r in rows)
        size = Path(p).stat().st_size
        with D.tx(conn):
            cur = conn.execute("INSERT INTO shards(lang, path, hf_path, n_chunks, duration_s, size_bytes, status, created_at, pushed_at) VALUES (?,?,?,?,?,?,'pushed',?,?)",
                               (lang, f"hub:{f}", f, len(rows), dur, size, time.time(), time.time()))
            sid = cur.lastrowid
            for r in rows:
                if conn.execute("SELECT 1 FROM chunks WHERE id=?", (r["id"],)).fetchone():
                    continue
                q = {k: r.get(k) for k in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808", "music_prob", "speech_prob", "noise_prob",
                                           "snr_db", "rms_dbfs", "peak_dbfs", "clipping_ratio", "bandwidth_hz", "vad_speech_ratio", "speaker_dominance")}
                if r.get("asr_confidence"):
                    q["asr_conf"] = r["asr_confidence"]
                try:
                    comp = json.loads(r.get("language_mix") or "{}")
                except Exception:  # noqa: BLE001
                    comp = {}
                lid = {"lang": r["language"], "confidence": r.get("language_confidence"), "composition": comp, "script": r.get("script"),
                       "code_mixed": r.get("code_mixed"), "restored": True}
                dur_ms = int(round(r["duration_s"] * 1000))
                try:
                    created = time.mktime(time.strptime(r.get("created_at") or "", "%Y-%m-%dT%H:%M:%SZ")) if r.get("created_at") else time.time()
                except Exception:  # noqa: BLE001
                    created = time.time()
                conn.execute("INSERT INTO chunks(id, video_id, idx, start_ms, end_ms, dur_ms, speaker, status, enhanced, metrics_json, text, lang, lang_conf, lang_json, shard_id, created_at, updated_at) "
                             "VALUES (?,?,?,?,?,?,?,'accepted',?,?,?,?,?,?,?,?,?)",
                             (r["id"], f"hub:{r.get('source_id')}", int(r.get("segment_index") or 0), 0, dur_ms, dur_ms, None, 1 if r.get("enhanced") else 0,
                              D.j(q), r.get("text"), r["language"], r.get("language_confidence"), D.j(lid), sid, created, created))
                n_clips += 1
                total_s += r["duration_s"]
        n_shards += 1
    if n_shards:
        conn.execute("INSERT INTO pushes(started_at, finished_at, status, hours, n_shards, n_chunks, bytes, commit_url) VALUES (?,?,'done',?,?,?,?,?)",
                     (time.time(), time.time(), total_s / 3600, n_shards, n_clips, 0, f"https://huggingface.co/datasets/{cfg.hf.repo_id}/commit/{info.sha}"))
        D.event(conn, "system", f"history restored from the Hub: {n_shards} shards, {n_clips} clips, {total_s / 3600:.2f} h")
    return {"shards": n_shards, "clips": n_clips, "hours": round(total_s / 3600, 3)}
