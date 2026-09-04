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
