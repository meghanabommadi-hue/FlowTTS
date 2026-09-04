"""SQLite (WAL) state store shared by every worker on the orchestrator box.

One connection per worker/thread; short explicit transactions; `BEGIN IMMEDIATE` for
claim operations so two workers never take the same row.  Leases make crashes harmless:
a row claimed by a dead worker is released by the janitor once `leased_until` passes.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
  id INTEGER PRIMARY KEY,
  lang TEXT NOT NULL,
  query TEXT NOT NULL,
  genre TEXT,
  source TEXT NOT NULL DEFAULT 'llm',
  created_at REAL NOT NULL,
  last_run_at REAL,
  runs INTEGER NOT NULL DEFAULT 0,
  videos_found INTEGER NOT NULL DEFAULT 0,
  videos_new INTEGER NOT NULL DEFAULT 0,
  accepted_sec REAL NOT NULL DEFAULT 0,
  UNIQUE(lang, query)
);
CREATE TABLE IF NOT EXISTS videos (
  id TEXT PRIMARY KEY,
  source_hash TEXT NOT NULL,
  lang_hint TEXT,
  query_id INTEGER,
  channel_id TEXT,
  channel TEXT,
  title TEXT,
  duration_s REAL,
  view_count INTEGER,
  upload_date TEXT,
  categories TEXT,
  orig_lang TEXT,
  audio_track_lang TEXT,
  status TEXT NOT NULL DEFAULT 'discovered',
  stage_entered_at REAL,
  attempts INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  leased_by TEXT,
  leased_until REAL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  src_path TEXT,
  src_sr INTEGER,
  work_dir TEXT,
  meta_json TEXT,
  stats_json TEXT,
  priority INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_updated ON videos(updated_at);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  dur_ms INTEGER NOT NULL,
  speaker INTEGER,
  status TEXT NOT NULL,             -- candidate | enhance | accepted | rejected
  reject_reason TEXT,
  enhanced INTEGER NOT NULL DEFAULT 0,
  metrics_json TEXT,
  text TEXT,
  lang TEXT,
  lang_conf REAL,
  lang_json TEXT,
  staged_path TEXT,
  shard_id INTEGER,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status, lang);
CREATE INDEX IF NOT EXISTS idx_chunks_shard ON chunks(shard_id);
CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at);
CREATE TABLE IF NOT EXISTS gpu_jobs (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,               -- diarize | transcribe | enhance
  video_id TEXT NOT NULL,
  payload_path TEXT NOT NULL,
  result_path TEXT,
  status TEXT NOT NULL DEFAULT 'queued',   -- queued | running | done | failed
  worker TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  created_at REAL NOT NULL,
  started_at REAL,
  finished_at REAL,
  leased_until REAL,
  payload_bytes INTEGER,
  audio_seconds REAL,
  n_items INTEGER,
  proc_seconds REAL
);
CREATE INDEX IF NOT EXISTS idx_gpu_jobs_status ON gpu_jobs(status, kind, created_at);
CREATE INDEX IF NOT EXISTS idx_gpu_jobs_video ON gpu_jobs(video_id);
CREATE TABLE IF NOT EXISTS shards (
  id INTEGER PRIMARY KEY,
  lang TEXT NOT NULL,
  path TEXT NOT NULL,
  hf_path TEXT,
  n_chunks INTEGER NOT NULL,
  duration_s REAL NOT NULL,
  size_bytes INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'built',    -- built | pushed | failed
  push_id INTEGER,
  created_at REAL NOT NULL,
  pushed_at REAL
);
CREATE TABLE IF NOT EXISTS pushes (
  id INTEGER PRIMARY KEY,
  started_at REAL NOT NULL,
  finished_at REAL,
  status TEXT NOT NULL,                    -- running | done | failed
  hours REAL,
  n_shards INTEGER,
  n_chunks INTEGER,
  bytes INTEGER,
  commit_url TEXT,
  error TEXT
);
CREATE TABLE IF NOT EXISTS workers (
  name TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  host TEXT,
  pid INTEGER,
  started_at REAL NOT NULL,
  heartbeat_at REAL NOT NULL,
  state TEXT,
  current TEXT,
  stats_json TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  level TEXT NOT NULL,
  kind TEXT NOT NULL,
  msg TEXT NOT NULL,
  data_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS metrics (
  ts REAL PRIMARY KEY,
  data_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
  k TEXT PRIMARY KEY,
  v TEXT
);
CREATE TABLE IF NOT EXISTS channels (
  id TEXT PRIMARY KEY,
  name TEXT,
  lang_hint TEXT,
  videos_seen INTEGER NOT NULL DEFAULT 0,
  videos_done INTEGER NOT NULL DEFAULT 0,
  accepted_sec REAL NOT NULL DEFAULT 0,
  source_sec REAL NOT NULL DEFAULT 0,
  expanded_at REAL,
  blocked INTEGER NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL
);
"""

# Video pipeline states, in order.  *_queued states wait for a GPU job.
VIDEO_STATES = [
    "discovered", "downloading", "downloaded", "decoding", "diarize_queued", "diarized",
    "segmenting", "enhance_queued", "enhanced", "transcribe_queued", "transcribed",
    "finalizing", "done", "rejected", "failed",
]
# Working state -> state to fall back to when a lease expires (crash recovery).
LEASE_FALLBACK = {
    "downloading": "discovered",
    "decoding": "downloaded",
    "segmenting": "diarized",
    "rescoring": "enhanced",
    "finalizing": "transcribed",
}
TERMINAL = {"done", "rejected", "failed"}


def connect(db_path: str | os.PathLike) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-65536")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(videos)")}
    if "fp" not in cols:
        conn.execute("ALTER TABLE videos ADD COLUMN fp TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_dur ON videos(duration_s)")
    if "norm_title" not in cols:
        conn.execute("ALTER TABLE videos ADD COLUMN norm_title TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_norm_title ON videos(norm_title)")
    ccols = {r["name"] for r in conn.execute("PRAGMA table_info(channels)")}
    if "india_verdict" not in ccols:
        conn.execute("ALTER TABLE channels ADD COLUMN india_verdict TEXT")
        conn.execute("ALTER TABLE channels ADD COLUMN india_conf REAL")


@contextmanager
def tx(conn: sqlite3.Connection, immediate: bool = True):
    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def now() -> float:
    return time.time()


def j(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def uj(s: str | None, default=None):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


# ----------------------------------------------------------------------------- events / kv
def event(conn: sqlite3.Connection, kind: str, msg: str, level: str = "info", data: dict | None = None) -> None:
    conn.execute("INSERT INTO events(ts, level, kind, msg, data_json) VALUES (?,?,?,?,?)",
                 (now(), level, kind, msg, j(data) if data else None))


def kv_get(conn: sqlite3.Connection, k: str, default=None):
    r = conn.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return uj(r["v"], default) if r else default


def kv_set(conn: sqlite3.Connection, k: str, v: Any) -> None:
    conn.execute("INSERT INTO kv(k, v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, j(v)))


# ----------------------------------------------------------------------------- videos
def claim_video(conn: sqlite3.Connection, from_status: str, to_status: str, worker: str,
                lease_s: int, extra_where: str = "", params: Iterable = ()) -> sqlite3.Row | None:
    with tx(conn):
        row = conn.execute(
            f"SELECT * FROM videos WHERE status=? AND (leased_until IS NULL OR leased_until < ?) {extra_where} "
            f"ORDER BY priority DESC, created_at ASC LIMIT 1",
            (from_status, now(), *params)).fetchone()
        if not row:
            return None
        t = now()
        conn.execute("UPDATE videos SET status=?, leased_by=?, leased_until=?, attempts=attempts+1, "
                     "stage_entered_at=?, updated_at=? WHERE id=?",
                     (to_status, worker, t + lease_s, t, t, row["id"]))
        return conn.execute("SELECT * FROM videos WHERE id=?", (row["id"],)).fetchone()


def set_video_status(conn: sqlite3.Connection, video_id: str, status: str, error: str | None = None,
                     **cols: Any) -> None:
    t = now()
    sets = ["status=?", "updated_at=?", "stage_entered_at=?", "leased_by=NULL", "leased_until=NULL", "error=?"]
    vals: list[Any] = [status, t, t, error]
    if status != "failed" and status not in ("downloading", "decoding", "segmenting", "finalizing"):
        sets.append("attempts=0")
    for k, v in cols.items():
        sets.append(f"{k}=?")
        vals.append(j(v) if k.endswith("_json") and not isinstance(v, str) else v)
    vals.append(video_id)
    conn.execute(f"UPDATE videos SET {', '.join(sets)} WHERE id=?", vals)


def touch_lease(conn: sqlite3.Connection, video_id: str, lease_s: int) -> None:
    conn.execute("UPDATE videos SET leased_until=?, updated_at=? WHERE id=?", (now() + lease_s, now(), video_id))


def count_by_status(conn: sqlite3.Connection) -> dict[str, int]:
    return {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) n FROM videos GROUP BY status")}


# ----------------------------------------------------------------------------- gpu jobs
def enqueue_job(conn: sqlite3.Connection, kind: str, video_id: str, payload_path: str,
                payload_bytes: int | None = None, audio_seconds: float | None = None, n_items: int | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO gpu_jobs(kind, video_id, payload_path, status, created_at, payload_bytes, audio_seconds, n_items) "
        "VALUES (?,?,?,'queued',?,?,?,?)", (kind, video_id, payload_path, now(), payload_bytes, audio_seconds, n_items))
    return int(cur.lastrowid)


def claim_job(conn: sqlite3.Connection, kinds: list[str], worker: str, lease_s: int) -> sqlite3.Row | None:
    """Claim the oldest queued job of any of `kinds`, preferring the kind with the longest queue."""
    with tx(conn):
        qs = ",".join("?" * len(kinds))
        t = now()
        row = conn.execute(
            f"SELECT * FROM gpu_jobs WHERE kind IN ({qs}) AND (status='queued' OR (status='running' AND leased_until < ?)) "
            f"ORDER BY created_at ASC LIMIT 1", (*kinds, t)).fetchone()
        if not row:
            return None
        conn.execute("UPDATE gpu_jobs SET status='running', worker=?, attempts=attempts+1, started_at=?, leased_until=? WHERE id=?",
                     (worker, t, t + lease_s, row["id"]))
        return conn.execute("SELECT * FROM gpu_jobs WHERE id=?", (row["id"],)).fetchone()


def finish_job(conn: sqlite3.Connection, job_id: int, ok: bool, result_path: str | None = None,
               error: str | None = None, proc_seconds: float | None = None) -> None:
    conn.execute("UPDATE gpu_jobs SET status=?, result_path=?, error=?, finished_at=?, proc_seconds=?, leased_until=NULL WHERE id=?",
                 ("done" if ok else "failed", result_path, error, now(), proc_seconds, job_id))


# ----------------------------------------------------------------------------- workers
def heartbeat(conn: sqlite3.Connection, name: str, kind: str, state: str, current: str | None = None,
              stats: dict | None = None, host: str | None = None, pid: int | None = None) -> None:
    t = now()
    conn.execute(
        "INSERT INTO workers(name, kind, host, pid, started_at, heartbeat_at, state, current, stats_json) VALUES (?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(name) DO UPDATE SET heartbeat_at=excluded.heartbeat_at, state=excluded.state, current=excluded.current, "
        "stats_json=COALESCE(excluded.stats_json, workers.stats_json), pid=COALESCE(excluded.pid, workers.pid), host=COALESCE(excluded.host, workers.host)",
        (name, kind, host, pid, t, t, state, current, j(stats) if stats is not None else None))


# ----------------------------------------------------------------------------- janitor
def release_all_video_leases(conn: sqlite3.Connection) -> dict[str, int]:
    """Used at supervisor start: every CPU worker restarted, so any working-state row is orphaned."""
    out: dict[str, int] = {}
    t = now()
    with tx(conn):
        for working, back in LEASE_FALLBACK.items():
            cur = conn.execute("UPDATE videos SET status=?, leased_by=NULL, leased_until=NULL, updated_at=? WHERE status=?", (back, t, working))
            if cur.rowcount:
                out[working] = cur.rowcount
    return out


def requeue_worker_jobs(conn: sqlite3.Connection, worker: str) -> int:
    """A GPU worker (re)started: whatever it had running is gone; make it claimable again."""
    cur = conn.execute("UPDATE gpu_jobs SET status='queued', worker=NULL, leased_until=NULL, started_at=NULL, attempts=MAX(0, attempts-1) "
                       "WHERE status='running' AND worker=?", (worker,))
    return cur.rowcount


def release_stale(conn: sqlite3.Connection) -> dict[str, int]:
    """Return rows whose lease expired to their previous stage; fail rows over the attempt budget."""
    out: dict[str, int] = {}
    t = now()
    with tx(conn):
        for working, back in LEASE_FALLBACK.items():
            cur = conn.execute(
                "UPDATE videos SET status=?, leased_by=NULL, leased_until=NULL, updated_at=?, error=COALESCE(error,'lease expired') "
                "WHERE status=? AND leased_until IS NOT NULL AND leased_until < ?", (back, t, working, t))
            if cur.rowcount:
                out[working] = cur.rowcount
        cur = conn.execute("UPDATE videos SET status='failed', error='attempt budget exhausted', updated_at=? "
                           "WHERE status NOT IN ('done','rejected','failed') AND attempts > 8", (t,))
        if cur.rowcount:
            out["failed_attempts"] = cur.rowcount
    return out
