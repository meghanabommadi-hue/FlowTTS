"""Publish worker: packs staged chunks into parquet shards and pushes to the Hub every
`push_every_hours` of newly accepted audio (or on demand). Also houses housekeeping."""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

from .. import db as D
from ..cardgen import render_card
from ..export import staged_seconds
from ..packer import build_shards
from ..pusher import ensure_repo, upload_shards, verify_present
from .base import Worker

log = logging.getLogger("chaashini.publish")


class PublishWorker(Worker):
    kind = "publish"

    def __init__(self, name: str, cfg=None):
        super().__init__(name, cfg)
        self.stats = {"pushes": 0, "shards": 0, "pushed_hours": 0.0, "failures": 0}
        self._last_check = 0.0
        self._card_done = bool(D.kv_get(self.conn, "card_initialized"))

    def idle_sleep(self) -> float:
        return 30.0

    # ------------------------------------------------------------------ card
    def per_lang_stats(self, extra: list[dict] | None = None) -> tuple[dict, dict]:
        per: dict[str, dict] = {}
        for r in self.conn.execute("SELECT lang, SUM(n_chunks) n, SUM(duration_s) s FROM shards WHERE status='pushed' GROUP BY lang"):
            per[r["lang"]] = {"chunks": int(r["n"] or 0), "seconds": float(r["s"] or 0)}
        for s in extra or []:
            d = per.setdefault(s["lang"], {"chunks": 0, "seconds": 0.0})
            d["chunks"] += s["n_chunks"]
            d["seconds"] += s["duration_s"]
        for l in per:
            r = self.conn.execute("SELECT AVG(json_extract(metrics_json,'$.dnsmos_ovrl')) a FROM chunks WHERE status='accepted' AND lang=?", (l,)).fetchone()
            per[l]["avg_ovrl"] = float(r["a"] or 0.0)
        tot = {"chunks": sum(v["chunks"] for v in per.values()), "seconds": sum(v["seconds"] for v in per.values())}
        return per, tot

    def card(self, extra: list[dict] | None = None) -> str:
        per, tot = self.per_lang_stats(extra)
        return render_card(self.cfg.hf.repo_id, self.cfg.hf.dataset_name, per, tot, self.cfg.audio.export_sr)

    # ------------------------------------------------------------------ push
    def due(self) -> tuple[bool, float]:
        staged = sum(staged_seconds(self.cfg.paths.staging_dir).values())
        built = self.conn.execute("SELECT COALESCE(SUM(duration_s),0) s FROM shards WHERE status='built'").fetchone()["s"]
        total = staged + built
        forced = bool(D.kv_get(self.conn, "force_push"))
        return (total >= self.cfg.hf.push_every_hours * 3600) or (forced and total > 0), total

    def build_all(self) -> list[dict]:
        out = []
        langs = [p.name for p in self.cfg.paths.staging_dir.iterdir() if p.is_dir()] if self.cfg.paths.staging_dir.exists() else []
        for lang in sorted(langs):
            r = self.conn.execute("SELECT COALESCE(MAX(CAST(substr(path, -13, 5) AS INTEGER)), -1) i FROM shards WHERE lang=?", (lang,)).fetchone()
            nxt = int(r["i"]) + 1
            shards = build_shards(self.cfg.paths.staging_dir, self.cfg.paths.shards_dir, lang, nxt, self.cfg.hf.shard_target_mb,
                                  name=self.cfg.hf.dataset_name.lower(), min_seconds=1.0)
            for s in shards:
                cur = self.conn.execute("INSERT INTO shards(lang, path, hf_path, n_chunks, duration_s, size_bytes, status, created_at) VALUES (?,?,?,?,?,?,'built',?)",
                                        (s["lang"], s["path"], s["hf_path"], s["n_chunks"], s["duration_s"], s["size_bytes"], time.time()))
                s["id"] = cur.lastrowid
                self.stats["shards"] += 1
                out.append(s)
        return out

    def push_built(self) -> bool:
        rows = self.conn.execute("SELECT * FROM shards WHERE status='built' ORDER BY id").fetchall()
        shards = [dict(r) for r in rows if os.path.exists(r["path"])]
        for r in rows:
            if not os.path.exists(r["path"]):
                self.conn.execute("UPDATE shards SET status='failed' WHERE id=?", (r["id"],))
        if not shards:
            return False
        hours = sum(s["duration_s"] for s in shards) / 3600
        nbytes = sum(s["size_bytes"] for s in shards)
        cur = self.conn.execute("INSERT INTO pushes(started_at, status, hours, n_shards, n_chunks, bytes) VALUES (?,?,?,?,?,?)",
                                (time.time(), "running", hours, len(shards), sum(s["n_chunks"] for s in shards), nbytes))
        push_id = cur.lastrowid
        self.heartbeat("pushing", f"{len(shards)} shards, {hours:.2f} h, {nbytes / 1e9:.2f} GB", force=True)
        self.event("push", f"pushing {len(shards)} shards ({hours:.2f} h, {nbytes / 1e9:.2f} GB) to {self.cfg.hf.repo_id}")
        t0 = time.time()
        try:
            readme = self.card(extra=shards)
            url = upload_shards(self.cfg.hf.token, self.cfg.hf.repo_id, shards, readme, self.cfg.hf.max_retries,
                                commit_message=f"add {len(shards)} shards (+{hours:.2f} h)")
            if not verify_present(self.cfg.hf.token, self.cfg.hf.repo_id, [s["hf_path"] for s in shards]):
                raise RuntimeError("uploaded files not visible in repo listing")
        except Exception as e:  # noqa: BLE001
            self.stats["failures"] += 1
            self.conn.execute("UPDATE pushes SET finished_at=?, status='failed', error=? WHERE id=?", (time.time(), str(e)[:1000], push_id))
            self.event("push", f"push failed: {str(e)[:200]}", level="error")
            log.error("push failed: %s", e)
            return True
        with D.tx(self.conn):
            for s in shards:
                self.conn.execute("UPDATE shards SET status='pushed', push_id=?, pushed_at=? WHERE id=?", (push_id, time.time(), s["id"]))
                self.conn.execute("UPDATE chunks SET shard_id=? WHERE status='accepted' AND shard_id IS NULL AND lang=? AND staged_path LIKE ?",
                                  (s["id"], s["lang"], f"%/{s['lang']}/%"))
            self.conn.execute("UPDATE pushes SET finished_at=?, status='done', commit_url=? WHERE id=?", (time.time(), url, push_id))
        D.kv_set(self.conn, "force_push", False)
        for s in shards:
            try:
                os.remove(s["path"])
            except OSError:
                pass
        self.stats["pushes"] += 1
        self.stats["pushed_hours"] = round(self.stats["pushed_hours"] + hours, 3)
        self.event("push", f"pushed {len(shards)} shards (+{hours:.2f} h) in {time.time() - t0:.0f}s: {url}", data={"url": url})
        log.info("pushed %d shards (%.2f h) in %.0fs -> %s", len(shards), hours, time.time() - t0, url)
        return True

    # ------------------------------------------------------------------ housekeeping
    def housekeeping(self) -> None:
        # stale work dirs of terminal videos
        ttl = self.cfg.storage.work_ttl_hours * 3600
        wd = self.cfg.paths.work_dir
        if wd.exists():
            for d in wd.iterdir():
                if not d.is_dir():
                    continue
                r = self.conn.execute("SELECT status, updated_at FROM videos WHERE id=?", (d.name,)).fetchone()
                try:
                    age = time.time() - d.stat().st_mtime
                except OSError:
                    continue
                if (r is None and age > 3600) or (r and r["status"] in D.TERMINAL) or (r and age > ttl):
                    shutil.rmtree(d, ignore_errors=True)
        # library caches that could grow silently (datasets arrow cache, yt-dlp cache, stale HF downloads)
        for cache in (Path.home() / ".cache" / "huggingface" / "datasets", Path.home() / ".cache" / "yt-dlp", Path("/tmp")):
            try:
                for p in cache.glob("*"):
                    if (p.name.startswith("chaashini") or cache.name in ("datasets", "yt-dlp")) and time.time() - p.stat().st_mtime > 86400:
                        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
        # stdout capture files of the watchdog: keep them bounded
        for out in self.cfg.paths.logs_dir.glob("*.out"):
            try:
                if out.stat().st_size > 200 << 20:
                    with open(out, "rb") as f:
                        f.seek(-(50 << 20), 2)
                        tail = f.read()
                    with open(out, "wb") as f:
                        f.write(tail)
            except OSError:
                pass
        # orphaned staging sidecars / audio (partner file missing)
        if self.cfg.paths.staging_dir.exists():
            for lang_dir in self.cfg.paths.staging_dir.iterdir():
                if not lang_dir.is_dir():
                    continue
                for p in lang_dir.glob("*.tmp"):
                    if time.time() - p.stat().st_mtime > 3600:
                        p.unlink(missing_ok=True)
        # discovered backlog hygiene: drop very old undownloaded items so the queue stays fresh
        self.conn.execute("DELETE FROM videos WHERE status='discovered' AND created_at < ?", (time.time() - 14 * 86400,))
        released = D.release_stale(self.conn)
        if released:
            log.info("janitor released: %s", released)

    def step(self) -> bool:
        if not self._card_done and self.cfg.hf.token:
            try:
                ensure_repo(self.cfg.hf.token, self.cfg.hf.repo_id)
                from ..pusher import update_card
                update_card(self.cfg.hf.token, self.cfg.hf.repo_id, self.card())
                D.kv_set(self.conn, "card_initialized", True)
                self._card_done = True
                self.event("push", f"dataset card initialised on {self.cfg.hf.repo_id}")
            except Exception as e:  # noqa: BLE001
                log.warning("card init failed: %s", e)
        now = time.time()
        if now - self._last_check < self.cfg.hf.push_check_interval_s and not D.kv_get(self.conn, "force_push"):
            return False
        self._last_check = now
        self.housekeeping()
        due, total = self.due()
        self.heartbeat("watching", f"{total / 3600:.2f} h staged / {self.cfg.hf.push_every_hours} h", force=True)
        if not due:
            return False
        if not self.cfg.hf.token:
            log.error("HF token missing; cannot push")
            return False
        self.build_all()
        return self.push_built()
