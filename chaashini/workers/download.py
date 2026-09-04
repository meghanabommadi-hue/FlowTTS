"""Download worker: pulls the original-language audio track of discovered items."""
from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path

from .. import db as D
from ..audio import probe_duration_sr
from ..languages import LANGUAGES
from ..ytsource import RateLimited, SkipVideo, Transient, download, slim_info
from .base import Worker, free_gb

log = logging.getLogger("chaashini.download")

IN_FLIGHT = ("downloaded", "decoding", "diarize_queued", "diarized", "segmenting", "enhance_queued", "enhanced",
             "rescoring", "transcribe_queued", "transcribed", "finalizing")


class DownloadWorker(Worker):
    kind = "download"

    def __init__(self, name: str, cfg=None):
        super().__init__(name, cfg)
        self.stats = {"downloaded": 0, "skipped": 0, "rate_limited": 0, "failed": 0, "bytes": 0, "source_hours": 0.0}
        self.allowed = {l.code for l in self.cfg.enabled_languages()} | {"en"}

    def idle_sleep(self) -> float:
        return 8.0

    def in_flight(self) -> int:
        qs = ",".join("?" * len(IN_FLIGHT))
        return self.conn.execute(f"SELECT COUNT(*) n FROM videos WHERE status IN ({qs})", IN_FLIGHT).fetchone()["n"]

    def step(self) -> bool:
        cd = D.kv_get(self.conn, "source_cooldown_until", 0) or 0
        if cd > time.time():
            self.heartbeat("cooldown", f"{int(cd - time.time())}s left", force=True)
            return False
        fg = free_gb(self.cfg.paths.data_dir)
        if fg < self.cfg.storage.min_free_gb:
            self.heartbeat("paused: low disk", f"{fg:.0f} GB free", force=True)
            return False
        n_inflight = self.in_flight()
        if n_inflight >= self.cfg.workers.max_videos_in_flight:
            self.heartbeat("paused: pipeline full", f"{n_inflight} in flight", force=True)
            return False
        langs = [l for l in self.cfg.enabled_languages() if l.weight > 0]
        v = None
        if langs:
            pick = random.choices(langs, weights=[l.weight for l in langs], k=1)[0].code
            v = D.claim_video(self.conn, "discovered", "downloading", self.name, self.cfg.workers.lease_s, "AND lang_hint=?", (pick,))
        if not v:
            v = D.claim_video(self.conn, "discovered", "downloading", self.name, self.cfg.workers.lease_s)
        if not v:
            return False
        vid = v["id"]
        self.heartbeat("downloading", f"{vid} [{v['lang_hint']}] {(v['title'] or '')[:50]}", force=True)
        out_dir = self.cfg.paths.work_dir / vid
        t0 = time.time()
        try:
            dl = download(self.cfg.source, vid, out_dir, allowed_langs=self.allowed)
        except SkipVideo as e:
            self.stats["skipped"] += 1
            D.set_video_status(self.conn, vid, "rejected", error=f"source: {str(e)[:200]}")
            log.info("skip %s: %s", vid, str(e)[:120])
            _rm(out_dir)
            return True
        except RateLimited as e:
            self.stats["rate_limited"] += 1
            lvl = int(D.kv_get(self.conn, "source_cooldown_level", 0) or 0) + 1
            cds = min(self.cfg.source.cooldown_max_s, self.cfg.source.cooldown_base_s * 2 ** (lvl - 1))
            D.kv_set(self.conn, "source_cooldown_until", time.time() + cds)
            D.kv_set(self.conn, "source_cooldown_level", lvl)
            self.conn.execute("UPDATE videos SET status='discovered', attempts=attempts-1, leased_by=NULL, leased_until=NULL, updated_at=? WHERE id=?", (time.time(), vid))
            self.event("source", f"rate limited while downloading {vid}; cooling down {cds}s (level {lvl})", level="warn")
            log.warning("rate limited (%s); cooldown %ds", str(e)[:100], cds)
            _rm(out_dir)
            return True
        except Transient as e:
            self.stats["failed"] += 1
            if v["attempts"] >= self.cfg.source.max_attempts:
                D.set_video_status(self.conn, vid, "failed", error=f"download: {str(e)[:200]}")
            else:
                D.set_video_status(self.conn, vid, "discovered", error=f"download retry: {str(e)[:200]}")
                self.conn.execute("UPDATE videos SET attempts=? WHERE id=?", (v["attempts"], vid))
            log.warning("transient failure %s: %s", vid, str(e)[:160])
            _rm(out_dir)
            return True
        try:
            dur, sr = probe_duration_sr(dl.path)
        except Exception as e:  # noqa: BLE001
            D.set_video_status(self.conn, vid, "failed", error=f"probe: {e}")
            _rm(out_dir)
            return True
        if dur < self.cfg.source.min_duration_s:
            D.set_video_status(self.conn, vid, "rejected", error=f"source: too short after download ({dur:.0f}s)")
            _rm(out_dir)
            return True
        size = os.path.getsize(dl.path)
        info = dl.info
        lang_hint = v["lang_hint"]
        declared = (dl.orig_lang or "").split("-")[0].lower()
        if declared and declared in LANGUAGES and declared != lang_hint:
            lang_hint = declared
        D.kv_set(self.conn, "source_cooldown_level", 0)
        D.set_video_status(self.conn, vid, "downloaded", src_path=dl.path, src_sr=sr, duration_s=dur, work_dir=str(out_dir),
                           channel_id=info.get("channel_id") or v["channel_id"], channel=info.get("channel") or v["channel"],
                           title=info.get("title") or v["title"], view_count=info.get("view_count"), upload_date=info.get("upload_date"),
                           categories=",".join(dl.categories), orig_lang=dl.orig_lang, audio_track_lang=dl.audio_track_lang,
                           lang_hint=lang_hint, meta_json=slim_info(info))
        self.stats["downloaded"] += 1
        self.stats["bytes"] += size
        self.stats["source_hours"] = round(self.stats["source_hours"] + dur / 3600, 3)
        log.info("downloaded %s [%s] %.0fs %.1fMB in %.0fs (%s, %s kbps)", vid, lang_hint, dur, size / 1e6, time.time() - t0, dl.ext, dl.abr)
        return True


def _rm(d: Path) -> None:
    import shutil
    shutil.rmtree(d, ignore_errors=True)
