"""Download worker: pulls the original-language audio track of discovered items."""
from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path

from .. import db as D
from ..audio import probe_duration_sr
from ..dedup import duplicate_upload, normalize_title
from ..llm import LLM, judge_indian_english
from ..languages import LANGUAGES
from ..ytsource import RateLimited, SkipVideo, Transient, cookie_files, download, identity, slim_info
from .base import Worker, free_gb

log = logging.getLogger("chaashini.download")

IN_FLIGHT = ("downloaded", "decoding", "diarize_queued", "diarized", "segmenting", "enhance_queued", "enhanced",
             "rescoring", "transcribe_queued", "transcribed", "finalizing")


class DownloadWorker(Worker):
    kind = "download"

    def __init__(self, name: str, cfg=None):
        super().__init__(name, cfg)
        self.stats = {"downloaded": 0, "skipped": 0, "rate_limited": 0, "failed": 0, "bytes": 0, "source_hours": 0.0, "identity": ""}
        self.allowed = {l.code for l in self.cfg.enabled_languages()} | {"en"}
        try:
            self.slot = int(name.rsplit("-", 1)[-1])
        except ValueError:
            self.slot = 0
        self.rotation = 0
        self.llm = LLM(self.cfg.llm)

    def indian_english_check(self, v):
        """Pre-download gate for English items: the creator must be Indian (LLM verdict, cached per channel)."""
        def check(info: dict):
            declared = (info.get("language") or "").split("-")[0].lower()
            if v["lang_hint"] != "en" and declared != "en":
                return None                      # not an English video: the LID/consensus gates handle it later
            ch = info.get("channel_id") or v["channel_id"]
            if ch:
                r = self.conn.execute("SELECT india_verdict, india_conf FROM channels WHERE id=?", (ch,)).fetchone()
                if r and r["india_verdict"] == "yes" and (r["india_conf"] or 0) >= 0.7:
                    return None
                if r and r["india_verdict"] == "no":
                    return f"not Indian English (channel verdict: no)"
            ok, conf, why = judge_indian_english(self.llm, info.get("title") or v["title"] or "", info.get("channel") or v["channel"] or "",
                                                 info.get("description") or "", info.get("tags") or [])
            verdict = "yes" if ok and conf >= 0.7 else ("no" if (not ok and conf >= 0.7) else "unsure")
            if ch:
                self.conn.execute("INSERT INTO channels(id, name, lang_hint, updated_at, india_verdict, india_conf) VALUES (?,?,?,?,?,?) "
                                  "ON CONFLICT(id) DO UPDATE SET india_verdict=excluded.india_verdict, india_conf=excluded.india_conf, updated_at=excluded.updated_at",
                                  (ch, info.get("channel") or v["channel"], "en", time.time(), verdict, conf))
            if verdict != "yes":
                return f"not Indian English ({verdict}, {conf:.2f}: {why[:80]})"
            return None
        return check

    def n_identities(self) -> int:
        proxies = [p for p in self.cfg.source.proxies if p] or ([self.cfg.source.proxy] if self.cfg.source.proxy else [])
        return max(len(cookie_files(self.cfg.source)), len(proxies), 1)

    def pick_identity(self) -> tuple[str | None, str | None, str, int] | None:
        """Cookie/proxy pair for this worker: its own slot first, rotating past identities that are cooling down."""
        n = self.n_identities()
        now = time.time()
        for k in range(n):
            idx = (self.slot + self.rotation + k) % n
            ck, px, label = identity(self.cfg.source, idx)
            if (D.kv_get(self.conn, f"cooldown:{label}", 0) or 0) > now:
                continue
            return ck, px, label, idx
        return None

    def throttle_identity(self, label: str, idx: int, msg: str) -> None:
        """Cool one identity down (exponential); only when every identity is cooling does the whole source pause."""
        lvl = int(D.kv_get(self.conn, f"cooldown_level:{label}", 0) or 0) + 1
        cds = min(self.cfg.source.cooldown_max_s, self.cfg.source.cooldown_base_s * 2 ** (lvl - 1))
        D.kv_set(self.conn, f"cooldown:{label}", time.time() + cds)
        D.kv_set(self.conn, f"cooldown_level:{label}", lvl)
        self.rotation += 1
        self.event("source", f"{label} throttled ({msg[:100]}); cooling {cds}s (level {lvl})", level="warn")
        if self.pick_identity() is None:
            glvl = int(D.kv_get(self.conn, "source_cooldown_level", 0) or 0) + 1
            gcd = min(self.cfg.source.cooldown_max_s, self.cfg.source.cooldown_base_s * 2 ** (glvl - 1))
            D.kv_set(self.conn, "source_cooldown_until", time.time() + gcd)
            D.kv_set(self.conn, "source_cooldown_level", glvl)
            self.event("source", f"all identities throttled; global cooldown {gcd}s", level="warn")

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
        # weight by the configured share, then boost languages that are still short of audio, so the
        # download mix follows the corpus deficit rather than whatever discovery happened to queue
        langs = [l for l in self.cfg.enabled_languages() if l.weight > 0]
        v = None
        if langs:
            have = {r["lang"]: (r["s"] or 0) / 3600.0 for r in self.conn.execute(
                "SELECT lang, SUM(dur_ms)/1000.0 s FROM chunks WHERE status='accepted' GROUP BY lang")}
            lr = self.cfg.discovery.low_resource_hours
            weights = [l.weight * (2.5 if (l.weight >= 1.0 and have.get(l.code, 0.0) < lr) else 1.0) for l in langs]
            for pick in {random.choices(langs, weights=weights, k=1)[0].code for _ in range(3)}:
                v = D.claim_video(self.conn, "discovered", "downloading", self.name, self.cfg.workers.lease_s, "AND lang_hint=?", (pick,))
                if v:
                    break
        if not v and langs:
            codes = [l.code for l in langs]
            qs = ",".join("?" * len(codes))
            v = D.claim_video(self.conn, "discovered", "downloading", self.name, self.cfg.workers.lease_s,
                              f"AND lang_hint IN ({qs})", codes)
        if not v:
            return False
        vid = v["id"]
        ident = self.pick_identity()
        if ident is None:
            self.conn.execute("UPDATE videos SET status='discovered', attempts=attempts-1, leased_by=NULL, leased_until=NULL, updated_at=? WHERE id=?", (time.time(), vid))
            self.heartbeat("cooldown", "all identities cooling", force=True)
            return False
        ck, px, label, idx = ident
        self.stats["identity"] = label
        dup = duplicate_upload(self.conn, vid, v["title"], v["duration_s"])
        if dup:
            D.set_video_status(self.conn, vid, "rejected", error=f"duplicate_upload of {dup}")
            self.stats["skipped"] += 1
            log.info("skip %s: duplicate upload of %s", vid, dup)
            return True
        self.heartbeat("downloading", f"{vid} [{v['lang_hint']}] {(v['title'] or '')[:50]} via {label}", force=True)
        out_dir = self.cfg.paths.work_dir / vid
        t0 = time.time()
        try:
            extra = self.indian_english_check(v)   # applies itself only to English (declared or discovered) videos

            def _progress(d):
                # a large download must not look like a dead worker: report while bytes are moving
                if d.get("status") == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    got = d.get("downloaded_bytes") or 0
                    pct = f"{100 * got / total:.0f}%" if total else f"{got / 1e6:.0f} MB"
                    self.heartbeat("downloading", f"{vid} [{v['lang_hint']}] {pct} {(v['title'] or '')[:40]} via {label}")

            dl = download(self.cfg.source, vid, out_dir, allowed_langs=self.allowed, cookies_file=ck, proxy=px,
                          extra_check=extra, progress=_progress)
        except SkipVideo as e:
            self.stats["skipped"] += 1
            D.set_video_status(self.conn, vid, "rejected", error=f"source: {str(e)[:200]}")
            log.info("skip %s: %s", vid, str(e)[:120])
            _rm(out_dir)
            return True
        except RateLimited as e:
            self.stats["rate_limited"] += 1
            self.throttle_identity(label, idx, str(e))
            self.conn.execute("UPDATE videos SET status='discovered', attempts=attempts-1, leased_by=NULL, leased_until=NULL, updated_at=? WHERE id=?", (time.time(), vid))
            log.warning("rate limited on %s (%s)", label, str(e)[:100])
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
        enabled = {l.code for l in self.cfg.enabled_languages()}
        if declared and declared in LANGUAGES and declared != lang_hint:
            if declared in enabled:
                lang_hint = declared
            else:
                # the source declares a language we deliberately do not collect (unsupported by the recogniser)
                D.set_video_status(self.conn, vid, "rejected", error=f"declared language not enabled: {declared}")
                self.stats["skipped"] += 1
                _rm(out_dir)
                return True
        D.kv_set(self.conn, "source_cooldown_level", 0)
        D.kv_set(self.conn, f"cooldown_level:{label}", 0)
        dup = duplicate_upload(self.conn, vid, info.get("title") or v["title"], dur)
        if dup:
            D.set_video_status(self.conn, vid, "rejected", error=f"duplicate_upload of {dup}")
            _rm(out_dir)
            return True
        D.set_video_status(self.conn, vid, "downloaded", src_path=dl.path, src_sr=sr, duration_s=dur, work_dir=str(out_dir),
                           norm_title=normalize_title(info.get("title") or v["title"]),
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
