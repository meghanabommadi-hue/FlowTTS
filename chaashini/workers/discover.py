"""Discovery worker: keeps the download backlog full.

Sources, in priority order:
1. Channel expansion - channels whose processed videos yielded well are crawled fully.
2. Seed playlists/channels from the config.
3. LLM-generated search queries per language (weighted by config, informed by past yield).
"""
from __future__ import annotations

import logging
import random
import secrets
import time

from .. import db as D
from ..languages import LANGUAGES
from ..llm import LLM, generate_queries
from ..ytsource import RateLimited, SkipVideo, Transient, channel_videos, playlist_videos, search, source_hash
from .base import Worker

log = logging.getLogger("chaashini.discover")


class DiscoverWorker(Worker):
    kind = "discover"

    def __init__(self, name: str, cfg=None):
        super().__init__(name, cfg)
        self.llm = LLM(self.cfg.llm)
        self.salt = D.kv_get(self.conn, "source_salt")
        if not self.salt:
            self.salt = secrets.token_hex(16)
            D.kv_set(self.conn, "source_salt", self.salt)
        self.stats = {"rounds": 0, "queries_run": 0, "videos_new": 0, "channels_expanded": 0, "llm_calls": 0}

    def idle_sleep(self) -> float:
        return float(self.cfg.discovery.round_sleep_s)

    def backlog(self) -> int:
        return self.conn.execute("SELECT COUNT(*) n FROM videos WHERE status='discovered'").fetchone()["n"]

    def backlog_by_lang(self) -> dict[str, int]:
        return {r["lang_hint"]: r["n"] for r in self.conn.execute("SELECT lang_hint, COUNT(*) n FROM videos WHERE status='discovered' GROUP BY lang_hint")}

    def cooldown_active(self) -> bool:
        return (D.kv_get(self.conn, "source_cooldown_until", 0) or 0) > time.time()

    # ------------------------------------------------------------------ insert helpers
    def add_videos(self, found, lang: str, query_id: int | None) -> int:
        n = 0
        per_ch = self.cfg.source.max_videos_per_channel
        for f in found:
            if f.duration is not None and not (self.cfg.source.min_duration_s <= f.duration <= self.cfg.source.max_duration_s):
                continue
            if f.channel_id:
                ch = self.conn.execute("SELECT videos_seen, blocked FROM channels WHERE id=?", (f.channel_id,)).fetchone()
                if ch and (ch["blocked"] or ch["videos_seen"] >= per_ch):
                    continue
            t = time.time()
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO videos(id, source_hash, lang_hint, query_id, channel_id, channel, title, duration_s, view_count, status, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,'discovered',?,?)",
                (f.id, source_hash(f.id, self.salt), lang, query_id, f.channel_id, f.channel, f.title, f.duration, f.view_count, t, t))
            if cur.rowcount:
                n += 1
                if f.channel_id:
                    self.conn.execute(
                        "INSERT INTO channels(id, name, lang_hint, videos_seen, updated_at) VALUES (?,?,?,1,?) "
                        "ON CONFLICT(id) DO UPDATE SET videos_seen=videos_seen+1, name=COALESCE(excluded.name, channels.name), updated_at=excluded.updated_at",
                        (f.channel_id, f.channel, lang, t))
        return n

    # ------------------------------------------------------------------ sources
    def expand_channels(self, budget: int) -> int:
        dc = self.cfg.discovery
        rows = self.conn.execute(
            "SELECT id, name, lang_hint, videos_done, accepted_sec, source_sec FROM channels WHERE expanded_at IS NULL AND blocked=0 AND videos_done >= ? "
            "AND source_sec > 0 AND accepted_sec/source_sec >= ? ORDER BY accepted_sec DESC LIMIT 5",
            (dc.channel_expand_min_videos, dc.channel_expand_min_accept_ratio)).fetchall()
        added = 0
        for ch in rows:
            if added >= budget or self.stop:
                break
            try:
                found = channel_videos(self.cfg.source, ch["id"], dc.channel_expand_max_items)
            except RateLimited as e:
                self.rate_limited(str(e))
                break
            except (SkipVideo, Transient) as e:
                log.info("channel %s expansion skipped: %s", ch["id"], e)
                self.conn.execute("UPDATE channels SET expanded_at=? WHERE id=?", (time.time(), ch["id"]))
                continue
            # temporarily lift the per-channel cap for a proven channel
            self.conn.execute("UPDATE channels SET videos_seen=0, expanded_at=? WHERE id=?", (time.time(), ch["id"]))
            n = self.add_videos(found, ch["lang_hint"] or "hi", None)
            added += n
            self.stats["channels_expanded"] += 1
            self.event("discover", f"expanded channel '{ch['name']}' ({ch['lang_hint']}): +{n} videos", data={"channel": ch["id"]})
            log.info("expanded channel %s (%s): +%d", ch["name"], ch["lang_hint"], n)
        return added

    def run_seeds(self) -> int:
        added = 0
        for url in self.cfg.discovery.seed_channel_playlists:
            key = f"seed_done:{url}"
            if D.kv_get(self.conn, key):
                continue
            lang = "hi"
            if "|" in url:
                lang, url = url.split("|", 1)
            try:
                found = playlist_videos(self.cfg.source, url, 500)
            except RateLimited as e:
                self.rate_limited(str(e))
                return added
            except Exception as e:  # noqa: BLE001
                log.warning("seed %s failed: %s", url, e)
                D.kv_set(self.conn, key, {"error": str(e)[:200], "ts": time.time()})
                continue
            n = self.add_videos(found, lang, None)
            added += n
            D.kv_set(self.conn, key, {"added": n, "ts": time.time()})
            self.event("discover", f"seed playlist ({lang}): +{n} videos")
        return added

    def pick_language(self) -> str:
        deficits = self.language_deficits()
        if deficits:
            return deficits[0][0]
        langs = [l for l in self.cfg.enabled_languages() if LANGUAGES.get(l.code, None) and LANGUAGES[l.code].asr_supported]
        bl = self.backlog_by_lang()
        # weight by config weight and inverse backlog share
        weights = []
        for l in langs:
            have = bl.get(l.code, 0)
            weights.append(l.weight / (1.0 + have / 25.0))
        return random.choices(langs, weights=weights, k=1)[0].code

    def pending_queries(self, lang: str, limit: int) -> list:
        return self.conn.execute("SELECT * FROM queries WHERE lang=? AND runs=0 ORDER BY created_at LIMIT ?", (lang, limit)).fetchall()

    def make_queries(self, lang: str) -> int:
        L = LANGUAGES[lang]
        known = [r["query"] for r in self.conn.execute("SELECT query FROM queries WHERE lang=? ORDER BY created_at DESC LIMIT 80", (lang,))]
        good = [r["query"] for r in self.conn.execute("SELECT query FROM queries WHERE lang=? AND videos_new>0 ORDER BY accepted_sec/(videos_new+1) DESC LIMIT 15", (lang,))]
        bad = [r["query"] for r in self.conn.execute("SELECT query FROM queries WHERE lang=? AND runs>0 AND videos_new>=3 AND accepted_sec < 60 ORDER BY videos_new DESC LIMIT 15", (lang,))]
        self.stats["llm_calls"] += 1
        items = generate_queries(self.llm, L, self.cfg.discovery.queries_per_language_per_round, known, good, bad)
        n = 0
        for it in items:
            cur = self.conn.execute("INSERT OR IGNORE INTO queries(lang, query, genre, source, created_at) VALUES (?,?,?,?,?)",
                                    (lang, it["query"], it.get("genre", ""), "llm", time.time()))
            n += cur.rowcount
        log.info("LLM produced %d new queries for %s", n, lang)
        return n

    def run_query(self, q) -> int:
        try:
            found = search(self.cfg.source, q["query"], self.cfg.source.search_results_per_query)
        except RateLimited as e:
            self.rate_limited(str(e))
            return 0
        except (SkipVideo, Transient) as e:
            log.warning("query '%s' failed: %s", q["query"], e)
            self.conn.execute("UPDATE queries SET runs=runs+1, last_run_at=? WHERE id=?", (time.time(), q["id"]))
            return 0
        n = self.add_videos(found, q["lang"], q["id"])
        self.conn.execute("UPDATE queries SET runs=runs+1, last_run_at=?, videos_found=videos_found+?, videos_new=videos_new+? WHERE id=?",
                          (time.time(), len(found), n, q["id"]))
        self.stats["queries_run"] += 1
        self.stats["videos_new"] += n
        log.info("query [%s] '%s': %d found, %d new", q["lang"], q["query"], len(found), n)
        return n

    def rate_limited(self, msg: str) -> None:
        lvl = int(D.kv_get(self.conn, "source_cooldown_level", 0) or 0) + 1
        cd = min(self.cfg.source.cooldown_max_s, self.cfg.source.cooldown_base_s * 2 ** (lvl - 1))
        D.kv_set(self.conn, "source_cooldown_until", time.time() + cd)
        D.kv_set(self.conn, "source_cooldown_level", lvl)
        self.event("source", f"rate limited / bot check ({msg[:120]}); cooling down {cd}s (level {lvl})", level="warn")
        log.warning("rate limited: cooling down %ds", cd)

    # ------------------------------------------------------------------ main step
    def language_deficits(self) -> list[tuple[str, int]]:
        """Languages whose queued backlog is below their weight-proportional share (min 8), worst first."""
        langs = [l for l in self.cfg.enabled_languages() if l.code in LANGUAGES and LANGUAGES[l.code].asr_supported]
        tw = sum(l.weight for l in langs) or 1.0
        bl = self.backlog_by_lang()
        target = self.cfg.discovery.target_backlog_videos
        out = []
        for l in langs:
            want = max(8, int(target * l.weight / tw))
            have = bl.get(l.code, 0)
            if have < want:
                out.append((l.code, want - have))
        return sorted(out, key=lambda t: -t[1])

    def step(self) -> bool:
        if self.cooldown_active():
            self.heartbeat("cooldown", force=True)
            return False
        target = self.cfg.discovery.target_backlog_videos
        have = self.backlog()
        deficits = self.language_deficits()
        if have >= target and not deficits:
            self.heartbeat("backlog full", f"{have}/{target} queued", force=True)
            return False
        if have >= target and deficits:
            # global backlog is fine but some languages are starved: top those up only
            added = 0
            for code, need in deficits[:4]:
                if self.stop or self.cooldown_active():
                    break
                qs = self.pending_queries(code, 3)
                if not qs:
                    self.make_queries(code)
                    qs = self.pending_queries(code, 3)
                for q in qs:
                    added += self.run_query(q)
                    self.heartbeat("topping up", f"[{code}] {q['query'][:60]}")
                    time.sleep(1.0)
            log.info("language top-up: +%d videos for %s", added, [d[0] for d in deficits[:4]])
            return True
        self.stats["rounds"] += 1
        self.heartbeat("discovering", f"backlog {have}/{target}", force=True)
        need = target - have
        added = self.run_seeds()
        added += self.expand_channels(need)
        rounds = 0
        while added < need and rounds < 6 and not self.stop and not self.cooldown_active():
            rounds += 1
            lang = self.pick_language()
            qs = self.pending_queries(lang, 4)
            if not qs:
                self.make_queries(lang)
                qs = self.pending_queries(lang, 4)
            for q in qs:
                if self.stop or self.cooldown_active():
                    break
                added += self.run_query(q)
                self.heartbeat("discovering", f"[{lang}] {q['query'][:60]}")
                time.sleep(1.0)
        if added:
            D.kv_set(self.conn, "source_cooldown_level", 0)
        log.info("discovery round: +%d videos (backlog now %d)", added, self.backlog())
        return True
