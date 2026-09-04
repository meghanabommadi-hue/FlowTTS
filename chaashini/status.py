"""Dashboard snapshot: one SQL sweep -> JSON. Refreshed every few seconds by the API process."""
from __future__ import annotations

import shutil
import sqlite3
import time

from .config import Config
from .db import kv_get, now, uj
from .export import staged_seconds
from .languages import LANGUAGES


def _h(sec: float | None) -> float:
    return round((sec or 0.0) / 3600.0, 3)


def snapshot(conn: sqlite3.Connection, cfg: Config) -> dict:
    t = now()
    out: dict = {"ts": t, "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))}

    vs = {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) n FROM videos GROUP BY status")}
    out["videos"] = vs
    out["videos_total"] = sum(vs.values())

    r = conn.execute("SELECT COUNT(*) n, COALESCE(SUM(dur_ms),0)/1000.0 s FROM chunks WHERE status='accepted'").fetchone()
    acc_n, acc_s = r["n"], r["s"]
    r = conn.execute("SELECT COUNT(*) n, COALESCE(SUM(dur_ms),0)/1000.0 s FROM chunks WHERE status='rejected'").fetchone()
    rej_n, rej_s = r["n"], r["s"]
    r = conn.execute("SELECT COALESCE(SUM(duration_s),0) s, COUNT(*) n FROM shards WHERE status='pushed'").fetchone()
    pushed_s, pushed_shards = r["s"], r["n"]
    r = conn.execute("SELECT COALESCE(SUM(duration_s),0) s FROM shards WHERE status='built'").fetchone()
    built_s = r["s"]
    r = conn.execute("SELECT COALESCE(SUM(duration_s),0) s, COUNT(*) n FROM videos WHERE status IN ('done','rejected')").fetchone()
    src_s, src_n = r["s"], r["n"]
    r = conn.execute("SELECT COALESCE(SUM(duration_s),0) s FROM videos WHERE status='done'").fetchone()
    src_done_s = r["s"]
    r = conn.execute("SELECT COUNT(*) n FROM chunks WHERE status='accepted' AND enhanced=1").fetchone()
    enh_n = r["n"]
    staged = staged_seconds(cfg.paths.staging_dir)
    staged_s = sum(staged.values())
    out["totals"] = {
        "accepted_hours": _h(acc_s), "accepted_chunks": acc_n, "rejected_chunks": rej_n, "rejected_hours": _h(rej_s),
        "pushed_hours": _h(pushed_s), "pushed_shards": pushed_shards, "built_unpushed_hours": _h(built_s),
        "staged_hours": _h(staged_s), "source_hours_processed": _h(src_s), "source_hours_kept_videos": _h(src_done_s),
        "videos_processed": src_n, "enhanced_accepted": enh_n,
        "yield_ratio": round(acc_s / src_s, 4) if src_s else 0.0,
        "accept_ratio_chunks": round(acc_n / (acc_n + rej_n), 4) if (acc_n + rej_n) else 0.0,
        "avg_chunk_s": round(acc_s / acc_n, 2) if acc_n else 0.0,
        "next_push_hours": cfg.hf.push_every_hours, "push_progress": round(min(1.0, (staged_s + built_s) / (cfg.hf.push_every_hours * 3600)), 4),
    }

    def window(sec: int) -> dict:
        r = conn.execute("SELECT COUNT(*) n, COALESCE(SUM(dur_ms),0)/1000.0 s FROM chunks WHERE status='accepted' AND created_at > ?", (t - sec,)).fetchone()
        v = conn.execute("SELECT COUNT(*) n, COALESCE(SUM(duration_s),0) s FROM videos WHERE status IN ('done','rejected') AND updated_at > ?", (t - sec,)).fetchone()
        return {"accepted_hours": _h(r["s"]), "accepted_chunks": r["n"], "videos": v["n"], "source_hours": _h(v["s"]),
                "hours_per_hour": round((r["s"] / 3600.0) / (sec / 3600.0), 3)}
    out["throughput"] = {"1h": window(3600), "6h": window(6 * 3600), "24h": window(24 * 3600)}
    last = conn.execute("SELECT MAX(updated_at) t FROM chunks WHERE status='accepted'").fetchone()["t"]
    last_done = conn.execute("SELECT MAX(updated_at) t FROM videos WHERE status IN ('done','rejected')").fetchone()["t"]
    active = sum(v for k, v in vs.items() if k not in ("discovered", "done", "rejected", "failed"))
    out["freshness"] = {"last_accept_age_s": round(t - last) if last else None, "last_finish_age_s": round(t - last_done) if last_done else None,
                        "active_sources": active, "stalled": bool(last and t - last > 1800 and vs.get("discovered", 0) > 0)}

    langs: dict[str, dict] = {}
    for r in conn.execute("SELECT lang, COUNT(*) n, COALESCE(SUM(dur_ms),0)/1000.0 s FROM chunks WHERE status='accepted' GROUP BY lang"):
        l = r["lang"] or "und"
        langs.setdefault(l, {"code": l, "name": LANGUAGES[l].name if l in LANGUAGES else l})
        langs[l].update({"accepted_hours": _h(r["s"]), "accepted_chunks": r["n"]})
    for r in conn.execute("SELECT lang, COALESCE(SUM(duration_s),0) s FROM shards WHERE status='pushed' GROUP BY lang"):
        langs.setdefault(r["lang"], {"code": r["lang"], "name": LANGUAGES[r["lang"]].name if r["lang"] in LANGUAGES else r["lang"]})
        langs[r["lang"]]["pushed_hours"] = _h(r["s"])
    for r in conn.execute("SELECT lang, COUNT(*) n FROM chunks WHERE status='rejected' GROUP BY lang"):
        l = r["lang"] or "und"
        if l in langs:
            langs[l]["rejected_chunks"] = r["n"]
    for r in conn.execute("SELECT lang_hint l, COUNT(*) n, COALESCE(SUM(duration_s),0) s FROM videos WHERE status NOT IN ('done','rejected','failed') GROUP BY lang_hint"):
        langs.setdefault(r["l"] or "und", {"code": r["l"] or "und", "name": LANGUAGES[r["l"]].name if r["l"] in LANGUAGES else (r["l"] or "und")})
        langs[r["l"] or "und"]["queued_videos"] = r["n"]
        langs[r["l"] or "und"]["queued_source_hours"] = _h(r["s"])
    for l, s in staged.items():
        langs.setdefault(l, {"code": l, "name": LANGUAGES[l].name if l in LANGUAGES else l})["staged_hours"] = _h(s)
    for l in [x.code for x in cfg.enabled_languages()]:
        langs.setdefault(l, {"code": l, "name": LANGUAGES[l].name if l in LANGUAGES else l})
    out["languages"] = sorted(langs.values(), key=lambda d: -(d.get("accepted_hours", 0)))

    reasons = {r["reject_reason"] or "unknown": r["n"] for r in conn.execute(
        "SELECT reject_reason, COUNT(*) n FROM chunks WHERE status='rejected' AND created_at > ? GROUP BY reject_reason ORDER BY n DESC LIMIT 20", (t - 86400,))}
    out["reject_reasons_24h"] = reasons
    vreasons = {}
    for r in conn.execute("SELECT error FROM videos WHERE status='rejected' AND updated_at > ?", (t - 86400,)):
        k = (r["error"] or "unknown").split(":")[0][:40]
        vreasons[k] = vreasons.get(k, 0) + 1
    out["video_reject_reasons_24h"] = dict(sorted(vreasons.items(), key=lambda kv: -kv[1])[:12])

    q = {}
    for r in conn.execute("SELECT kind, status, COUNT(*) n, AVG(proc_seconds) p, AVG(audio_seconds) a FROM gpu_jobs WHERE created_at > ? GROUP BY kind, status", (t - 86400,)):
        q.setdefault(r["kind"], {})[r["status"]] = {"n": r["n"], "avg_proc_s": round(r["p"] or 0, 2), "avg_audio_s": round(r["a"] or 0, 1)}
    for r in conn.execute("SELECT kind, COUNT(*) n FROM gpu_jobs WHERE status='queued' GROUP BY kind"):
        q.setdefault(r["kind"], {})["queued_now"] = r["n"]
    for r in conn.execute("SELECT kind, COUNT(*) n FROM gpu_jobs WHERE status='running' GROUP BY kind"):
        q.setdefault(r["kind"], {})["running_now"] = r["n"]
    out["gpu_jobs"] = q

    workers = []
    for r in conn.execute("SELECT * FROM workers ORDER BY kind, name"):
        age = t - r["heartbeat_at"]
        workers.append({"name": r["name"], "kind": r["kind"], "host": r["host"], "pid": r["pid"], "state": r["state"],
                        "current": r["current"], "age_s": round(age, 1), "alive": age < 90, "stats": uj(r["stats_json"], {}),
                        "uptime_s": round(t - r["started_at"])})
    out["workers"] = workers

    out["recent_videos"] = [
        {"id": r["id"], "title": r["title"], "lang": r["lang_hint"], "status": r["status"], "duration_s": r["duration_s"],
         "channel": r["channel"], "error": r["error"], "stats": uj(r["stats_json"], {}), "updated_at": r["updated_at"]}
        for r in conn.execute("SELECT id, title, lang_hint, status, duration_s, channel, error, stats_json, updated_at FROM videos "
                              "WHERE status IN ('done','rejected','failed') ORDER BY updated_at DESC LIMIT 25")]
    out["active_videos"] = [
        {"id": r["id"], "title": r["title"], "lang": r["lang_hint"], "status": r["status"], "duration_s": r["duration_s"],
         "since_s": round(t - (r["stage_entered_at"] or r["updated_at"])), "worker": r["leased_by"]}
        for r in conn.execute("SELECT id, title, lang_hint, status, duration_s, stage_entered_at, updated_at, leased_by FROM videos "
                              "WHERE status NOT IN ('discovered','done','rejected','failed') ORDER BY updated_at DESC LIMIT 40")]
    out["pushes"] = [dict(r) for r in conn.execute("SELECT * FROM pushes ORDER BY started_at DESC LIMIT 10")]
    out["events"] = [{"ts": r["ts"], "level": r["level"], "kind": r["kind"], "msg": r["msg"], "data": uj(r["data_json"], None)}
                     for r in conn.execute("SELECT * FROM events ORDER BY ts DESC LIMIT 40")]
    out["quality_hist"] = _hist(conn, "dnsmos_ovrl", 1.0, 5.0, 16)
    out["duration_hist"] = _dur_hist(conn)
    du = shutil.disk_usage(str(cfg.paths.data_dir)) if cfg.paths.data_dir.exists() else None
    out["disk"] = {"free_gb": round(du.free / 1e9, 1), "total_gb": round(du.total / 1e9, 1), "used_pct": round(100 * du.used / du.total, 1)} if du else {}
    cd = kv_get(conn, "source_cooldown_until", 0) or 0
    out["source"] = {"cooldown_until": cd, "cooldown_s": max(0, round(cd - t)), "cooldown_level": kv_get(conn, "source_cooldown_level", 0) or 0,
                     "queries": conn.execute("SELECT COUNT(*) n FROM queries").fetchone()["n"],
                     "channels": conn.execute("SELECT COUNT(*) n FROM channels").fetchone()["n"]}
    out["series"] = [{"ts": r["ts"], **(uj(r["data_json"], {}) or {})} for r in conn.execute(
        "SELECT ts, data_json FROM metrics WHERE ts > ? ORDER BY ts", (t - 7 * 86400,))]
    out["config"] = {"push_every_hours": cfg.hf.push_every_hours, "repo_id": cfg.hf.repo_id, "export_sr": cfg.audio.export_sr,
                     "languages": [l.code for l in cfg.enabled_languages()], "workers": cfg.workers.model_dump()}
    return out


def _hist(conn: sqlite3.Connection, key: str, lo: float, hi: float, bins: int) -> dict:
    rows = conn.execute("SELECT metrics_json FROM chunks WHERE status='accepted' ORDER BY created_at DESC LIMIT 4000").fetchall()
    counts = [0] * bins
    w = (hi - lo) / bins
    for r in rows:
        m = uj(r["metrics_json"], {}) or {}
        v = m.get(key)
        if v is None:
            continue
        i = int((float(v) - lo) / w)
        counts[min(max(i, 0), bins - 1)] += 1
    return {"lo": lo, "hi": hi, "bins": bins, "counts": counts, "key": key}


def _dur_hist(conn: sqlite3.Connection) -> dict:
    edges = [0.5, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30.01]
    counts = [0] * (len(edges) - 1)
    for r in conn.execute("SELECT dur_ms FROM chunks WHERE status='accepted' ORDER BY created_at DESC LIMIT 6000"):
        d = r["dur_ms"] / 1000.0
        for i in range(len(edges) - 1):
            if edges[i] <= d < edges[i + 1]:
                counts[i] += 1
                break
    return {"edges": edges, "counts": counts}


def metrics_point(conn: sqlite3.Connection, cfg: Config) -> dict:
    t = now()
    r = conn.execute("SELECT COUNT(*) n, COALESCE(SUM(dur_ms),0)/1000.0 s FROM chunks WHERE status='accepted'").fetchone()
    p = conn.execute("SELECT COALESCE(SUM(duration_s),0) s FROM shards WHERE status='pushed'").fetchone()
    v = conn.execute("SELECT COUNT(*) n FROM videos WHERE status IN ('done','rejected')").fetchone()
    q = conn.execute("SELECT COUNT(*) n FROM gpu_jobs WHERE status='queued'").fetchone()
    return {"accepted_h": _h(r["s"]), "accepted_n": r["n"], "pushed_h": _h(p["s"]), "videos_done": v["n"], "gpu_queue": q["n"]}
