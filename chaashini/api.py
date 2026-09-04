"""FastAPI application: public dashboard API + static UI, and the token-protected internal
API that the GPU box workers pull jobs from."""
from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import db as D
from .config import Config, get_config
from .status import metrics_point, snapshot

log = logging.getLogger("chaashini.api")
UI_DIR = Path(__file__).resolve().parent.parent / "ui"

_local = threading.local()


def conn() -> "D.sqlite3.Connection":
    c = getattr(_local, "conn", None)
    if c is None:
        c = D.connect(get_config().paths.db_path)
        _local.conn = c
    return c


class Snapshot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data: dict = {"ts": 0}
        self.lock = threading.Lock()
        self._last_metric = 0.0
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True, name="snapshot")

    def start(self):
        self.thread.start()

    def _loop(self):
        c = D.connect(self.cfg.paths.db_path)
        while not self._stop:
            t0 = time.time()
            try:
                snap = snapshot(c, self.cfg)
                with self.lock:
                    self.data = snap
                if t0 - self._last_metric >= 60:
                    c.execute("INSERT OR REPLACE INTO metrics(ts, data_json) VALUES (?,?)", (t0, D.j(metrics_point(c, self.cfg))))
                    c.execute("DELETE FROM metrics WHERE ts < ?", (t0 - 30 * 86400,))
                    c.execute("DELETE FROM events WHERE ts < ?", (t0 - 7 * 86400,))
                    self._last_metric = t0
            except Exception as e:  # noqa: BLE001
                log.exception("snapshot failed: %s", e)
            time.sleep(max(1.0, self.cfg.api.status_refresh_s - (time.time() - t0)))


def create_app() -> FastAPI:
    cfg = get_config()
    app = FastAPI(title="Chaashini", docs_url=None, redoc_url=None)
    snap = Snapshot(cfg)

    @app.on_event("startup")
    def _startup():
        D.init_schema(D.connect(cfg.paths.db_path))
        snap.start()

    # ------------------------------------------------------------------ public
    @app.get("/api/status")
    def api_status():
        with snap.lock:
            return JSONResponse(snap.data, headers={"Cache-Control": "no-store"})

    @app.get("/api/recent_chunks")
    def api_recent(limit: int = 30, lang: str | None = None, status: str = "accepted"):
        c = conn()
        q = "SELECT c.*, v.title, v.lang_hint FROM chunks c LEFT JOIN videos v ON v.id=c.video_id WHERE c.status=? "
        params: list = [status]
        if lang:
            q += "AND c.lang=? "
            params.append(lang)
        q += "ORDER BY c.created_at DESC LIMIT ?"
        params.append(min(200, max(1, limit)))
        rows = []
        for r in c.execute(q, params):
            d = dict(r)
            d["metrics"] = D.uj(d.pop("metrics_json"), {})
            d["lid"] = D.uj(d.pop("lang_json"), {})
            sp = cfg.paths.samples_dir / f"{d['id']}.flac"
            d["preview"] = f"api/samples/{d['id']}.flac" if sp.exists() else None
            rows.append(d)
        return JSONResponse(rows, headers={"Cache-Control": "no-store"})

    @app.get("/api/samples/{name}")
    def api_sample(name: str):
        if "/" in name or ".." in name:
            raise HTTPException(404)
        p = cfg.paths.samples_dir / name
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(str(p), media_type="audio/flac", headers={"Cache-Control": "public, max-age=300"})

    @app.get("/api/video/{vid}")
    def api_video(vid: str):
        c = conn()
        r = c.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
        if not r:
            raise HTTPException(404)
        d = dict(r)
        d["meta"] = D.uj(d.pop("meta_json"), {})
        d["stats"] = D.uj(d.pop("stats_json"), {})
        d["chunks"] = [{**dict(x), "metrics": D.uj(x["metrics_json"], {})} for x in c.execute(
            "SELECT id, idx, start_ms, end_ms, dur_ms, speaker, status, reject_reason, enhanced, text, lang, lang_conf, metrics_json "
            "FROM chunks WHERE video_id=? ORDER BY idx", (vid,))]
        return d

    @app.get("/api/languages")
    def api_langs():
        with snap.lock:
            return snap.data.get("languages", [])

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "service": "chaashini", "ts": time.time()}

    # ------------------------------------------------------------------ internal (GPU workers)
    def auth(x_chaashini_token: str | None = Header(default=None)):
        if not cfg.api.internal_token or x_chaashini_token != cfg.api.internal_token:
            raise HTTPException(401, "bad token")
        return True

    @app.get("/internal/health")
    def i_health(_=Depends(auth)):
        return {"ok": True}

    @app.post("/internal/jobs/claim")
    def i_claim(body: dict, _=Depends(auth)):
        kinds = [k for k in body.get("kinds", []) if k in ("diarize", "transcribe", "enhance")]
        worker = str(body.get("worker", "gpu"))[:64]
        if not kinds:
            raise HTTPException(400, "kinds required")
        c = conn()
        # bias toward the deepest queue so neither stage starves
        depth = {r["kind"]: r["n"] for r in c.execute(
            f"SELECT kind, COUNT(*) n FROM gpu_jobs WHERE status='queued' AND kind IN ({','.join('?' * len(kinds))}) GROUP BY kind", kinds)}
        if not depth:
            stale = c.execute("SELECT COUNT(*) n FROM gpu_jobs WHERE status='running' AND leased_until < ?", (time.time(),)).fetchone()["n"]
            if not stale:
                return Response(status_code=204)
        order = sorted(kinds, key=lambda k: -depth.get(k, 0))
        job = D.claim_job(c, order[:1] if depth else kinds, worker, cfg.gpu.job_lease_s) or D.claim_job(c, kinds, worker, cfg.gpu.job_lease_s)
        if not job:
            return Response(status_code=204)
        if job["attempts"] > cfg.gpu.max_attempts:
            D.finish_job(c, job["id"], False, error="attempt budget exhausted")
            _on_job_finished(c, cfg, job["id"])
            return Response(status_code=204)
        return {"id": job["id"], "kind": job["kind"], "video_id": job["video_id"], "n_items": job["n_items"],
                "audio_seconds": job["audio_seconds"], "payload_bytes": job["payload_bytes"], "attempt": job["attempts"]}

    @app.get("/internal/jobs/{job_id}/payload")
    def i_payload(job_id: int, _=Depends(auth)):
        c = conn()
        r = c.execute("SELECT payload_path FROM gpu_jobs WHERE id=?", (job_id,)).fetchone()
        if not r or not os.path.exists(r["payload_path"]):
            raise HTTPException(404)
        return FileResponse(r["payload_path"], media_type="application/octet-stream")

    @app.post("/internal/jobs/{job_id}/heartbeat")
    def i_job_hb(job_id: int, _=Depends(auth)):
        c = conn()
        c.execute("UPDATE gpu_jobs SET leased_until=? WHERE id=? AND status='running'", (time.time() + cfg.gpu.job_lease_s, job_id))
        return {"ok": True}

    @app.post("/internal/jobs/{job_id}/complete")
    async def i_complete(job_id: int, ok: str = Form(...), error: str = Form(""), proc_seconds: float = Form(0.0),
                         result: UploadFile | None = File(default=None), _=Depends(auth)):
        c = conn()
        job = c.execute("SELECT * FROM gpu_jobs WHERE id=?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404)
        if job["status"] != "running":
            return {"ok": True, "note": "job not running (already finished?)"}
        good = ok.lower() in ("1", "true", "yes")
        result_path = None
        if good and result is not None:
            ext = Path(result.filename or "result.bin").suffix or ".bin"
            wd = Path(job["payload_path"]).parent
            result_path = str(wd / f"{job['kind']}_result{ext}")
            tmp = result_path + ".tmp"
            with open(tmp, "wb") as f:
                while True:
                    chunk = await result.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, result_path)
        D.finish_job(c, job_id, good, result_path=result_path, error=(error or None)[:2000] if error else None, proc_seconds=proc_seconds)
        _on_job_finished(c, cfg, job_id)
        return {"ok": True}

    @app.post("/internal/workers/heartbeat")
    def i_worker_hb(body: dict, _=Depends(auth)):
        c = conn()
        D.heartbeat(c, str(body.get("name", "gpu"))[:64], str(body.get("kind", "gpu"))[:32], str(body.get("state", ""))[:64],
                    (body.get("current") or None), body.get("stats"), host=body.get("host"), pid=body.get("pid"))
        return {"ok": True}

    # ------------------------------------------------------------------ UI
    if UI_DIR.exists():
        @app.get("/", response_class=HTMLResponse)
        def index():
            return HTMLResponse((UI_DIR / "index.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})
        app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")
    return app


def _on_job_finished(c, cfg: Config, job_id: int) -> None:
    """Advance the video state machine when a GPU job finishes."""
    job = c.execute("SELECT * FROM gpu_jobs WHERE id=?", (job_id,)).fetchone()
    if not job:
        return
    vid, kind, ok = job["video_id"], job["kind"], job["status"] == "done"
    v = c.execute("SELECT status FROM videos WHERE id=?", (vid,)).fetchone()
    if not v:
        return
    expect = {"diarize": "diarize_queued", "enhance": "enhance_queued", "transcribe": "transcribe_queued"}[kind]
    if v["status"] != expect:
        return
    if ok:
        nxt = {"diarize": "diarized", "enhance": "enhanced", "transcribe": "transcribed"}[kind]
        D.set_video_status(c, vid, nxt)
    else:
        if job["attempts"] < cfg.gpu.max_attempts:
            # requeue the same payload
            c.execute("UPDATE gpu_jobs SET status='queued', worker=NULL, leased_until=NULL, started_at=NULL WHERE id=?", (job_id,))
            return
        if kind == "enhance":
            D.set_video_status(c, vid, "enhanced", error=f"enhance failed: {job['error']}")
        else:
            D.set_video_status(c, vid, "failed", error=f"{kind} failed: {job['error']}")
        D.event(c, "gpu", f"{kind} job {job_id} failed for {vid}: {job['error']}", level="error")


app = None


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app
