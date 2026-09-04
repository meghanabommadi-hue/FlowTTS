"""Shared helpers for the GPU box workers (no dependency on the orchestrator package)."""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import httpx

log = logging.getLogger("chaashini.gpu")


def setup_logging(name: str, log_dir: str | None = None) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = []
    if sys.stdout.isatty() or os.environ.get("CHAASHINI_LOG_STDOUT") == "1":
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_dir:
        from logging.handlers import RotatingFileHandler
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(Path(log_dir) / f"{name}.log", maxBytes=50 << 20, backupCount=5))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers or [logging.NullHandler()])
    for noisy in ("httpx", "httpcore", "urllib3", "nemo_logger", "lightning", "pytorch_lightning"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Orchestrator:
    """HTTP client for the orchestrator's internal API."""

    def __init__(self, base_url: str, token: str, host_header: str | None = None, worker: str = "gpu"):
        headers = {"X-Chaashini-Token": token}
        if host_header:
            headers["Host"] = host_header
        self.worker = worker
        self.client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=httpx.Timeout(600.0, connect=20.0))

    def health(self) -> bool:
        try:
            return self.client.get("/internal/health").status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def reset(self) -> int:
        try:
            r = self.client.post("/internal/workers/reset", json={"name": self.worker})
            return int(r.json().get("requeued", 0)) if r.status_code == 200 else 0
        except Exception:  # noqa: BLE001
            return 0

    def claim(self, kinds: list[str]) -> dict | None:
        r = self.client.post("/internal/jobs/claim", json={"kinds": kinds, "worker": self.worker})
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.json()

    def download_payload(self, job_id: int, dst: Path) -> int:
        n = 0
        with self.client.stream("GET", f"/internal/jobs/{job_id}/payload") as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
        return n

    def job_heartbeat(self, job_id: int) -> None:
        try:
            self.client.post(f"/internal/jobs/{job_id}/heartbeat")
        except Exception:  # noqa: BLE001
            pass

    def complete(self, job_id: int, ok: bool, result_path: Path | None, error: str = "", proc_seconds: float = 0.0) -> None:
        data = {"ok": "1" if ok else "0", "error": error[:2000], "proc_seconds": str(proc_seconds)}
        delay = 5.0
        for attempt in range(6):
            try:
                if result_path is not None and ok:
                    with open(result_path, "rb") as f:
                        r = self.client.post(f"/internal/jobs/{job_id}/complete", data=data,
                                             files={"result": (result_path.name, f, "application/octet-stream")})
                else:
                    r = self.client.post(f"/internal/jobs/{job_id}/complete", data=data)
                r.raise_for_status()
                return
            except Exception as e:  # noqa: BLE001
                log.warning("complete(%s) failed (%d): %s", job_id, attempt, e)
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("could not report job completion")

    def heartbeat(self, kind: str, state: str, current: str | None = None, stats: dict | None = None) -> None:
        try:
            self.client.post("/internal/workers/heartbeat", json={
                "name": self.worker, "kind": kind, "state": state, "current": current, "stats": stats,
                "host": socket.gethostname(), "pid": os.getpid()})
        except Exception as e:  # noqa: BLE001
            log.debug("heartbeat failed: %s", e)


class HeartbeatThread:
    """Posts a worker heartbeat every `interval` seconds with whatever state/current was last set."""

    def __init__(self, api: "Orchestrator", kind: str, interval: float = 15.0):
        import threading
        self.api, self.kind, self.interval = api, kind, interval
        self.state, self.current, self.stats = "starting", None, {}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name="heartbeat")

    def start(self):
        self._t.start()
        return self

    def set(self, state: str, current: str | None = None, stats: dict | None = None):
        self.state, self.current = state, current
        if stats is not None:
            self.stats = stats

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.api.heartbeat(self.kind, self.state, self.current, {**self.stats, **gpu_stats()})
            except Exception:  # noqa: BLE001
                pass

    def stop(self):
        self._stop.set()


def gpu_stats() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        u, m, t, temp = [x.strip() for x in out.split(",")]
        return {"gpu_util": int(u), "mem_used_mb": int(m), "mem_total_mb": int(t), "temp_c": int(temp)}
    except Exception:  # noqa: BLE001
        return {}


def extract_tar(path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:*") as tf:
        for m in tf.getmembers():
            if m.name.startswith("/") or ".." in m.name:
                raise ValueError("unsafe tar member")
        tf.extractall(dst)


def make_tar(src_files: list[Path], dst: Path, arcnames: list[str] | None = None) -> None:
    with tarfile.open(dst, "w") as tf:
        for i, p in enumerate(src_files):
            tf.add(p, arcname=(arcnames[i] if arcnames else p.name))


def load_env(path: str = "/opt/chaashini/gpu.env") -> dict:
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    env.update({k: v for k, v in os.environ.items() if k.startswith("CHAASHINI_")})
    return env
