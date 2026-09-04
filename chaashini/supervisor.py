"""Spawns and babysits the worker processes and the API server. Restarts crashed children
with exponential backoff; runs the lease janitor; guards the nginx location (optional)."""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import db as D
from .config import Config, get_config
from .workers.base import setup_logging

log = logging.getLogger("chaashini.supervisor")


class Child:
    def __init__(self, name: str, argv: list[str]):
        self.name = name
        self.argv = argv
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self.last_start = 0.0

    def start(self, env: dict) -> None:
        self.proc = subprocess.Popen(self.argv, env=env, stdin=subprocess.DEVNULL)
        self.last_start = time.time()
        log.info("started %s (pid %d)", self.name, self.proc.pid)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def build_children(cfg: Config) -> list[Child]:
    py = sys.executable
    kids = [Child("api", [py, "-m", "chaashini", "api"])]
    w = cfg.workers
    for i in range(w.discover):
        kids.append(Child(f"discover-{i}", [py, "-m", "chaashini", "worker", "discover", f"discover-{i}"]))
    for i in range(w.download):
        kids.append(Child(f"download-{i}", [py, "-m", "chaashini", "worker", "download", f"download-{i}"]))
    for i in range(w.process):
        kids.append(Child(f"process-{i}", [py, "-m", "chaashini", "worker", "process", f"process-{i}"]))
    for i in range(w.publish):
        kids.append(Child(f"publish-{i}", [py, "-m", "chaashini", "worker", "publish", f"publish-{i}"]))
    return kids


def main() -> None:
    cfg = get_config()
    setup_logging("supervisor", cfg.paths.logs_dir)
    for p in (cfg.paths.data_dir, cfg.paths.work_dir, cfg.paths.staging_dir, cfg.paths.shards_dir, cfg.paths.samples_dir, cfg.paths.logs_dir):
        p.mkdir(parents=True, exist_ok=True)
    conn = D.connect(cfg.paths.db_path)
    D.init_schema(conn)
    D.event(conn, "system", "supervisor started")
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", str(cfg.workers.torch_threads))
    env.setdefault("MKL_NUM_THREADS", str(cfg.workers.torch_threads))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    kids = build_children(cfg)
    stop = False

    def _sig(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    for k in kids:
        k.start(env)
        time.sleep(0.5)
    last_janitor = 0.0
    while not stop:
        for k in kids:
            if not k.alive():
                rc = k.proc.returncode if k.proc else None
                k.restarts += 1
                delay = min(120, 2 ** min(k.restarts, 7))
                log.warning("%s exited (rc=%s); restart #%d in %ds", k.name, rc, k.restarts, delay)
                D.event(conn, "system", f"{k.name} exited (rc={rc}); restarting in {delay}s", level="warn")
                time.sleep(delay)
                if stop:
                    break
                k.start(env)
        if time.time() - last_janitor > 60:
            try:
                rel = D.release_stale(conn)
                if rel:
                    log.info("janitor: %s", rel)
                    D.event(conn, "system", f"janitor released stale leases: {rel}", level="warn")
            except Exception as e:  # noqa: BLE001
                log.warning("janitor failed: %s", e)
            last_janitor = time.time()
        time.sleep(2)
    log.info("stopping children")
    for k in kids:
        if k.alive():
            k.proc.terminate()
    deadline = time.time() + 30
    for k in kids:
        if k.proc:
            try:
                k.proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                k.proc.kill()
    D.event(conn, "system", "supervisor stopped")
    log.info("supervisor stopped")


if __name__ == "__main__":
    main()
