"""Worker loop scaffolding: logging, heartbeat, graceful stop, backoff on repeated errors."""
from __future__ import annotations

import logging
import os
import signal
import socket
import sys
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .. import db as D
from ..config import Config, get_config

log = logging.getLogger("chaashini.worker")


def setup_logging(name: str, logs_dir: Path, level: int = logging.INFO) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = RotatingFileHandler(logs_dir / f"{name}.log", maxBytes=50 << 20, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if sys.stdout.isatty() or os.environ.get("CHAASHINI_LOG_STDOUT") == "1":
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    for noisy in ("httpx", "httpcore", "urllib3", "numba", "filelock", "huggingface_hub", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Worker:
    kind = "base"

    def __init__(self, name: str, cfg: Config | None = None):
        self.name = name
        self.cfg = cfg or get_config()
        self.conn = D.connect(self.cfg.paths.db_path)
        D.init_schema(self.conn)
        self.stop = False
        self.stats: dict = {}
        self._last_hb = 0.0
        self._errors = 0
        signal.signal(signal.SIGTERM, self._sig)
        signal.signal(signal.SIGINT, self._sig)

    def _sig(self, *_):
        log.info("%s: stop requested", self.name)
        self.stop = True

    def heartbeat(self, state: str, current: str | None = None, force: bool = False) -> None:
        t = time.time()
        if force or t - self._last_hb >= self.cfg.workers.heartbeat_s:
            try:
                D.heartbeat(self.conn, self.name, self.kind, state, current, self.stats, host=socket.gethostname(), pid=os.getpid())
            except Exception as e:  # noqa: BLE001
                log.warning("heartbeat failed: %s", e)
            self._last_hb = t

    def event(self, kind: str, msg: str, level: str = "info", data: dict | None = None) -> None:
        try:
            D.event(self.conn, kind, msg, level, data)
        except Exception:  # noqa: BLE001
            pass

    def run(self) -> None:
        log.info("%s starting (pid %d)", self.name, os.getpid())
        self.heartbeat("starting", force=True)
        while not self.stop:
            try:
                did = self.step()
                self._errors = 0
                if not did:
                    self.heartbeat("idle", force=True)
                    self.sleep(self.idle_sleep())
            except KeyboardInterrupt:
                break
            except Exception as e:  # noqa: BLE001
                self._errors += 1
                log.error("%s step failed: %s\n%s", self.name, e, traceback.format_exc())
                self.event("worker", f"{self.name}: {type(e).__name__}: {e}", level="error")
                self.sleep(min(300, 5 * 2 ** min(self._errors, 6)))
        self.heartbeat("stopped", force=True)
        log.info("%s stopped", self.name)

    def sleep(self, s: float) -> None:
        end = time.time() + s
        while not self.stop and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    def idle_sleep(self) -> float:
        return 5.0

    def step(self) -> bool:
        raise NotImplementedError


def free_gb(path: Path) -> float:
    import shutil
    try:
        return shutil.disk_usage(str(path)).free / 1e9
    except FileNotFoundError:
        path.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(str(path)).free / 1e9
