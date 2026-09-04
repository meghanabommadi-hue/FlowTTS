"""Command line entry point: `python -m chaashini <command>`."""
from __future__ import annotations

import argparse
import json
import secrets
import sys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="chaashini")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="run the supervisor (all workers + API)")
    sub.add_parser("api", help="run the API/UI server only")
    w = sub.add_parser("worker", help="run a single worker")
    w.add_argument("kind", choices=["discover", "download", "process", "publish"])
    w.add_argument("name")
    sub.add_parser("init-db")
    sub.add_parser("status")
    p = sub.add_parser("push", help="force a push at the next publish check")
    p.add_argument("--now", action="store_true")
    sub.add_parser("card", help="print the dataset card")
    sub.add_parser("gen-token", help="print a fresh internal API token")
    a = sub.add_parser("add", help="queue a video/playlist/channel URL for a language")
    a.add_argument("lang")
    a.add_argument("url")
    a.add_argument("--priority", type=int, default=10)
    t = sub.add_parser("test-lid")
    t.add_argument("text")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        from .supervisor import main as run
        run()
    elif args.cmd == "api":
        import uvicorn
        from .config import get_config
        cfg = get_config()
        from .workers.base import setup_logging
        setup_logging("api", cfg.paths.logs_dir)
        uvicorn.run("chaashini.api:get_app", factory=True, host=cfg.api.host, port=cfg.api.port, workers=1, log_level="warning",
                    timeout_keep_alive=30, limit_concurrency=64)
    elif args.cmd == "worker":
        from .config import get_config
        from .workers.base import setup_logging
        cfg = get_config()
        setup_logging(args.name, cfg.paths.logs_dir)
        mod = {"discover": "DiscoverWorker", "download": "DownloadWorker", "process": "ProcessWorker", "publish": "PublishWorker"}[args.kind]
        import importlib
        cls = getattr(importlib.import_module(f"chaashini.workers.{args.kind}"), mod)
        cls(args.name, cfg).run()
    elif args.cmd == "init-db":
        from . import db as D
        from .config import get_config
        D.init_schema(D.connect(get_config().paths.db_path))
        print("ok")
    elif args.cmd == "status":
        from . import db as D
        from .config import get_config
        from .status import snapshot
        cfg = get_config()
        s = snapshot(D.connect(cfg.paths.db_path), cfg)
        print(json.dumps({k: s[k] for k in ("totals", "throughput", "videos", "gpu_jobs", "source", "disk")}, indent=2, ensure_ascii=False))
        for l in s["languages"]:
            print(f"  {l['code']:>4} {l.get('name',''):<12} accepted {l.get('accepted_hours',0):7.2f} h  pushed {l.get('pushed_hours',0):7.2f} h  queued {l.get('queued_videos',0)}")
        for w in s["workers"]:
            print(f"  [{'ok ' if w['alive'] else 'DEAD'}] {w['name']:<14} {w['state']:<18} {w['current'] or ''}")
    elif args.cmd == "push":
        from . import db as D
        from .config import get_config
        c = D.connect(get_config().paths.db_path)
        D.kv_set(c, "force_push", True)
        print("force_push set; the publish worker will pack and push on its next check")
    elif args.cmd == "card":
        from .config import get_config
        from .workers.publish import PublishWorker
        print(PublishWorker("card-cli", get_config()).card())
    elif args.cmd == "gen-token":
        print(secrets.token_urlsafe(32))
    elif args.cmd == "add":
        from . import db as D
        from .config import get_config
        from .ytsource import playlist_videos, source_hash
        import re, time
        cfg = get_config()
        c = D.connect(cfg.paths.db_path)
        salt = D.kv_get(c, "source_salt") or secrets.token_hex(16)
        D.kv_set(c, "source_salt", salt)
        url = args.url
        m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
        found = []
        if m and "list=" not in url:
            class F: pass
            f = F(); f.id, f.title, f.duration, f.channel_id, f.channel, f.view_count = m.group(1), "", None, None, None, None
            found = [f]
        else:
            found = playlist_videos(cfg.source, url, 1000)
        n = 0
        for f in found:
            t = time.time()
            cur = c.execute("INSERT OR IGNORE INTO videos(id, source_hash, lang_hint, channel_id, channel, title, duration_s, view_count, status, created_at, updated_at, priority) "
                            "VALUES (?,?,?,?,?,?,?,?,'discovered',?,?,?)", (f.id, source_hash(f.id, salt), args.lang, f.channel_id, f.channel, f.title, f.duration, f.view_count, t, t, args.priority))
            n += cur.rowcount
        print(f"queued {n} new items for {args.lang}")
    elif args.cmd == "test-lid":
        from .lid import identify
        print(json.dumps(identify(args.text).as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
