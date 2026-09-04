"""Remove every accepted clip whose language is not in the enabled set (or a language named on the
command line) from the local state, the staging area and the shards already published.

    python scripts/purge_language.py            # purge every language not in configs/chaashini.yaml
    python scripts/purge_language.py ur ks      # purge specific languages
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chaashini import db as D  # noqa: E402
from chaashini.config import get_config  # noqa: E402

cfg = get_config()
c = D.connect(cfg.paths.db_path)
D.init_schema(c)
enabled = {l.code for l in cfg.enabled_languages()}
targets = set(sys.argv[1:]) or {r["lang"] for r in c.execute("SELECT DISTINCT lang FROM chunks WHERE status='accepted' AND lang IS NOT NULL")
                                if r["lang"] not in enabled}
if not targets:
    print("nothing to purge: every accepted language is in the enabled set")
    raise SystemExit(0)
print("purging languages:", sorted(targets))

removed: list[str] = []
for lang in sorted(targets):
    rows = c.execute("SELECT id, staged_path FROM chunks WHERE status='accepted' AND lang=?", (lang,)).fetchall()
    for r in rows:
        if r["staged_path"]:
            for p in (r["staged_path"], os.path.splitext(r["staged_path"])[0] + ".json"):
                try:
                    os.remove(p)
                except OSError:
                    pass
        sp = cfg.paths.samples_dir / f"{r['id']}.flac"
        if sp.exists():
            sp.unlink()
    c.execute("UPDATE chunks SET status='rejected', reject_reason='lang_not_collected', staged_path=NULL, updated_at=? "
              "WHERE status='accepted' AND lang=?", (time.time(), lang))
    c.execute("DELETE FROM clip_fps WHERE chunk_id IN (SELECT id FROM chunks WHERE reject_reason='lang_not_collected' AND lang=?)", (lang,))
    c.commit()
    removed.extend(r["id"] for r in rows)
    print(f"  {lang}: {len(rows)} clips rejected locally")

if removed:
    import pyarrow.parquet as pq
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download
    from chaashini.packer import write_parquet
    api = HfApi(token=cfg.hf.token)
    rm = set(removed)
    ops = []
    tmp = Path("/tmp/chaashini_langpurge"); tmp.mkdir(exist_ok=True)
    hub_files = set(api.list_repo_files(cfg.hf.repo_id, repo_type="dataset"))
    for sh in c.execute("SELECT id, hf_path, lang FROM shards WHERE status='pushed'").fetchall():
        if sh["hf_path"] not in hub_files:
            continue
        if sh["lang"] not in targets:
            continue
        p = hf_hub_download(cfg.hf.repo_id, sh["hf_path"], repo_type="dataset", token=cfg.hf.token)
        rows = pq.read_table(p).to_pylist()
        keep = [r for r in rows if r["id"] not in rm]
        if len(keep) == len(rows):
            continue
        if keep:
            out = tmp / Path(sh["hf_path"]).name
            write_parquet(keep, out)
            ops.append(CommitOperationAdd(path_in_repo=sh["hf_path"], path_or_fileobj=str(out)))
            c.execute("UPDATE shards SET n_chunks=?, duration_s=? WHERE id=?", (len(keep), sum(r["duration_s"] for r in keep), sh["id"]))
        else:
            ops.append(CommitOperationDelete(path_in_repo=sh["hf_path"]))
            c.execute("UPDATE shards SET status='failed', n_chunks=0, duration_s=0 WHERE id=?", (sh["id"],))
        print(f"  shard {sh['hf_path']}: {len(rows)} -> {len(keep)} clips")
    c.commit()
    if ops:
        info = api.create_commit(repo_id=cfg.hf.repo_id, repo_type="dataset", operations=ops,
                                 commit_message=f"remove clips in languages not collected: {', '.join(sorted(targets))}")
        print("hub updated:", info.commit_url)
    from chaashini.pusher import update_card
    from chaashini.workers.publish import PublishWorker
    update_card(cfg.hf.token, cfg.hf.repo_id, PublishWorker("langpurge-card", cfg).card())
    print("card refreshed")
