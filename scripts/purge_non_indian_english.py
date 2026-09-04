"""One-off / re-runnable audit: every English source in the corpus must be verified Indian English.

For each source with accepted English clips:
  * real sources (still known locally): LLM judge on title + channel + description + a transcript sample,
    plus the Indian-context cue count over its transcripts;
  * sources restored from the Hub (only transcripts known): LLM judge on the transcripts, plus cue count.
Sources that fail are rejected (`not_indian_english`): staged files removed, and the affected shards on the
Hub rewritten without their clips (deleted when empty). Run on the orchestrator box:
    cd /opt/chaashini/app && set -a && . /opt/chaashini/chaashini.env && set +a && ../venv/bin/python scripts/purge_non_indian_english.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chaashini import db as D  # noqa: E402
from chaashini.config import get_config  # noqa: E402
from chaashini.llm import LLM, india_cue_count  # noqa: E402

cfg = get_config()
c = D.connect(cfg.paths.db_path)
D.init_schema(c)
llm = LLM(cfg.llm)

SYS = ("You audit an English speech corpus that must contain INDIAN ENGLISH only: speakers from India with an Indian accent. "
       "Non-Indian speakers (American, British, Canadian, Australian, other), Indian-diaspora creators based abroad, and recordings "
       "where a featured guest is non-Indian do NOT qualify. Judge from whatever is given (title, channel, description, transcript). "
       "Transcripts come from an ASR system tuned for Indian speech; Indian place names, rupees/lakh/crore, Indian institutions, "
       "Indian names and idioms are strong evidence FOR; foreign institutions, dollars, foreign places, foreign cultural references are "
       "evidence AGAINST. Output ONLY JSON: {\"indian\": true|false, \"confidence\": 0.0-1.0, \"reason\": \"...\"}.")


def judge(payload: str) -> tuple[bool, float, str]:
    try:
        text = llm.chat([{"role": "system", "content": SYS}, {"role": "user", "content": payload[:6000]}], temperature=0.0, max_tokens=200, retries=2)
        m = re.search(r"\{.*\}", text, re.S)
        o = json.loads(m.group(0)) if m else {}
        return bool(o.get("indian")), float(o.get("confidence", 0) or 0), str(o.get("reason", ""))[:160]
    except Exception as e:  # noqa: BLE001
        return False, 0.0, f"llm error: {e}"


sources = c.execute("SELECT video_id, COUNT(*) n, SUM(dur_ms)/1000.0 s FROM chunks WHERE status='accepted' AND lang='en' GROUP BY 1 ORDER BY n DESC").fetchall()
removed_ids: list[str] = []
kept = dropped = 0
for r in sources:
    vid, n, secs = r["video_id"], r["n"], r["s"]
    texts = [x["text"] or "" for x in c.execute("SELECT text FROM chunks WHERE video_id=? AND status='accepted' AND lang='en' ORDER BY idx", (vid,))]
    joined = " ".join(texts)
    cues = india_cue_count(joined)
    if vid.startswith("hub:"):
        payload = f"No title available (restored source).\nTRANSCRIPT ({n} clips, {secs / 60:.1f} min):\n" + joined[:5000]
        need = 2
    else:
        v = c.execute("SELECT title, channel, meta_json FROM videos WHERE id=?", (vid,)).fetchone()
        meta = D.uj(v["meta_json"], {}) if v else {}
        payload = (f"Title: {v['title'] if v else ''}\nChannel: {v['channel'] if v else ''}\nDescription: {(meta.get('description') or '')[:1200]}\n"
                   f"Tags: {', '.join((meta.get('tags') or [])[:20])}\nTRANSCRIPT SAMPLE ({n} clips, {secs / 60:.1f} min):\n" + joined[:3000])
        need = 1 if secs < 600 else 3
    ok, conf, why = judge(payload)
    keep = ok and conf >= 0.7 and cues >= need
    label = "KEEP" if keep else "DROP"
    print(f"{label} {vid[:26]:26} clips={n:4d} min={secs / 60:5.1f} cues={cues:3d} judge={ok}/{conf:.2f} :: {why[:90]}")
    if keep:
        kept += 1
        continue
    dropped += 1
    ids = [x["id"] for x in c.execute("SELECT id, staged_path FROM chunks WHERE video_id=? AND status='accepted' AND lang='en'", (vid,))]
    for x in c.execute("SELECT staged_path FROM chunks WHERE video_id=? AND status='accepted' AND lang='en' AND staged_path IS NOT NULL", (vid,)):
        for p in (x["staged_path"], os.path.splitext(x["staged_path"])[0] + ".json"):
            try:
                os.remove(p)
            except OSError:
                pass
    c.execute("UPDATE chunks SET status='rejected', reject_reason='not_indian_english', staged_path=NULL, updated_at=? WHERE video_id=? AND status='accepted' AND lang='en'", (time.time(), vid))
    c.commit()
    removed_ids.extend(ids)

print(f"\nsources kept={kept} dropped={dropped}; clips to remove from the Hub: {len(removed_ids)}")

# ---- rewrite the affected shards on the Hub
if removed_ids:
    import pyarrow.parquet as pq
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download
    from chaashini.packer import write_parquet
    api = HfApi(token=cfg.hf.token)
    rm = set(removed_ids)
    ops = []
    tmp = Path("/tmp/chaashini_purge"); tmp.mkdir(exist_ok=True)
    for sh in c.execute("SELECT id, hf_path FROM shards WHERE status='pushed' AND lang='en'").fetchall():
        p = hf_hub_download(cfg.hf.repo_id, sh["hf_path"], repo_type="dataset", token=cfg.hf.token)
        rows = pq.read_table(p).to_pylist()
        keep_rows = [r for r in rows if r["id"] not in rm]
        if len(keep_rows) == len(rows):
            continue
        if keep_rows:
            out = tmp / Path(sh["hf_path"]).name
            write_parquet(keep_rows, out)
            ops.append(CommitOperationAdd(path_in_repo=sh["hf_path"], path_or_fileobj=str(out)))
            c.execute("UPDATE shards SET n_chunks=?, duration_s=? WHERE id=?", (len(keep_rows), sum(r["duration_s"] for r in keep_rows), sh["id"]))
        else:
            ops.append(CommitOperationDelete(path_in_repo=sh["hf_path"]))
            c.execute("UPDATE shards SET status='failed', n_chunks=0, duration_s=0 WHERE id=?", (sh["id"],))
        print(f"shard {sh['hf_path']}: {len(rows)} -> {len(keep_rows)} clips")
    c.commit()
    if ops:
        info = api.create_commit(repo_id=cfg.hf.repo_id, repo_type="dataset", operations=ops, commit_message="remove English clips that are not Indian English")
        print("hub updated:", info.commit_url)
    # refresh the card
    from chaashini.pusher import update_card
    from chaashini.workers.publish import PublishWorker
    update_card(cfg.hf.token, cfg.hf.repo_id, PublishWorker("purge-card", cfg).card())
    print("card refreshed")
