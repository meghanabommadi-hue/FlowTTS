#!/usr/bin/env python3
"""Collect producer + trainer state into one status.json for the dashboard.

Answers the question the logs make hard: how much data has actually gone into
training, and is quality moving.
"""
from __future__ import annotations

import glob, json, os, re, subprocess, time

BASE = os.environ.get("BASE", "/home/jovyan/omnivoice-train")
PROG = f"{BASE}/run_prog"
OUT = f"{PROG}/ui/status.json"
LANGS = {"ibo": "Igbo", "yor": "Yorùbá", "hau": "Hausa", "pcm": "Nigerian Pidgin"}


def manifest_stats(path):
    """(shards, utts, hours) from a webdataset data.lst."""
    if not os.path.exists(path):
        return 0, 0, 0.0
    n = u = 0
    secs = 0.0
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                n += 1
                u += int(p[2])
                secs += float(p[3])
    return n, u, secs / 3600


def tail_progress(log):
    """Latest step / loss / lr out of the tqdm line."""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, os.path.getsize(log) - 200_000))
            txt = f.read().decode("utf-8", "ignore").replace("\r", "\n")
    except OSError:
        return {}
    out = {}
    m = list(re.finditer(r"(\d+)/(\d+) \[[^\]]*\]?,? ?([\d.]+)it/s", txt))
    if m:
        out["step"], out["total_steps"] = int(m[-1].group(1)), int(m[-1].group(2))
        out["it_per_s"] = float(m[-1].group(3))
    for key, pat in (("train_loss", r"loss=([\d.]+)"), ("lr", r"lr=([\d.e+-]+)")):
        mm = list(re.finditer(pat, txt))
        if mm:
            try:
                out[key] = float(mm[-1].group(1))
            except ValueError:
                pass
    ev = list(re.finditer(r"Eval Loss: ([\d.]+)", txt))
    if ev:
        out["eval_loss"] = float(ev[-1].group(1))
        out["n_evals"] = len(ev)
    q = list(re.finditer(r"tts-bench metrics: ([^\n]+)", txt))
    if q:
        for part in q[-1].group(1).split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    out[f"q_{k.strip()}"] = float(v)
                except ValueError:
                    pass
    return out


def gpu():
    try:
        o = subprocess.run(["nvidia-smi",
                            "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15).stdout.strip()
        u, t, util = [int(x) for x in o.split("\n")[0].split(", ")]
        return {"used_gb": round(u / 1024, 1), "total_gb": round(t / 1024, 1),
                "util_pct": util}
    except Exception:
        return {}


def main():
    os.makedirs(f"{PROG}/ui", exist_ok=True)
    while True:
        st = {}
        try:
            st = json.load(open(f"{PROG}/producer_state.json"))
        except Exception:
            pass

        base_n, base_u, base_h = manifest_stats(f"{BASE}/run/tokens/train/data.lst")
        new_n, new_u, new_h = manifest_stats(f"{PROG}/tokens/train/data.lst")

        logs = sorted(glob.glob(f"{PROG}/logs/train-c*.log"), key=os.path.getmtime)
        prog = tail_progress(logs[-1]) if logs else {}

        ckpts = sorted(
            (int(d.rsplit("-", 1)[1]) for d in glob.glob(f"{PROG}/exp/checkpoint-*")
             if d.rsplit("-", 1)[1].isdigit()))

        langs = {}
        for lg, name in LANGS.items():
            langs[lg] = {"name": name,
                         "hours": round(float(st.get("hours", {}).get(lg, 0.0)), 3),
                         "chunks": int(st.get("chunks", {}).get(lg, 0))}

        doc = {
            "updated_utc": int(time.time()),
            "data": {
                "base_hours": round(base_h, 2), "base_utts": base_u, "base_shards": base_n,
                "new_hours": round(new_h, 2), "new_utts": new_u, "new_shards": new_n,
                "total_hours": round(base_h + new_h, 2),
                "total_utts": base_u + new_u,
            },
            "producer": {"langs": langs, "rejected": int(st.get("rejected", 0)),
                         "clips_seen": len(st.get("done_ids", []))},
            "training": prog,
            "checkpoints": {"count": len(ckpts), "latest": ckpts[-1] if ckpts else None},
            "gpu": gpu(),
            "cycle_log": _tail(f"{PROG}/logs/progressive.log", 12),
            "producer_log": _tail(f"{PROG}/logs/producer.log", 8),
        }
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=1)
        os.replace(tmp, OUT)
        time.sleep(20)


def _tail(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 20000))
            return [l for l in f.read().decode("utf-8", "ignore").splitlines()
                    if l.strip()][-n:]
    except OSError:
        return []


if __name__ == "__main__":
    main()
