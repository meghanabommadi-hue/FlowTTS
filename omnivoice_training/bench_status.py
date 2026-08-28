#!/usr/bin/env python3
"""Collect tts-bench runs into one live JSON for the benchmark report page.

Reads every completed run's run_meta.json (which carries a `headline` dict of
metric -> {engine_id: value}) and folds them into a language x engine matrix,
then adds the in-flight run's progress from the driver log. Built so a third
engine (OmniNaija) drops in without changing anything here.
"""
from __future__ import annotations

import glob, json, os, re, time

BASE = os.environ.get("BASE", "/home/jovyan")
RUNS = f"{BASE}/tts-bench/models/omnivoice-naija/runs"
LOGS = f"{BASE}/bench_logs"
OUT = os.environ.get("BENCH_OUT",
                     "/home/jovyan/omnivoice-train/run_prog/ui/bench_status.json")

# engine_id -> display label. Adding OmniNaija later needs only a line here.
ENGINES = {
    "omnivoice-naija:local-base": {"label": "OmniVoice (base)", "order": 0,
                                   "ref": "k2-fsa/OmniVoice"},
    "omnivoice-naija:local-ohun-ft": {"label": "ohun fine-tune", "order": 1,
                                      "ref": "kapturecx/ohun-omnivoice"},
    "omnivoice-naija:local-omninaija": {"label": "OmniNaija", "order": 2,
                                        "ref": "kapturecx/OmniNaija"},
}
LANGS = {"ohun-ibo": ("ibo", "Igbo", "ig"), "ohun-yor": ("yor", "Yorùbá", "yo"),
         "ohun-hau": ("hau", "Hausa", "ha"), "ohun-pcm": ("pcm", "Nigerian Pidgin", "pcm")}

# metric -> (label, higher_is_better, plain-English meaning)
METRICS = {
    "wer": ("WER", False, "Words the ASR got wrong listening back to the synthesized speech. The core intelligibility number."),
    "cer": ("CER", False, "Same idea at character level — catches near-misses that WER counts as whole-word errors."),
    "sentence_error_rate": ("Sentence ER", False, "Fraction of sentences with at least one error."),
    "mos": ("MOS", True, "Predicted naturalness, 1–5, from a reference-free model. How human it sounds."),
    "speech_rate_deviation": ("Speech-rate dev.", False, "How far the speaking pace sits from a human baseline."),
    "rtfx": ("RTFx", True, "Speed vs realtime. NOTE: unreliable while the GPU is shared with training."),
    "ttfb_ms": ("TTFB", False, "Time to first audio (ms). Also contended during shared-GPU runs."),
}


def collect_runs():
    out = []
    for meta_p in sorted(glob.glob(f"{RUNS}/*/run_meta.json")):
        try:
            m = json.load(open(meta_p))
        except Exception:
            continue
        if m.get("status") != "completed" or not m.get("headline"):
            continue
        out.append(m)
    return out


ART = f"{BASE}/tts-bench/artifacts"
_art_hist = []


def artifact_progress():
    """Synthesis heartbeat: artifact count and rate (files/min)."""
    try:
        n = sum(len(f) for _, _, f in os.walk(ART))
    except OSError:
        return {}
    now = time.time()
    _art_hist.append((now, n))
    while len(_art_hist) > 12:
        _art_hist.pop(0)
    rate = None
    if len(_art_hist) >= 2:
        dt = now - _art_hist[0][0]
        if dt > 30:
            rate = (n - _art_hist[0][1]) / (dt / 60.0)
    return {"artifacts": n, "per_min": round(rate, 1) if rate is not None else None}


def asr_progress():
    """Scoring heartbeat: transcriptions served by the ASR since it started."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8899/health", timeout=5) as r:
            h = json.load(r)
        return {"transcribe_calls": h.get("transcribe_calls"),
                "align_calls": h.get("align_calls")}
    except Exception:
        return {}


def in_flight():
    """Which config is running and how far along."""
    drv = f"{LOGS}/driver.log"
    cur, done = None, []
    if os.path.exists(drv):
        for line in open(drv):
            m = re.search(r"=== (naija-\S+) ===", line)
            if m:
                cur = m.group(1)
            m2 = re.search(r"(naija-\S+) rc=(\d+) \| (\d+) ok", line)
            if m2:
                done.append(m2.group(1))
                if cur == m2.group(1):
                    cur = None
    prog = {}
    if cur:
        lg = f"{LOGS}/{cur}.log"
        try:
            with open(lg, "rb") as f:
                f.seek(max(0, os.path.getsize(lg) - 60000))
                txt = f.read().decode("utf-8", "ignore").replace("\r", "\n")
            syn = list(re.finditer(r"synth\s+\S+\s+.*?(\d+)%\s+(\d+)/(\d+)", txt))
            if syn:
                prog = {"pct": int(syn[-1].group(1)), "done": int(syn[-1].group(2)),
                        "total": int(syn[-1].group(3)), "stage": "synth"}
            sc = list(re.finditer(r"score\s+(\S+)\s+.*?(\d+)%\s+(\d+)/(\d+)", txt))
            if sc and (not syn or sc[-1].start() > syn[-1].start()):
                prog = {"pct": int(sc[-1].group(2)), "done": int(sc[-1].group(3)),
                        "total": int(sc[-1].group(4)), "stage": f"score {sc[-1].group(1)}"}
        except OSError:
            pass
    return cur, done, prog


_marks = {}          # config -> counters at its start


def config_progress(cur, arts, asr_calls, n_target):
    """Samples synthesized / scored within the CURRENT config."""
    if not cur:
        return {}
    if cur not in _marks:
        _marks.clear()
        _marks[cur] = {"art0": arts, "asr0": asr_calls or 0}
    m = _marks[cur]
    synth = max(0, arts - m["art0"])
    scored = max(0, (asr_calls or 0) - m["asr0"])
    stage = "scoring" if synth >= n_target * 0.95 or scored > 0 else "synthesizing"
    done = scored if stage == "scoring" else synth
    return {"stage": stage, "synthesized": min(synth, n_target),
            "scored": min(scored, n_target), "n_target": n_target,
            "pct": round(100 * min(done, n_target) / max(n_target, 1), 1)}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    while True:
        runs = collect_runs()
        # language -> engine -> metric -> value  (latest run wins)
        matrix, seen_engines, cases = {}, set(), {}
        for m in sorted(runs, key=lambda r: r.get("finished_ts") or ""):
            ds = m.get("dataset")
            if ds not in LANGS:
                continue
            cfg, name, code = LANGS[ds]
            n = int(m.get("n_success") or 0)
            for metric, per_engine in (m.get("headline") or {}).items():
                base_metric = metric.split("@")[0]
                if base_metric not in METRICS:
                    continue
                for eng, val in per_engine.items():
                    if eng not in ENGINES:
                        continue
                    seen_engines.add(eng)
                    matrix.setdefault(cfg, {}).setdefault(eng, {})[base_metric] = val
                    cases[(cfg, eng)] = n

        cur, done, prog = in_flight()
        _ap = artifact_progress()
        _sp = asr_progress()
        # sample count comes from the config name suffix (-b100 -> 100)
        n_target = 100 if (cur or "").endswith("-b100") else (
            1000 if (cur or "").endswith("-full") else 20)
        cfg_prog = config_progress(cur, _ap.get("artifacts", 0),
                                   _sp.get("transcribe_calls"), n_target)
        doc = {
            "updated_utc": int(time.time()),
            "engines": {e: ENGINES[e] for e in sorted(
                seen_engines | {"omnivoice-naija:local-base",
                                "omnivoice-naija:local-ohun-ft"},
                key=lambda e: ENGINES[e]["order"])},
            "languages": {c: {"name": n, "code": k}
                          for c, (n2, n, k) in
                          [(v[0], v) for v in LANGS.values()]},
            "metrics": {k: {"label": v[0], "higher_is_better": v[1], "help": v[2]}
                        for k, v in METRICS.items()},
            "matrix": matrix,
            "n_cases": {f"{c}|{e}": v for (c, e), v in cases.items()},
            "progress": {"current": cur, "completed": done,
                         "n_completed": len(done), "n_total": 8, "detail": prog,
                         "synthesis": _ap, "scoring": _sp,
                         "config": cfg_prog},
            "note": ("rtfx/ttfb are measured while the GPU is shared with "
                     "training and should not be compared across runs; "
                     "wer/cer/mos are unaffected."),
        }
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=1)
        os.replace(tmp, OUT)
        time.sleep(30)


if __name__ == "__main__":
    main()
