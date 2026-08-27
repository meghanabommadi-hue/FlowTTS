#!/usr/bin/env python3
"""Find the largest batch_tokens that trains without OOM, empirically.

Guessing this number either wastes the GPU or crashes at 3am. Instead we run a
few REAL steps of the actual training path at each candidate, in a subprocess,
and keep the largest that survives. A subprocess is essential: a CUDA OOM can
leave the allocator in a state that makes in-process retries unreliable.
"""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def try_tokens(base_cfg, data_cfg, n, workdir, env, steps=6):
    cfg = dict(base_cfg)
    cfg["batch_tokens"] = n
    cfg["steps"] = steps
    cfg["eval_steps"] = 10 ** 9      # no eval during the probe
    cfg["save_steps"] = 10 ** 9      # and never write a checkpoint
    cfg["logging_steps"] = 1
    out = os.path.join(workdir, f"probe_{n}")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "cfg.json")
    with open(p, "w") as f:
        json.dump(cfg, f)
    e = dict(env)
    e.pop("OHUN_EVAL_SET", None)     # no audio previews while probing
    e.pop("OHUN_HF_REPO", None)      # and definitely no pushes
    cmd = ["accelerate", "launch", "--num_processes", "1", "--mixed_precision",
           cfg.get("mixed_precision", "bf16"),
           os.path.join(HERE, "train_ohun.py"),
           "--train_config", p, "--data_config", data_cfg, "--output_dir", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=e)
    log = (r.stdout or "") + (r.stderr or "")
    oom = ("CUDA out of memory" in log or "OutOfMemoryError" in log
           or "CUBLAS_STATUS_ALLOC_FAILED" in log)
    peak = None
    for line in log.splitlines():
        if "max_memory_allocated_gb=" in line:
            try:
                peak = float(line.split("max_memory_allocated_gb=")[1].split()[0])
            except Exception:
                pass
    shutil.rmtree(out, ignore_errors=True)
    return (r.returncode == 0 and not oom), oom, peak, log[-1500:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--out", required=True, help="write chosen batch_tokens json here")
    ap.add_argument("--candidates", default="8192,12288,16384,24576,32768,40960")
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()

    base = json.load(open(a.train_config))
    work = a.workdir or tempfile.mkdtemp(prefix="vram_probe_")
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cands = [int(x) for x in a.candidates.split(",")]
    best = None
    for n in cands:
        print(f"\n=== probing batch_tokens={n} ===", flush=True)
        ok, oom, peak, tail = try_tokens(base, a.data_config, n, work, env)
        print(f"  -> ok={ok} oom={oom} peak={peak}", flush=True)
        if ok:
            best = n
        else:
            if oom:
                print("  OOM - stopping the sweep here", flush=True)
                break
            print(f"  non-OOM failure, tail:\n{tail}", flush=True)
            break

    if best is None:
        print("WARNING: no candidate succeeded; falling back to 8192", flush=True)
        best = 8192
    # keep a safety margin: the probe runs a handful of steps and may not have
    # hit the worst-case batch in the length distribution.
    chosen = max(4096, int(best * 0.85) // 1024 * 1024)
    print(f"\nlargest passing={best}  chosen (15% margin)={chosen}")
    with open(a.out, "w") as f:
        json.dump({"batch_tokens": chosen, "largest_passing": best,
                   "candidates": cands}, f, indent=1)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
