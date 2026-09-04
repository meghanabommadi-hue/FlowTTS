"""GPU worker: speech enhancement (denoise + enhance) for borderline chunks. Pulls `enhance`
jobs from the orchestrator; each job is a tar of 44.1 kHz WAVs + manifest.json."""
from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HeartbeatThread, Orchestrator, extract_tar, gpu_stats, load_env, make_tar, setup_logging  # noqa: E402

log = logging.getLogger("chaashini.gpu.enhance")


import os

class Enhancer:
    def __init__(self, run_dir: Path):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dev == "cuda":
            frac = float(os.environ.get("CHAASHINI_GPU_ENH_MEM_FRACTION", "0.22"))
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            log.info("GPU memory cap: %.0f%% of device", 100 * frac)
        self.run_dir = run_dir
        from resemble_enhance.enhancer.inference import load_enhancer
        from resemble_enhance.inference import inference
        self._inference = inference
        t = time.time()
        self.model = load_enhancer(run_dir, self.dev)
        # shorter inference windows bound peak memory on the shared GPU (default is 30 s)
        self.chunk_seconds = float(os.environ.get("CHAASHINI_ENH_CHUNK_S", "8"))
        log.info("enhancer loaded in %.1fs (inference window %.0fs)", time.time() - t, self.chunk_seconds)

    def run(self, wav, sr: int, mode: str, nfe: int, solver: str, lambd: float, tau: float):
        import numpy as np
        # the enhancer's chunker misbehaves on very short inputs: pad to >= 1.5 s, trim afterwards
        min_len = int(1.5 * sr)
        pad = max(0, min_len - len(wav))
        if pad:
            wav = np.concatenate([wav, np.zeros(pad, dtype=wav.dtype)])
        x = self.torch.from_numpy(wav)
        with self.torch.inference_mode():
            if mode == "denoise":
                y, osr = self._inference(model=self.model.denoiser, dwav=x, sr=sr, device=self.dev,
                                         chunk_seconds=self.chunk_seconds, overlap_seconds=1.0)
            else:
                self.model.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
                y, osr = self._inference(model=self.model, dwav=x, sr=sr, device=self.dev,
                                         chunk_seconds=self.chunk_seconds, overlap_seconds=1.0)
        y = y.detach().cpu().numpy()
        if pad:
            keep = int(round((len(wav) - pad) * int(osr) / sr))
            y = y[:keep]
        return y, int(osr)


def handle(enh: Enhancer, payload: Path, tmp: Path, api: Orchestrator, jid: int) -> Path:
    import soundfile as sf
    d = tmp / "enh_in"
    extract_tar(payload, d)
    manifest = json.loads((d / "manifest.json").read_text())
    p = manifest.get("params", {})
    mode, nfe, solver = p.get("mode", "enhance"), int(p.get("nfe", 32)), p.get("solver", "midpoint")
    lambd, tau = float(p.get("lambd", 0.5)), float(p.get("tau", 0.5))
    out_dir = tmp / "enh_out"
    out_dir.mkdir()
    outs, names, done = [], [], []
    last_hb = time.time()
    for it in manifest["items"]:
        try:
            x, sr = sf.read(str(d / it["file"]), dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x[:, 0]
            y, osr = enh.run(x, sr, mode, nfe, solver, lambd, tau)
            op = out_dir / f"{it['id']}.wav"
            sf.write(str(op), y, osr, subtype="PCM_16")
            outs.append(op)
            names.append(op.name)
            done.append({"id": it["id"], "file": op.name, "sr": osr})
        except Exception as e:  # noqa: BLE001
            log.warning("enhance item %s failed: %s", it.get("id"), e)
            if "out of memory" in str(e).lower():
                enh.torch.cuda.empty_cache()
        if time.time() - last_hb > 60:
            api.job_heartbeat(jid)
            last_hb = time.time()
    (out_dir / "manifest.json").write_text(json.dumps({"items": done}))
    outs.append(out_dir / "manifest.json")
    names.append("manifest.json")
    res = tmp / "enhance_result.tar"
    make_tar(outs, res, names)
    return res


def main() -> None:
    env = load_env()
    for k, v in env.items():
        if k.startswith("CHAASHINI_"):
            os.environ.setdefault(k, v)
    setup_logging("enhance_worker", env.get("CHAASHINI_LOG_DIR", "/opt/chaashini/logs"))
    api = Orchestrator(env["CHAASHINI_API_URL"], env["CHAASHINI_INTERNAL_TOKEN"], env.get("CHAASHINI_API_HOST") or None,
                       worker=env.get("CHAASHINI_GPU_ENH_NAME", "gpu-enhance"))
    run_dir = Path(env.get("CHAASHINI_ENHANCE_RUN_DIR", "/opt/chaashini/models/resemble-enhance/enhancer_stage2"))
    idle = float(env.get("CHAASHINI_IDLE_POLL", 5))
    while not api.health():
        log.warning("orchestrator unreachable; retrying")
        time.sleep(10)
    hb = HeartbeatThread(api, "gpu-enhance").start()
    enh = Enhancer(run_dir)
    stats = {"jobs": 0, "items": 0, "enhance_s": 0.0, "errors": 0}
    last_hb = 0.0
    while True:
        try:
            hb.set("idle", None, stats)
            job = api.claim(["enhance"])
            if not job:
                time.sleep(idle)
                continue
            jid = job["id"]
            t0 = time.time()
            hb.set("running:enhance", f"enhance job {jid} ({job.get('video_id')}, {job.get('n_items')} items)", stats)
            api.heartbeat("gpu-enhance", hb.state, hb.current, {**stats, **gpu_stats()})
            with tempfile.TemporaryDirectory(prefix="chaashini-enh-") as td:
                tmp = Path(td)
                payload = tmp / "payload.tar"
                api.download_payload(jid, payload)
                try:
                    out = handle(enh, payload, tmp, api, jid)
                    dt = time.time() - t0
                    api.complete(jid, True, out, proc_seconds=dt)
                    if enh.dev == "cuda":
                        enh.torch.cuda.empty_cache()
                    stats["jobs"] += 1
                    stats["items"] += int(job.get("n_items") or 0)
                    stats["enhance_s"] += dt
                    log.info("enhance job %d done in %.1fs (%s items)", jid, dt, job.get("n_items"))
                except Exception as e:  # noqa: BLE001
                    stats["errors"] += 1
                    log.error("enhance job %d failed: %s\n%s", jid, e, traceback.format_exc())
                    api.complete(jid, False, None, error=f"{type(e).__name__}: {e}", proc_seconds=time.time() - t0)
        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001
            log.error("loop error: %s", e)
            time.sleep(10)


if __name__ == "__main__":
    main()
