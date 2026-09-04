"""GPU worker: speaker diarization (streaming Sortformer, offline preset) + multilingual
Indic ASR (cache-aware streaming FastConformer, batched).  Pulls jobs from the orchestrator.

Environment (gpu.env): CHAASHINI_API_URL, CHAASHINI_API_HOST (optional Host header),
CHAASHINI_INTERNAL_TOKEN, CHAASHINI_MODELS_DIR, CHAASHINI_LOG_DIR, CHAASHINI_ASR_BATCH,
CHAASHINI_GPU_NAME.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Orchestrator, extract_tar, gpu_stats, load_env, setup_logging  # noqa: E402

log = logging.getLogger("chaashini.gpu.asr")


class Models:
    def __init__(self, models_dir: Path, diar_cfg: dict, lookahead: int = 13):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dev == "cuda":
            # This GPU is shared with production services: never let the caching allocator grow past our share.
            frac = float(os.environ.get("CHAASHINI_GPU_MEM_FRACTION", "0.30"))
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            log.info("GPU memory cap: %.0f%% of device", 100 * frac)
        t = time.time()
        from transformers import AutoModel
        self.asr = AutoModel.from_pretrained(str(models_dir / "sravaani-0.5-live"), trust_remote_code=True).to(self.dev).eval()
        left = self.asr.config.att_context_size[0]
        self.asr.set_att_context_size([left, lookahead])
        log.info("ASR loaded in %.1fs (lookahead=%d, chunk=%.2fs)", time.time() - t, lookahead, self.asr.config.chunk_seconds)
        t = time.time()
        from nemo.collections.asr.models import SortformerEncLabelModel
        nemo_path = next((models_dir / "sortformer").glob("*.nemo"))
        self.diar = SortformerEncLabelModel.restore_from(restore_path=str(nemo_path), map_location=self.dev, strict=False)
        self.diar.eval()
        sm = self.diar.sortformer_modules
        sm.chunk_len = diar_cfg["chunk_len"]
        sm.chunk_right_context = diar_cfg["right_context"]
        sm.fifo_len = diar_cfg["fifo_len"]
        sm.spkcache_update_period = diar_cfg["update_period"]
        sm.spkcache_len = diar_cfg["spkcache_len"]
        sm._check_streaming_parameters()
        log.info("diarizer loaded in %.1fs", time.time() - t)

    # ------------------------------------------------------------------ diarization
    def diarize(self, wav16: np.ndarray) -> tuple[list[list], np.ndarray]:
        segs, probs = self.diar.diarize(audio=[wav16.astype(np.float32)], batch_size=1, sample_rate=16000, include_tensor_outputs=True)
        p = probs[0]
        if hasattr(p, "detach"):
            p = p.detach().float().cpu().numpy()
        p = np.asarray(p)
        if p.ndim == 3:
            p = p[0]
        out = []
        for s in segs[0]:
            a, b, spk = s.split()
            out.append([float(a), float(b), int(spk.replace("speaker_", ""))])
        return out, p.astype(np.float16)

    # ------------------------------------------------------------------ ASR (batched streaming)
    TAIL_SILENCE_S = 2.4   # > lookahead (1.04 s) + one chunk (1.12 s): flushes the last word of every clip

    def transcribe_batch(self, wavs: list[np.ndarray]) -> list[str]:
        """Batched cache-aware streaming decode.

        Every clip gets `TAIL_SILENCE_S` of digital silence appended so the encoder's right
        context is satisfied for the final frames (the stock single-stream path silently drops
        the last ~1 s of speech). Tokens are cut once the stream has passed the clip's real end
        plus lookahead + one chunk, so nothing decoded from padding leaks into the transcript.
        """
        torch = self.torch
        m = self.asr
        cfg = m.config
        if not wavs:
            return []
        sr = cfg.sample_rate
        sil = int(self.TAIL_SILENCE_S * sr)
        real = [len(w) for w in wavs]
        lens = [n + sil for n in real]
        B, L = len(wavs), max(lens)
        batch = torch.zeros(B, L, dtype=torch.float32)
        for i, w in enumerate(wavs):
            batch[i, : len(w)] = torch.as_tensor(w, dtype=torch.float32)
        feats, flen = m.extract_features(batch, torch.tensor(lens))
        hop = int(m._p["hop_length"])
        la_frames = int(cfg.att_context_size[1]) * int(cfg.subsampling_factor)
        stop_at = [n // hop + la_frames + cfg.chunk_size for n in real]
        T = int(flen.max().item())
        state = m.new_stream(B)
        cut: list[int | None] = [None] * B
        idx = 0
        while idx < T:
            n_new = min(cfg.chunk_size, T - idx)
            if n_new < cfg.min_sampling_frames:
                break
            x, _ = m._chunk_with_cache(feats[:, :, :T], idx)
            m.stream_features(state, x, torch.full((B,), x.shape[-1], dtype=torch.long))
            idx += cfg.shift_size
            for b in range(B):
                if cut[b] is None and idx >= stop_at[b]:
                    cut[b] = len(state.tokens[b])
        tok = m._get_tokenizer()
        out = []
        for b in range(B):
            toks = state.tokens[b] if cut[b] is None else state.tokens[b][: cut[b]]
            out.append(tok.decode(toks).strip())
        return out

    def transcribe_single(self, wav: np.ndarray) -> str:
        sr = self.asr.config.sample_rate
        padded = np.concatenate([wav.astype(np.float32), np.zeros(int(self.TAIL_SILENCE_S * sr), dtype=np.float32)])
        return self.asr.transcribe([padded])[0].strip()

    def transcribe_many(self, items: list[tuple[str, np.ndarray]], batch_size: int = 16) -> dict[str, str]:
        order = sorted(range(len(items)), key=lambda i: len(items[i][1]))
        res: dict[str, str] = {}
        for i in range(0, len(order), batch_size):
            ids = order[i: i + batch_size]
            wavs = [items[k][1] for k in ids]
            try:
                texts = self.transcribe_batch(wavs)
            except Exception as e:  # noqa: BLE001
                log.warning("batch failed (%s); falling back to single", e)
                texts = []
                for w in wavs:
                    try:
                        texts.append(self.transcribe_single(w))
                    except Exception as e2:  # noqa: BLE001
                        log.warning("single failed: %s", e2)
                        texts.append("")
            for k, t in zip(ids, texts):
                res[items[k][0]] = t
        return res


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    return x, sr


def handle_diarize(models: Models, payload: Path, tmp: Path) -> Path:
    wav, sr = read_wav(payload)
    assert sr == 16000, f"expected 16 kHz, got {sr}"
    segs, probs = models.diarize(wav)
    out = tmp / "diar_result.npz"
    np.savez_compressed(out, probs=probs, segments=np.array(segs, dtype=np.float32), frame_ms=np.array(80), n_samples=np.array(len(wav)))
    return out


def handle_transcribe(models: Models, payload: Path, tmp: Path, batch: int) -> Path:
    d = tmp / "asr_in"
    extract_tar(payload, d)
    manifest = json.loads((d / "manifest.json").read_text())
    items = []
    for it in manifest["items"]:
        p = d / it["file"]
        try:
            x, sr = read_wav(p)
            if sr != 16000:
                raise ValueError(f"expected 16 kHz, got {sr}")
            items.append((it["id"], x))
        except Exception as e:  # noqa: BLE001
            log.warning("bad item %s: %s", it, e)
    res = models.transcribe_many(items, batch)
    out = tmp / "transcribe_result.json"
    out.write_text(json.dumps({"texts": res}, ensure_ascii=False))
    return out


def main() -> None:
    env = load_env()
    for k, v in env.items():
        if k.startswith("CHAASHINI_"):
            os.environ.setdefault(k, v)
    setup_logging("asr_worker", env.get("CHAASHINI_LOG_DIR", "/opt/chaashini/logs"))
    api = Orchestrator(env["CHAASHINI_API_URL"], env["CHAASHINI_INTERNAL_TOKEN"], env.get("CHAASHINI_API_HOST") or None,
                       worker=env.get("CHAASHINI_GPU_NAME", "gpu-asr"))
    models_dir = Path(env.get("CHAASHINI_MODELS_DIR", "/opt/chaashini/models"))
    batch = int(env.get("CHAASHINI_ASR_BATCH", "16"))
    diar_cfg = {"chunk_len": int(env.get("CHAASHINI_DIAR_CHUNK", 340)), "right_context": int(env.get("CHAASHINI_DIAR_RC", 40)),
                "fifo_len": int(env.get("CHAASHINI_DIAR_FIFO", 40)), "update_period": int(env.get("CHAASHINI_DIAR_UPD", 300)),
                "spkcache_len": int(env.get("CHAASHINI_DIAR_CACHE", 188))}
    idle = float(env.get("CHAASHINI_IDLE_POLL", 3))
    while not api.health():
        log.warning("orchestrator unreachable at %s; retrying", env["CHAASHINI_API_URL"])
        time.sleep(10)
    models = Models(models_dir, diar_cfg, lookahead=int(env.get("CHAASHINI_ASR_LOOKAHEAD", 13)))
    stats = {"jobs": 0, "diarize_s": 0.0, "transcribe_s": 0.0, "audio_s": 0.0, "errors": 0}
    last_hb = 0.0
    while True:
        try:
            if time.time() - last_hb > 10:
                api.heartbeat("gpu-asr", "idle", stats={**stats, **gpu_stats()})
                last_hb = time.time()
            job = api.claim(["diarize", "transcribe"])
            if not job:
                time.sleep(idle)
                continue
            jid, kind = job["id"], job["kind"]
            t0 = time.time()
            api.heartbeat("gpu-asr", f"running:{kind}", current=f"{kind} job {jid} ({job.get('video_id')})", stats={**stats, **gpu_stats()})
            with tempfile.TemporaryDirectory(prefix="chaashini-") as td:
                tmp = Path(td)
                payload = tmp / ("payload.wav" if kind == "diarize" else "payload.tar")
                api.download_payload(jid, payload)
                api.job_heartbeat(jid)
                try:
                    if kind == "diarize":
                        out = handle_diarize(models, payload, tmp)
                    else:
                        out = handle_transcribe(models, payload, tmp, batch)
                    dt = time.time() - t0
                    api.complete(jid, True, out, proc_seconds=dt)
                    if models.dev == "cuda":
                        models.torch.cuda.empty_cache()      # hand reserved memory back to the other tenants
                    stats["jobs"] += 1
                    stats[f"{kind}_s"] += dt
                    stats["audio_s"] += float(job.get("audio_seconds") or 0)
                    log.info("%s job %d done in %.1fs (audio %.0fs)", kind, jid, dt, float(job.get("audio_seconds") or 0))
                except Exception as e:  # noqa: BLE001
                    stats["errors"] += 1
                    log.error("%s job %d failed: %s\n%s", kind, jid, e, traceback.format_exc())
                    api.complete(jid, False, None, error=f"{type(e).__name__}: {e}", proc_seconds=time.time() - t0)
                    if "CUDA" in str(e) or "out of memory" in str(e).lower():
                        models.torch.cuda.empty_cache()
        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001
            log.error("loop error: %s", e)
            time.sleep(10)


if __name__ == "__main__":
    main()
