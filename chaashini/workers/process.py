"""Process worker: everything CPU-side between "downloaded" and "done".

    downloaded  -> decode  -> [source gate] -> diarize job  (diarize_queued)
    diarized    -> segment -> per-chunk scoring -> enhance job / transcribe job
    enhanced    -> rescore -> transcribe job
    transcribed -> finalize: text checks, LID, export FLAC + metadata, cleanup
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import db as D
from ..audio import decode_to_wav, peak_normalize, read_wav_int16, resample
from ..dedup import clip_fingerprint, duplicate_audio, duplicate_clip, fingerprint, remember_clip
from ..export import stage_chunk
from ..languages import LANGUAGES
from ..lid import identify
from ..llm import india_cue_count
from ..quality import (get_dnsmos, get_tagger, noise_floor_from_vad, signal_metrics, window_stat, bandwidth_hz, clipping_ratio)
from ..segment import diar_frames, make_chunks
from ..vad import VADFrames, run_vad, speech_regions
from .base import Worker, free_gb

log = logging.getLogger("chaashini.process")


def _cpu_percent() -> float:
    """Box-wide CPU utilisation over a short window (psutil), 0 if unavailable."""
    try:
        import psutil
        return float(psutil.cpu_percent(interval=0.5))
    except Exception:  # noqa: BLE001
        return 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ProcessWorker(Worker):
    kind = "process"

    def __init__(self, name: str, cfg=None):
        super().__init__(name, cfg)
        import torch
        torch.set_num_threads(self.cfg.workers.torch_threads)
        self.stats = {"decoded": 0, "segmented": 0, "rescored": 0, "finalized": 0, "chunks_accepted": 0, "chunks_rejected": 0,
                      "accepted_sec": 0.0, "videos_rejected": 0}
        self._enabled_codes = {l.code for l in self.cfg.enabled_languages()}
        self.dnsmos = get_dnsmos(self.cfg.paths.models_dir / "dnsmos", threads=2)
        self.tagger = get_tagger(threads=self.cfg.workers.torch_threads)

    def idle_sleep(self) -> float:
        return 4.0

    # ------------------------------------------------------------------ dispatch
    def step(self) -> bool:
        fg = free_gb(self.cfg.paths.data_dir)
        if fg < self.cfg.storage.min_free_gb * 0.5:
            self.heartbeat("paused: low disk", f"{fg:.0f} GB free", force=True)
            return False
        cpu = _cpu_percent()
        if cpu > self.cfg.workers.cpu_max_percent:
            self.heartbeat("paused: cpu buffer", f"box at {cpu:.0f}%", force=True)
            return False
        lease = self.cfg.workers.lease_s
        for from_s, to_s, fn in (("transcribed", "finalizing", self.finalize), ("enhanced", "rescoring", self.rescore),
                                 ("diarized", "segmenting", self.segment), ("downloaded", "decoding", self.decode)):
            v = D.claim_video(self.conn, from_s, to_s, self.name, lease)
            if v:
                self.heartbeat(to_s, f"{v['id']} [{v['lang_hint']}] {(v['title'] or '')[:50]}", force=True)
                t0 = time.time()
                try:
                    fn(v)
                except InterruptedError:
                    D.set_video_status(self.conn, v["id"], from_s)
                    self.conn.execute("UPDATE videos SET attempts=MAX(0, attempts-1) WHERE id=?", (v["id"],))
                    log.info("%s of %s abandoned for shutdown; will be redone", to_s, v["id"])
                    return True
                except Exception as e:  # noqa: BLE001
                    log.exception("%s failed for %s: %s", to_s, v["id"], e)
                    if v["attempts"] >= 3:
                        D.set_video_status(self.conn, v["id"], "failed", error=f"{to_s}: {type(e).__name__}: {str(e)[:300]}")
                        self._cleanup(v)
                    else:
                        D.set_video_status(self.conn, v["id"], from_s, error=f"{to_s} retry: {type(e).__name__}: {str(e)[:300]}")
                        self.conn.execute("UPDATE videos SET attempts=? WHERE id=?", (v["attempts"], v["id"]))
                log.info("%s %s done in %.1fs", to_s, v["id"], time.time() - t0)
                return True
        return False

    # ------------------------------------------------------------------ helpers
    def wd(self, v) -> Path:
        d = Path(v["work_dir"] or (self.cfg.paths.work_dir / v["id"]))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _cleanup(self, v) -> None:
        d = Path(v["work_dir"] or (self.cfg.paths.work_dir / v["id"]))
        shutil.rmtree(d, ignore_errors=True)

    def _reject_video(self, v, reason: str) -> None:
        D.set_video_status(self.conn, v["id"], "rejected", error=reason)
        self.stats["videos_rejected"] += 1
        self._touch_channel(v, 0.0, v["duration_s"] or 0.0)
        self._cleanup(v)
        self.event("video", f"rejected {v['id']} [{v['lang_hint']}]: {reason}", data={"title": v["title"]})

    def _touch_channel(self, v, accepted_sec: float, source_sec: float) -> None:
        if v["channel_id"]:
            self.conn.execute(
                "INSERT INTO channels(id, name, lang_hint, videos_done, accepted_sec, source_sec, updated_at) VALUES (?,?,?,1,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET videos_done=videos_done+1, accepted_sec=accepted_sec+excluded.accepted_sec, "
                "source_sec=source_sec+excluded.source_sec, updated_at=excluded.updated_at",
                (v["channel_id"], v["channel"], v["lang_hint"], accepted_sec, source_sec, time.time()))
        if v["query_id"]:
            self.conn.execute("UPDATE queries SET accepted_sec=accepted_sec+? WHERE id=?", (accepted_sec, v["query_id"]))

    def _load_master(self, wd: Path) -> tuple[np.ndarray, int]:
        return read_wav_int16(wd / "audio16k.wav")

    def _load_export_master(self, wd: Path, v=None) -> tuple[np.ndarray, int]:
        p = wd / f"audio{self.cfg.audio.export_sr // 1000}k.wav"
        if not p.exists():
            src = None
            if v is not None and v["src_path"] and os.path.exists(v["src_path"]):
                src = v["src_path"]
            else:
                cands = [q for q in wd.iterdir() if q.suffix in (".webm", ".m4a", ".mp4", ".opus", ".mka", ".ogg")]
                src = str(cands[0]) if cands else None
            if not src:
                raise RuntimeError("export master and source both missing")
            decode_to_wav(src, p, self.cfg.audio.export_sr)
        x, sr = read_wav_int16(p)
        return x, sr

    @staticmethod
    def _slice(x: np.ndarray, sr: int, start_ms: int, end_ms: int) -> np.ndarray:
        a, b = int(start_ms * sr / 1000), int(end_ms * sr / 1000)
        return x[a:b].astype(np.float32) / 32768.0

    def _load_vad(self, wd: Path, n_samples: int) -> VADFrames:
        probs = np.load(wd / "vad.npy")
        return VADFrames(probs, (probs >= self.cfg.vad.threshold).astype(np.uint8), self.cfg.vad.hop, self.cfg.audio.analysis_sr)

    # ------------------------------------------------------------------ stage 1: decode + gate + VAD
    def decode(self, v) -> None:
        """Decode the source to a 16 kHz analysis master and an export-rate master, then screen it.

        Idempotent: a retry (lease expiry, restart) reuses masters that already exist, because the
        compressed source is deleted once the video has moved on and cannot be decoded a second time.
        """
        wd = self.wd(v)
        src = v["src_path"]
        master = wd / "audio16k.wav"
        export_master = wd / f"audio{self.cfg.audio.export_sr // 1000}k.wav"
        have_masters = master.exists() and master.stat().st_size > 1000 and export_master.exists() and export_master.stat().st_size > 1000
        if have_masters:
            dur = sf.info(str(master)).duration
            sr = self.cfg.audio.analysis_sr
            log.info("decode %s: reusing masters from a previous attempt (%.0fs)", v["id"], dur)
        else:
            if not src or not os.path.exists(src):
                D.set_video_status(self.conn, v["id"], "failed", error="source file missing and no usable masters")
                self._cleanup(v)
                return
            dur, sr = decode_to_wav(src, master, self.cfg.audio.analysis_sr)
            decode_to_wav(src, export_master, self.cfg.audio.export_sr)
        x16, sr = self._load_master(wd)
        fp = fingerprint(x16, sr)
        self.conn.execute("UPDATE videos SET fp=? WHERE id=?", (fp, v["id"]))
        dup = duplicate_audio(self.conn, v["id"], fp, dur)
        if dup:
            self._reject_video(v, f"duplicate_audio of {dup}")
            return
        # VAD over the full file
        vad = run_vad(x16, sr, self.cfg.vad.hop, self.cfg.vad.threshold)
        np.save(wd / "vad.npy", vad.probs)
        speech_ratio = float((vad.probs >= self.cfg.vad.threshold).mean()) if len(vad.probs) else 0.0
        # coarse source-level gate with the tagger on evenly spaced windows
        n_win = self.cfg.quality.video_gate_windows
        xf = x16.astype(np.float32) / 32768.0
        starts = np.linspace(0, max(0.0, dur - 4.0), n_win)
        win32 = []
        for s in starts:
            a = int(s * sr)
            seg = xf[a: a + 4 * sr]
            if len(seg) < 4 * sr:
                seg = np.pad(seg, (0, 4 * sr - len(seg)))
            win32.append(resample(seg, sr, self.tagger.SR))
        tags = self.tagger.tag_windows(np.concatenate(win32), win_s=4.0, hop_s=4.0)
        music_mean = float(np.mean(tags["music"])) if len(tags["music"]) else 0.0
        music_frac = float(np.mean(tags["music"] > 0.5)) if len(tags["music"]) else 0.0
        speech_mean = float(np.mean(tags["speech"])) if len(tags["speech"]) else 0.0
        clip_src = clipping_ratio(xf)
        gate = {"speech_ratio": round(speech_ratio, 3), "music_mean": round(music_mean, 3), "music_frac": round(music_frac, 3),
                "speech_mean": round(speech_mean, 3), "clipping_ratio": round(clip_src, 5)}
        self.stats["decoded"] += 1
        if clip_src > self.cfg.quality.accept.clipping_max * 3:
            self._reject_video(v, f"clipped_source ({100 * clip_src:.2f}% flat-topped samples)")
            return
        if music_frac >= self.cfg.quality.video_music_gate:
            self._reject_video(v, f"mostly_music ({music_frac:.0%} of windows)")
            return
        if speech_mean < 0.3:
            self._reject_video(v, f"not_speech (speech tag {speech_mean:.2f})")
            return
        if speech_ratio < 0.15:
            self._reject_video(v, f"little_speech (VAD {speech_ratio:.0%})")
            return
        job = D.enqueue_job(self.conn, "diarize", v["id"], str(master), payload_bytes=master.stat().st_size, audio_seconds=dur, n_items=1)
        D.set_video_status(self.conn, v["id"], "diarize_queued", stats_json={"gate": gate, "duration_s": dur})
        if src and os.path.exists(src):
            try:
                os.remove(src)      # only now: both masters exist and the state has advanced, so a retry cannot need it
            except OSError:
                pass
        log.info("decoded %s: %.0fs, VAD %.0f%%, music %.2f, speech %.2f -> diarize job %d", v["id"], dur, 100 * speech_ratio, music_mean, speech_mean, job)

    # ------------------------------------------------------------------ stage 2: segment + score
    def _tag_file(self, x16: np.ndarray, sr: int) -> dict:
        """Tag the whole recording in 4 s windows, in 10-minute blocks to bound memory."""
        block = 600 * sr
        parts = []
        for a in range(0, len(x16), block):
            seg = x16[a: a + block + 4 * sr].astype(np.float32) / 32768.0
            if len(seg) < sr:
                continue
            t = self.tagger.tag_windows(resample(seg, sr, self.tagger.SR), win_s=4.0, hop_s=4.0)
            t["start_s"] = t["start_s"] + a / sr
            parts.append(t)
        if not parts:
            return {"start_s": np.zeros(0), "win_s": 4.0, "music": np.zeros(0), "speech": np.zeros(0), "noise": np.zeros(0), "singing": np.zeros(0)}
        return {"start_s": np.concatenate([p["start_s"] for p in parts]), "win_s": 4.0,
                **{k: np.concatenate([p[k] for p in parts]) for k in ("music", "speech", "noise", "singing")}}

    def _score_chunk(self, x16f: np.ndarray, sr: int, vad_probs: np.ndarray | None, noise_floor: float | None, tags: dict | None,
                     start_s: float, end_s: float) -> dict:
        d = self.dnsmos.score(x16f, sr)
        sm = signal_metrics(x16f, sr, vad_probs, self.cfg.vad.hop, noise_floor)
        m = {"dnsmos_sig": round(d.sig, 3), "dnsmos_bak": round(d.bak, 3), "dnsmos_ovrl": round(d.ovrl, 3), "dnsmos_p808": round(d.p808, 3),
             **sm.as_dict()}
        if tags is not None:
            m["music_prob"] = round(window_stat(tags, start_s, end_s, "music", np.max), 4)
            m["singing_prob"] = round(window_stat(tags, start_s, end_s, "singing", np.max), 4)
            m["speech_prob"] = round(window_stat(tags, start_s, end_s, "speech", np.mean), 4)
            m["noise_prob"] = round(window_stat(tags, start_s, end_s, "noise", np.max), 4)
        return m

    def _decide(self, m: dict, ch_info: dict) -> tuple[str, str]:
        """Return (status, reason) with status in accepted|enhance|rejected."""
        A, E = self.cfg.quality.accept, self.cfg.quality.enhance
        if ch_info["dominance"] < self.cfg.diar.min_dominance:
            return "rejected", "multi_speaker"
        if ch_info["diar_coverage"] < 0.5:
            return "rejected", "diar_no_speaker"
        if ch_info["vad_ratio"] < A.vad_ratio_min:
            return "rejected", "low_speech_ratio"
        if m["clipping_ratio"] > A.clipping_max:
            return "rejected", "clipping"
        if m["rms_dbfs"] < A.rms_dbfs_min:
            return "rejected", "too_quiet"
        if m["rms_dbfs"] > A.rms_dbfs_max:
            return "rejected", "too_loud"
        if m.get("singing_prob", 0) > 0.5:
            return "rejected", "singing"
        if m.get("music_prob", 0) > E.music_prob_max:
            return "rejected", "music"
        if m.get("speech_prob", 1) < A.speech_prob_min:
            return "rejected", "not_speech"
        ok = (m["dnsmos_ovrl"] >= A.ovrl_min and m["dnsmos_sig"] >= A.sig_min and m["dnsmos_bak"] >= A.bak_min and
              m["dnsmos_p808"] >= A.p808_min and m.get("music_prob", 0) <= A.music_prob_max and m["snr_db"] >= A.snr_db_min)
        if ok:
            return "accepted", ""
        # which gate failed (for the reason)
        if m.get("music_prob", 0) > A.music_prob_max:
            reason = "bgm"
        elif m["dnsmos_bak"] < A.bak_min:
            reason = "background_noise"
        elif m["snr_db"] < A.snr_db_min:
            reason = "low_snr"
        elif m["dnsmos_sig"] < A.sig_min:
            reason = "signal_quality"
        elif m["dnsmos_ovrl"] < A.ovrl_min:
            reason = "overall_quality"
        else:
            reason = "p808"
        # Enhancement only pays off when the problem is the BACKGROUND (noise / low SNR / faint bgm) and the
        # speech itself is healthy; poor signal quality is not fixable and would just burn GPU time.
        background_issue = (m["dnsmos_bak"] < A.bak_min or m["snr_db"] < A.snr_db_min or m.get("music_prob", 0) > A.music_prob_max)
        if E.enabled and background_issue and ch_info.get("dur_ms", 0) >= E.min_chunk_ms and m["dnsmos_ovrl"] >= E.ovrl_min \
                and m["dnsmos_sig"] >= E.sig_min and m["snr_db"] >= E.snr_db_min and m.get("music_prob", 0) <= E.music_prob_max:
            return "enhance", reason
        return "rejected", reason

    def _write_asr_payload(self, wd: Path, x16: np.ndarray, sr: int, chunk_rows: list, enh_dir: Path | None) -> tuple[Path, int, float]:
        """Tar of 16 kHz WAVs (+manifest) for all candidate chunks. Enhanced chunks come from enh_dir."""
        tar_path = wd / "asr_in.tar"
        items = []
        total = 0.0
        with tarfile.open(tar_path, "w") as tf:
            for c in chunk_rows:
                cid = c["id"]
                if c["enhanced"] and enh_dir is not None and (enh_dir / f"{cid}.16k.wav").exists():
                    p = enh_dir / f"{cid}.16k.wav"
                    tf.add(p, arcname=f"{cid}.wav")
                else:
                    a, b = int(c["start_ms"] * sr / 1000), int(c["end_ms"] * sr / 1000)
                    buf = io.BytesIO()
                    sf.write(buf, x16[a:b], sr, format="WAV", subtype="PCM_16")
                    data = buf.getvalue()
                    ti = tarfile.TarInfo(name=f"{cid}.wav")
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
                items.append({"id": cid, "file": f"{cid}.wav"})
                total += c["dur_ms"] / 1000.0
            man = json.dumps({"items": items}).encode()
            ti = tarfile.TarInfo(name="manifest.json")
            ti.size = len(man)
            tf.addfile(ti, io.BytesIO(man))
        return tar_path, len(items), total

    def segment(self, v) -> None:
        wd = self.wd(v)
        vid = v["id"]
        x16, sr = self._load_master(wd)
        vad = self._load_vad(wd, len(x16))
        res = wd / "diarize_result.npz"
        if not res.exists():
            raise RuntimeError("diarization result missing")
        z = np.load(res)
        probs = z["probs"].astype(np.float32)
        frame_ms = int(z["frame_ms"]) if "frame_ms" in z else self.cfg.diar.frame_ms
        total_ms = int(len(x16) * 1000 / sr)
        regions = speech_regions(vad, self.cfg.vad.threshold, self.cfg.vad.min_speech_ms, self.cfg.vad.merge_gap_ms, self.cfg.vad.pad_ms)
        dom, n_active = diar_frames(probs, frame_ms, total_ms, self.cfg.diar.active_prob, vad.frame_ms)
        chunks = make_chunks(vad, regions, dom, n_active, self.cfg.vad, self.cfg.diar)
        n_spk = int(len(set(c.speaker for c in chunks if c.speaker >= 0)))
        if not chunks:
            self._finish_video(v, [], {"reason": "no_chunks"})
            return
        tags = self._tag_file(x16, sr)
        noise_floor = noise_floor_from_vad(x16.astype(np.float32) / 32768.0, vad.probs, self.cfg.vad.hop)
        xf = x16.astype(np.float32) / 32768.0
        t = time.time()
        rows = []
        max_enh = min(self.cfg.quality.enhance.max_chunks_per_video, int(self.cfg.quality.enhance.max_fraction_per_video * len(chunks)) + 1)
        counts = {"accepted": 0, "enhance": 0, "rejected": 0}
        reasons: dict[str, int] = {}
        for i, c in enumerate(chunks):
            a, b = int(c.start_ms * sr / 1000), int(c.end_ms * sr / 1000)
            seg = xf[a:b]
            fa, fb = int(c.start_ms / vad.frame_ms), int(c.end_ms / vad.frame_ms)
            m = self._score_chunk(seg, sr, vad.probs[fa:fb], noise_floor, tags, c.start_ms / 1000, c.end_ms / 1000)
            m["vad_speech_ratio"] = round(c.vad_ratio, 3)
            m["speaker_dominance"] = round(c.dominance, 3)
            m["diar_coverage"] = round(c.diar_coverage, 3)
            status, reason = self._decide(m, {"dominance": c.dominance, "diar_coverage": c.diar_coverage, "vad_ratio": c.vad_ratio, "dur_ms": c.dur_ms})
            if status == "accepted":
                status = "candidate"
            cid = f"{v['source_hash']}_{i:04d}"
            rows.append([cid, vid, i, c.start_ms, c.end_ms, c.dur_ms, c.speaker, status, reason, D.j(m), time.time(), time.time()])
            if i % 50 == 0:
                if self.stop:
                    raise InterruptedError("stop requested")
                self.heartbeat("segmenting", f"{vid} chunk {i}/{len(chunks)}")
                D.touch_lease(self.conn, vid, self.cfg.workers.lease_s)
        # enhancement budget: keep the longest borderline chunks (most audio per GPU second), reject the rest
        enh_idx = sorted((k for k, r in enumerate(rows) if r[7] == "enhance"), key=lambda k: -rows[k][5])
        for k in enh_idx[max_enh:]:
            rows[k][7] = "rejected"
        for r in rows:
            st = r[7]
            counts["accepted" if st == "candidate" else st] += 1
            if st == "rejected":
                reasons[r[8]] = reasons.get(r[8], 0) + 1
            else:
                r[8] = r[8] or None
        rows = [tuple(r) for r in rows]
        with D.tx(self.conn):
            self.conn.execute("DELETE FROM chunks WHERE video_id=?", (vid,))
            self.conn.executemany(
                "INSERT INTO chunks(id, video_id, idx, start_ms, end_ms, dur_ms, speaker, status, reject_reason, metrics_json, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.stats["segmented"] += 1
        stats = D.uj(v["stats_json"], {}) or {}
        stats.update({"n_chunks": len(chunks), "n_speakers": n_spk, "segment": counts, "reject_reasons": reasons,
                      "score_s": round(time.time() - t, 1)})
        log.info("segmented %s: %d chunks (%d spk): %s in %.0fs", vid, len(chunks), n_spk, counts, time.time() - t)
        cand = self.conn.execute("SELECT * FROM chunks WHERE video_id=? AND status IN ('candidate','enhance') ORDER BY idx", (vid,)).fetchall()
        if not cand:
            self._finish_video(v, [], stats)
            return
        enh_rows = [c for c in cand if c["status"] == "enhance"]
        if enh_rows:
            queued = self.conn.execute("SELECT COALESCE(SUM(audio_seconds),0) s FROM gpu_jobs WHERE kind='enhance' AND status IN ('queued','running')").fetchone()["s"]
            if queued > self.cfg.quality.enhance.max_queue_s:
                # the enhancer must never hold the pipeline hostage: skip it for this recording
                self.conn.execute("UPDATE chunks SET status='rejected', reject_reason='enhance_skipped', updated_at=? WHERE video_id=? AND status='enhance'", (time.time(), vid))
                stats["enhance"] = {"skipped_backpressure": len(enh_rows), "queue_s": round(queued)}
                enh_rows = []
                cand = [c for c in cand if c["status"] == "candidate"]
                if not cand:
                    self._finish_video(v, [], stats)
                    return
        if enh_rows:
            tar_path = self._write_enhance_payload(wd, v, enh_rows)
            secs = sum(c["dur_ms"] for c in enh_rows) / 1000.0
            D.enqueue_job(self.conn, "enhance", vid, str(tar_path), payload_bytes=tar_path.stat().st_size, audio_seconds=secs, n_items=len(enh_rows))
            D.set_video_status(self.conn, vid, "enhance_queued", stats_json=stats)
            return
        tar_path, n, secs = self._write_asr_payload(wd, x16, sr, cand, None)
        D.enqueue_job(self.conn, "transcribe", vid, str(tar_path), payload_bytes=tar_path.stat().st_size, audio_seconds=secs, n_items=n)
        D.set_video_status(self.conn, vid, "transcribe_queued", stats_json=stats)

    def _write_enhance_payload(self, wd: Path, v, rows: list) -> Path:
        E = self.cfg.quality.enhance
        xm, sr_in = self._load_export_master(wd, v)
        tar_path = wd / "enh_in.tar"
        items = []
        with tarfile.open(tar_path, "w") as tf:
            for c in rows:
                x = self._slice(xm, sr_in, c["start_ms"], c["end_ms"])
                buf = io.BytesIO()
                sf.write(buf, x, sr_in, format="WAV", subtype="PCM_16")
                data = buf.getvalue()
                ti = tarfile.TarInfo(name=f"{c['id']}.wav")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
                items.append({"id": c["id"], "file": f"{c['id']}.wav"})
            man = json.dumps({"items": items, "params": {"mode": E.mode, "nfe": E.nfe, "solver": E.solver, "lambd": E.lambd, "tau": E.tau}}).encode()
            ti = tarfile.TarInfo(name="manifest.json")
            ti.size = len(man)
            tf.addfile(ti, io.BytesIO(man))
        return tar_path

    # ------------------------------------------------------------------ stage 3: rescore enhanced chunks
    def rescore(self, v) -> None:
        wd = self.wd(v)
        vid = v["id"]
        x16, sr = self._load_master(wd)
        vad = self._load_vad(wd, len(x16))
        noise_floor = None
        res = wd / "enhance_result.tar"
        enh_dir = wd / "enh"
        enh_dir.mkdir(exist_ok=True)
        got: dict[str, Path] = {}
        if res.exists():
            with tarfile.open(res, "r:*") as tf:
                for m in tf.getmembers():
                    if m.name.startswith("/") or ".." in m.name:
                        continue
                    tf.extract(m, enh_dir)
            man = enh_dir / "manifest.json"
            if man.exists():
                for it in json.loads(man.read_text()).get("items", []):
                    p = enh_dir / it["file"]
                    if p.exists():
                        got[it["id"]] = p
        rows = self.conn.execute("SELECT * FROM chunks WHERE video_id=? AND status='enhance' ORDER BY idx", (vid,)).fetchall()
        A = self.cfg.quality.accept
        n_ok = n_bad = 0
        for c in rows:
            cid = c["id"]
            m_old = D.uj(c["metrics_json"], {}) or {}
            p = got.get(cid)
            if p is None:
                self.conn.execute("UPDATE chunks SET status='rejected', reject_reason=?, updated_at=? WHERE id=?", ("enhance_failed", time.time(), cid))
                n_bad += 1
                continue
            y, ysr = sf.read(str(p), dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y[:, 0]
            y16 = resample(y, ysr, sr)
            sf.write(str(enh_dir / f"{cid}.16k.wav"), y16, sr, subtype="PCM_16")
            # tag the enhanced chunk itself
            tags = self.tagger.tag_windows(resample(y16, sr, self.tagger.SR), win_s=4.0, hop_s=2.0)
            fa, fb = int(c["start_ms"] / vad.frame_ms), int(c["end_ms"] / vad.frame_ms)
            m = self._score_chunk(y16, sr, vad.probs[fa:fb][: len(y16) // self.cfg.vad.hop], None, tags, 0.0, len(y16) / sr)
            m.update({"vad_speech_ratio": m_old.get("vad_speech_ratio"), "speaker_dominance": m_old.get("speaker_dominance"),
                      "diar_coverage": m_old.get("diar_coverage"), "pre_enhance": {k: m_old.get(k) for k in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "snr_db", "music_prob")}})
            ok = (m["dnsmos_ovrl"] >= A.ovrl_min and m["dnsmos_sig"] >= A.sig_min and m["dnsmos_bak"] >= A.bak_min and m["dnsmos_p808"] >= A.p808_min
                  and m.get("music_prob", 0) <= A.music_prob_max and m["clipping_ratio"] <= A.clipping_max and A.rms_dbfs_min <= m["rms_dbfs"] <= A.rms_dbfs_max)
            if ok:
                self.conn.execute("UPDATE chunks SET status='candidate', enhanced=1, metrics_json=?, updated_at=? WHERE id=?", (D.j(m), time.time(), cid))
                n_ok += 1
            else:
                self.conn.execute("UPDATE chunks SET status='rejected', reject_reason='enhance_insufficient', enhanced=1, metrics_json=?, updated_at=? WHERE id=?",
                                  (D.j(m), time.time(), cid))
                n_bad += 1
        self.stats["rescored"] += 1
        stats = D.uj(v["stats_json"], {}) or {}
        stats["enhance"] = {"ok": n_ok, "rejected": n_bad}
        log.info("rescored %s: %d enhanced accepted, %d rejected", vid, n_ok, n_bad)
        cand = self.conn.execute("SELECT * FROM chunks WHERE video_id=? AND status='candidate' ORDER BY idx", (vid,)).fetchall()
        if not cand:
            self._finish_video(v, [], stats)
            return
        tar_path, n, secs = self._write_asr_payload(wd, x16, sr, cand, enh_dir)
        D.enqueue_job(self.conn, "transcribe", vid, str(tar_path), payload_bytes=tar_path.stat().st_size, audio_seconds=secs, n_items=n)
        D.set_video_status(self.conn, vid, "transcribe_queued", stats_json=stats)

    # ------------------------------------------------------------------ stage 4: finalize
    def finalize(self, v) -> None:
        wd = self.wd(v)
        vid = v["id"]
        res = wd / "transcribe_result.json"
        if not res.exists():
            raise RuntimeError("transcription result missing")
        asr_res = json.loads(res.read_text(encoding="utf-8"))
        texts = asr_res.get("texts", {})
        confs = asr_res.get("confs", {}) or {}
        cand = self.conn.execute("SELECT * FROM chunks WHERE video_id=? AND status='candidate' ORDER BY idx", (vid,)).fetchall()
        T, A = self.cfg.quality.text, self.cfg.quality.accept
        export_sr = self.cfg.audio.export_sr
        fmt = self.cfg.audio.export_format
        q = self.conn.execute("SELECT genre FROM queries WHERE id=?", (v["query_id"],)).fetchone() if v["query_id"] else None
        genre = (q["genre"] if q else "") or ""
        enh_dir = wd / "enh"
        xm, xm_sr = self._load_export_master(wd, v)
        accepted = []
        n_rej = 0
        reasons: dict[str, int] = {}
        # ---- pass 1: text sanity + per-chunk LID
        lids: dict[str, object] = {}
        texts_ok: dict[str, tuple[str, float]] = {}
        for c in cand:
            cid = c["id"]
            text = (texts.get(cid) or "").strip()
            dur_s = c["dur_ms"] / 1000.0
            cps = len(text) / dur_s if dur_s > 0 else 0.0
            reason = None
            aconf = confs.get(cid)
            if len(text) < T.min_chars:
                reason = "asr_empty"
            elif aconf is not None and aconf < T.min_asr_conf:
                reason = "asr_lowconf"
            elif cps < T.min_chars_per_sec:
                reason = "asr_rate_low"
            elif cps > T.max_chars_per_sec:
                reason = "asr_rate_high"
            if reason:
                n_rej += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                m0 = D.uj(c["metrics_json"], {}) or {}
                if aconf is not None:
                    m0["asr_conf"] = round(float(aconf), 4)
                self.conn.execute("UPDATE chunks SET status='rejected', reject_reason=?, text=?, metrics_json=?, updated_at=? WHERE id=?",
                                  (reason, text, D.j(m0), time.time(), cid))
                continue
            texts_ok[cid] = (text, cps, aconf)
            lids[cid] = identify(text, expected=v["lang_hint"], prior_weight=self.cfg.lid.prior_weight, code_mix_threshold=self.cfg.lid.code_mix_threshold)
        # ---- source-level consensus: majority script and language, weighted by duration
        from collections import Counter
        from ..languages import SCRIPT_LANGS
        w_script: Counter = Counter()
        w_lang: Counter = Counter()
        cert_w: dict[str, float] = {}
        cert_n: dict[str, float] = {}
        durs = {c["id"]: c["dur_ms"] / 1000.0 for c in cand}
        for cid, lid in lids.items():
            if lid.lang == "und":
                continue
            w_script[lid.script_key] += durs[cid]
            if lid.confidence >= 0.6:
                w_lang[lid.lang] += durs[cid]
                cert_w[lid.lang] = cert_w.get(lid.lang, 0.0) + lid.confidence * durs[cid]
                cert_n[lid.lang] = cert_n.get(lid.lang, 0.0) + durs[cid]
        star_script = w_script.most_common(1)[0][0] if w_script else None
        siblings = set(SCRIPT_LANGS.get(star_script, ())) if star_script else set()
        cands_lang = {l: w for l, w in w_lang.items() if l in siblings}
        if not cands_lang:
            for cid, lid in lids.items():
                if lid.lang in siblings:
                    cands_lang[lid.lang] = cands_lang.get(lid.lang, 0.0) + durs[cid]
        star_lang = max(cands_lang, key=cands_lang.get) if cands_lang else None
        star_cert = (cert_w[star_lang] / cert_n[star_lang]) if star_lang in cert_n else 0.7
        # A Latin-script majority is only believable when the recording was expected to be English
        # (discovery hint or declared language); otherwise the ASR failed on the whole recording.
        declared = ((v["orig_lang"] or "").split("-")[0].lower()) if "orig_lang" in v.keys() else ""
        if star_script == "latin" and v["lang_hint"] != "en" and declared != "en":
            star_lang = None
            reasons["latin_unexpected"] = reasons.get("latin_unexpected", 0) + len(texts_ok)
            n_rej += len(texts_ok)
            for cid in texts_ok:
                self.conn.execute("UPDATE chunks SET status='rejected', reject_reason='latin_unexpected', text=?, updated_at=? WHERE id=?", (texts_ok[cid][0], time.time(), cid))
            texts_ok = {}
        # English recordings must be INDIAN English: a long recording with no Indian context at all is not what we want
        if star_lang == "en" and texts_ok:
            cues = sum(india_cue_count(t[0]) for t in texts_ok.values())
            total_s = sum(durs[cid] for cid in texts_ok)
            stats_prev = D.uj(v["stats_json"], {}) or {}
            stats_prev["india_cues"] = cues
            self.conn.execute("UPDATE videos SET stats_json=? WHERE id=?", (D.j(stats_prev), v["id"]))
            verdict = self.conn.execute("SELECT india_verdict, india_conf FROM channels WHERE id=?", (v["channel_id"],)).fetchone() if v["channel_id"] else None
            verdict_ok = bool(verdict and verdict["india_verdict"] == "yes" and (verdict["india_conf"] or 0) >= 0.7)
            if not verdict_ok:
                # the creator check never ran (or said no): run it now on title/channel/description + a transcript sample
                from ..llm import LLM, judge_indian_english
                meta = D.uj(v["meta_json"], {}) or {}
                sample = " ".join(t[0] for t in list(texts_ok.values())[:40])[:1500]
                ok, conf, why = judge_indian_english(LLM(self.cfg.llm), v["title"] or "", v["channel"] or "",
                                                     (meta.get("description") or "") + "\n\nTRANSCRIPT SAMPLE: " + sample, meta.get("tags") or [])
                verdict_ok = ok and conf >= 0.7
                stats_prev["india_judge"] = {"ok": ok, "conf": conf, "why": why[:120]}
                self.conn.execute("UPDATE videos SET stats_json=? WHERE id=?", (D.j(stats_prev), v["id"]))
                if v["channel_id"]:
                    self.conn.execute("INSERT INTO channels(id, name, lang_hint, updated_at, india_verdict, india_conf) VALUES (?,?,?,?,?,?) "
                                      "ON CONFLICT(id) DO UPDATE SET india_verdict=excluded.india_verdict, india_conf=excluded.india_conf",
                                      (v["channel_id"], v["channel"], "en", time.time(), "yes" if verdict_ok else "no", conf))
            need_cues = 1 if total_s < 600 else 3
            if not verdict_ok or cues < need_cues:
                for cid in texts_ok:
                    self.conn.execute("UPDATE chunks SET status='rejected', reject_reason='not_indian_english', text=?, updated_at=? WHERE id=?", (texts_ok[cid][0], time.time(), cid))
                reasons["not_indian_english"] = reasons.get("not_indian_english", 0) + len(texts_ok)
                n_rej += len(texts_ok)
                texts_ok = {}
        # ---- pass 2: align every chunk to the consensus, reject script outliers, then export
        for i, c in enumerate(cand):
            cid = c["id"]
            if cid not in texts_ok:
                continue
            text, cps, aconf = texts_ok[cid]
            lid = lids[cid]
            dur_s = c["dur_ms"] / 1000.0
            reason = None
            other_scripts = sum(sh for sk, sh in lid.scripts.items() if sk not in ("latin", star_script))
            if lid.lang == "und" or lid.lang not in LANGUAGES or star_lang is None:
                reason = "lid_unknown"
            elif lid.lang not in self._enabled_codes:
                reason = "lang_not_collected"   # e.g. a script the recogniser is not trained for
            elif lid.script_key != star_script:
                reason = "script_outlier"          # transcript is in a different script than the recording: ASR unreliable here
            elif other_scripts >= 0.1 or (other_scripts > 0 and lid.n_tokens <= 6):
                reason = "script_mix"              # stray letters from other scripts: ASR unreliable here
            elif lid.mixed_tokens >= 2 or lid.mixed_tokens / max(1, lid.n_tokens) > 0.08:
                reason = "script_mix"              # letters of two scripts inside single words: garbled ASR output
            else:
                if lid.lang != star_lang and lid.lang in siblings:
                    # same-script sibling (bn/as, hi/mr, ...): a recording is monolingual, follow the majority
                    share = lid.composition.pop(lid.lang, 0.0)
                    lid.composition[star_lang] = lid.composition.get(star_lang, 0.0) + share
                    lid.composition = dict(sorted(lid.composition.items(), key=lambda kv: -kv[1]))
                    lid.lang = star_lang
                    lid.confidence = max(lid.confidence, star_cert)
                    lid.consensus = True
                elif lid.lang == star_lang and lid.confidence < star_cert:
                    lid.confidence = max(lid.confidence, min(star_cert, lid.confidence + 0.3))
                    lid.consensus = True
                lid.dominance = float(lid.composition.get(lid.lang, 0.0))
                if lid.confidence < T.min_lang_conf:
                    reason = "lid_lowconf"
                elif lid.dominance < 0.4:
                    reason = "lid_too_mixed"
            if reason:
                n_rej += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                self.conn.execute("UPDATE chunks SET status='rejected', reject_reason=?, text=?, lang=?, lang_json=?, updated_at=? WHERE id=?",
                                  (reason, text, lid.lang if lid.lang != "und" else None, D.j(lid.as_dict()), time.time(), cid))
                continue
            # render audio at the export rate
            if c["enhanced"] and (enh_dir / f"{cid}.wav").exists():
                y, ysr = sf.read(str(enh_dir / f"{cid}.wav"), dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = y[:, 0]
                audio = resample(y, ysr, export_sr)
            else:
                audio = self._slice(xm, xm_sr, c["start_ms"], c["end_ms"])
                if xm_sr != export_sr:
                    audio = resample(audio, xm_sr, export_sr)
            if audio.size < int(0.4 * export_sr):
                n_rej += 1
                reasons["export_empty"] = reasons.get("export_empty", 0) + 1
                self.conn.execute("UPDATE chunks SET status='rejected', reject_reason='export_empty', text=?, updated_at=? WHERE id=?", (text, time.time(), cid))
                continue
            m = D.uj(c["metrics_json"], {}) or {}
            if aconf is not None:
                m["asr_conf"] = round(float(aconf), 4)
            bw = bandwidth_hz(audio, export_sr)
            m["bandwidth_hz"] = round(bw, 0)
            if bw < A.bandwidth_hz_min:
                n_rej += 1
                reasons["bandwidth"] = reasons.get("bandwidth", 0) + 1
                self.conn.execute("UPDATE chunks SET status='rejected', reject_reason='bandwidth', text=?, metrics_json=?, updated_at=? WHERE id=?",
                                  (text, D.j(m), time.time(), cid))
                continue
            audio = peak_normalize(audio, -0.5)
            cfp = clip_fingerprint(audio, export_sr)
            dup = duplicate_clip(self.conn, cfp, int(len(audio) * 1000 / export_sr))
            if dup:
                n_rej += 1
                reasons["duplicate_clip"] = reasons.get("duplicate_clip", 0) + 1
                self.conn.execute("UPDATE chunks SET status='rejected', reject_reason='duplicate_clip', text=?, updated_at=? WHERE id=?", (text, time.time(), cid))
                continue
            remember_clip(self.conn, cid, cfp, int(len(audio) * 1000 / export_sr))
            meta = {
                "id": cid, "source_id": v["source_hash"], "segment_index": c["idx"], "duration_s": round(len(audio) / export_sr, 3),
                "sample_rate": export_sr, "format": fmt, "text": text, "lang": lid.lang, "lang_name": LANGUAGES[lid.lang].name,
                "lid": lid.as_dict(), "speaker_id": f"{v['source_hash']}_s{c['speaker'] if c['speaker'] is not None and c['speaker'] >= 0 else 0}",
                "enhanced": bool(c["enhanced"]), "quality": m, "chars_per_sec": round(cps, 3), "genre": genre, "created_at": _now_iso(),
            }
            path, nbytes = stage_chunk(self.cfg.paths.staging_dir, lid.lang, cid, audio, export_sr, meta, fmt)
            self.conn.execute("UPDATE chunks SET status='accepted', text=?, lang=?, lang_conf=?, lang_json=?, metrics_json=?, staged_path=?, updated_at=? WHERE id=?",
                              (text, lid.lang, lid.confidence, D.j(lid.as_dict()), D.j(m), path, time.time(), cid))
            accepted.append((cid, path, meta["duration_s"]))
            if i % 40 == 0:
                D.touch_lease(self.conn, vid, self.cfg.workers.lease_s)
                self.heartbeat("finalizing", f"{vid} {i}/{len(cand)}")
        self._keep_samples(accepted)
        stats = D.uj(v["stats_json"], {}) or {}
        if star_lang:
            stats["lang_detected"] = star_lang
            stats["lang_certainty"] = round(star_cert, 3)
            if star_lang != v["lang_hint"]:
                self.conn.execute("UPDATE videos SET lang_hint=? WHERE id=?", (star_lang, v["id"]))
                v = dict(v)
                v["lang_hint"] = star_lang
        acc_sec = sum(a[2] for a in accepted)
        stats.update({"finalize": {"accepted": len(accepted), "rejected": n_rej, "reasons": reasons}, "accepted_sec": round(acc_sec, 1),
                      "accepted_chunks": len(accepted), "yield": round(acc_sec / (v["duration_s"] or 1), 4)})
        self.stats["chunks_accepted"] += len(accepted)
        self.stats["chunks_rejected"] += n_rej
        self.stats["accepted_sec"] = round(self.stats["accepted_sec"] + acc_sec, 1)
        self._finish_video(v, accepted, stats)

    def _keep_samples(self, accepted: list) -> None:
        sd = self.cfg.paths.samples_dir
        sd.mkdir(parents=True, exist_ok=True)
        keep = self.cfg.storage.keep_accepted_samples
        for cid, path, _ in accepted:          # every fresh clip is playable on the dashboard; rotation keeps the dir bounded
            try:
                shutil.copyfile(path, sd / f"{cid}.flac")
            except OSError:
                pass
        try:
            files = sorted(sd.glob("*.flac"), key=lambda p: p.stat().st_mtime)
            for p in files[: max(0, len(files) - keep)]:
                p.unlink(missing_ok=True)
        except OSError:
            pass

    def _finish_video(self, v, accepted: list, stats: dict) -> None:
        vid = v["id"]
        acc_sec = sum(a[2] for a in accepted) if accepted else 0.0
        self._touch_channel(v, acc_sec, v["duration_s"] or 0.0)
        D.set_video_status(self.conn, vid, "done", stats_json=stats)
        self.stats["finalized"] += 1
        self._cleanup(v)
        self.event("video", f"done {vid} [{v['lang_hint']}]: {len(accepted)} chunks, {acc_sec / 60:.1f} min accepted of {(v['duration_s'] or 0) / 60:.0f} min",
                   data={"title": v["title"], "accepted_sec": acc_sec})
        log.info("finished %s: %d accepted (%.1f min) of %.0f min source", vid, len(accepted), acc_sec / 60, (v["duration_s"] or 0) / 60)
