"""Configuration: a YAML file layered over sane defaults, with a few env overrides.

Load order: defaults (below) <- configs/chaashini.yaml <- configs/local.yaml <- environment.
Secrets (HF token, internal API token) come from the environment or local.yaml, never from
the committed YAML.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PathsCfg(BaseModel):
    root: Path = Path("/opt/chaashini")
    data_dir: Path | None = None       # <root>/data
    work_dir: Path | None = None       # <data>/work      per-video scratch
    staging_dir: Path | None = None    # <data>/staging   accepted chunks waiting for a shard
    shards_dir: Path | None = None     # <data>/shards    parquet shards (built / pushed)
    samples_dir: Path | None = None    # <data>/samples   small audio previews for the dashboard
    models_dir: Path | None = None     # <root>/models
    logs_dir: Path | None = None       # <root>/logs
    db_path: Path | None = None        # <data>/chaashini.db

    def resolve(self) -> "PathsCfg":
        d = self.data_dir or self.root / "data"
        return PathsCfg(
            root=self.root,
            data_dir=d,
            work_dir=self.work_dir or d / "work",
            staging_dir=self.staging_dir or d / "staging",
            shards_dir=self.shards_dir or d / "shards",
            samples_dir=self.samples_dir or d / "samples",
            models_dir=self.models_dir or self.root / "models",
            logs_dir=self.logs_dir or self.root / "logs",
            db_path=self.db_path or d / "chaashini.db",
        )


class LLMCfg(BaseModel):
    base_url: str = "https://models.kapturecrm.com/llm/bolt-surge/v1"
    model: str = "bolt-surge"
    api_key: str = "none"
    timeout_s: float = 120.0
    temperature: float = 0.9
    max_tokens: int = 2048


class HFCfg(BaseModel):
    repo_id: str = "kapturecx/Chaashini"
    token: str = ""                      # env HF_TOKEN
    push_every_hours: float = 10.0       # push once this many hours of NEW accepted audio are staged
    push_check_interval_s: int = 300
    shard_target_mb: int = 400
    max_retries: int = 8
    dataset_name: str = "Chaashini"


class SourceCfg(BaseModel):
    """Public long-form audio discovery/download settings (yt-dlp)."""
    search_results_per_query: int = 40
    min_duration_s: int = 240
    max_duration_s: int = 4 * 3600
    max_videos_per_channel: int = 80
    skip_categories: list[str] = ["Music", "Gaming", "Film & Animation", "Trailers"]
    download_concurrency: int = 4
    cookies_file: str = ""               # Netscape cookies.txt exported from a logged-in browser
    proxy: str = ""                      # e.g. http://user:pass@host:port  (rotating residential recommended at scale)
    rate_limit: str = ""                 # e.g. "8M"
    sleep_requests: float = 0.75         # seconds between HTTP requests to the source
    sleep_interval: float = 2.0          # seconds between downloads
    js_runtime: str = "deno"
    format: str = "bestaudio[language_preference>=?0][acodec=opus]/bestaudio[language_preference>=?0]/bestaudio/best"
    cooldown_base_s: int = 300           # global back-off when the source rate-limits / bot-checks us
    cooldown_max_s: int = 3600
    max_attempts: int = 3
    negative_terms: list[str] = ["-song", "-songs", "-music", "-dj", "-remix", "-lyrics", "-karaoke", "-trailer", "-bhajan", "-jukebox"]


class LanguageCfg(BaseModel):
    code: str
    weight: float = 1.0
    enabled: bool = True


class DiscoveryCfg(BaseModel):
    queries_per_language_per_round: int = 12
    target_backlog_videos: int = 300     # keep at least this many undownloaded videos queued
    round_sleep_s: int = 120
    channel_expand_min_accept_ratio: float = 0.35
    channel_expand_min_videos: int = 2
    channel_expand_max_items: int = 60
    seed_channel_playlists: list[str] = []


class AudioCfg(BaseModel):
    analysis_sr: int = 16000
    export_sr: int = 24000
    export_format: str = "flac"
    enhance_input_sr: int = 44100


class VADCfg(BaseModel):
    threshold: float = 0.5
    hop: int = 256                       # 16 ms @ 16 kHz
    min_speech_ms: int = 500
    max_chunk_ms: int = 30000
    target_chunk_ms: int = 15000         # prefer to split long regions near this length
    min_silence_split_ms: int = 250
    pad_ms: int = 120
    merge_gap_ms: int = 200


class DiarCfg(BaseModel):
    frame_ms: int = 80
    active_prob: float = 0.5
    overlap_margin_ms: int = 250
    min_dominance: float = 0.9           # fraction of chunk frames that must belong to the top speaker
    max_speakers: int = 4


class AcceptCfg(BaseModel):
    ovrl_min: float = 3.0
    sig_min: float = 3.2
    bak_min: float = 3.8
    p808_min: float = 3.0
    music_prob_max: float = 0.15
    speech_prob_min: float = 0.5
    snr_db_min: float = 20.0
    rms_dbfs_min: float = -34.0
    rms_dbfs_max: float = -6.0
    clipping_max: float = 0.001
    bandwidth_hz_min: float = 6000.0
    vad_ratio_min: float = 0.6


class EnhanceGateCfg(BaseModel):
    """Chunks failing acceptance but inside these bounds are sent for enhancement, then re-scored."""
    enabled: bool = True
    ovrl_min: float = 2.3
    sig_min: float = 2.8
    music_prob_max: float = 0.3
    snr_db_min: float = 8.0
    max_fraction_per_video: float = 0.4
    max_chunks_per_video: int = 400
    mode: str = "enhance"                # enhance | denoise
    nfe: int = 32
    solver: str = "midpoint"
    lambd: float = 0.5
    tau: float = 0.5


class TextCfg(BaseModel):
    min_chars: int = 2
    min_chars_per_sec: float = 1.2
    max_chars_per_sec: float = 32.0
    min_lang_conf: float = 0.55


class QualityCfg(BaseModel):
    accept: AcceptCfg = AcceptCfg()
    enhance: EnhanceGateCfg = EnhanceGateCfg()
    text: TextCfg = TextCfg()
    video_music_gate: float = 0.5        # reject a source outright if this fraction of windows is music
    video_gate_windows: int = 24
    min_accept_seconds_per_video: float = 0.0


class WorkersCfg(BaseModel):
    discover: int = 1
    download: int = 4
    process: int = 6
    publish: int = 1
    torch_threads: int = 8
    heartbeat_s: int = 10
    lease_s: int = 3600
    max_videos_in_flight: int = 16


class APICfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8978
    internal_token: str = ""             # env CHAASHINI_INTERNAL_TOKEN; GPU workers must present it
    public_base_url: str = ""            # what GPU workers dial, e.g. http://127.0.0.1:8978 through a tunnel
    status_refresh_s: int = 5


class GPUCfg(BaseModel):
    job_lease_s: int = 2400
    max_attempts: int = 2
    diarize_chunk_len: int = 340         # sortformer "very high latency" (offline quality) preset
    diarize_right_context: int = 40
    diarize_fifo_len: int = 40
    diarize_update_period: int = 300
    diarize_spkcache_len: int = 188
    asr_lookahead_frames: int = 13       # SraVaani [70,13] = highest accuracy
    idle_poll_s: float = 3.0


class StorageCfg(BaseModel):
    min_free_gb: float = 40.0
    keep_rejected_samples: int = 200     # keep a few rejected previews for QA
    keep_accepted_samples: int = 400
    work_ttl_hours: int = 48


class LIDCfg(BaseModel):
    prior_weight: float = 0.15
    code_mix_threshold: float = 0.15


class Config(BaseModel):
    paths: PathsCfg = PathsCfg()
    llm: LLMCfg = LLMCfg()
    hf: HFCfg = HFCfg()
    source: SourceCfg = SourceCfg()
    languages: list[LanguageCfg] = Field(default_factory=list)
    discovery: DiscoveryCfg = DiscoveryCfg()
    audio: AudioCfg = AudioCfg()
    vad: VADCfg = VADCfg()
    diar: DiarCfg = DiarCfg()
    quality: QualityCfg = QualityCfg()
    workers: WorkersCfg = WorkersCfg()
    api: APICfg = APICfg()
    gpu: GPUCfg = GPUCfg()
    storage: StorageCfg = StorageCfg()
    lid: LIDCfg = LIDCfg()

    def enabled_languages(self) -> list[LanguageCfg]:
        return [l for l in self.languages if l.enabled]


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _env_overrides(d: dict) -> dict:
    env = os.environ
    d = dict(d)
    d.setdefault("hf", {})
    d.setdefault("api", {})
    d.setdefault("paths", {})
    d.setdefault("llm", {})
    if env.get("HF_TOKEN"):
        d["hf"]["token"] = env["HF_TOKEN"]
    if env.get("CHAASHINI_INTERNAL_TOKEN"):
        d["api"]["internal_token"] = env["CHAASHINI_INTERNAL_TOKEN"]
    if env.get("CHAASHINI_ROOT"):
        d["paths"]["root"] = env["CHAASHINI_ROOT"]
    if env.get("LLM_API_KEY"):
        d["llm"]["api_key"] = env["LLM_API_KEY"]
    return d


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load configs/chaashini.yaml (+ configs/local.yaml if present) relative to the repo or CHAASHINI_CONFIG."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    elif os.environ.get("CHAASHINI_CONFIG"):
        candidates.append(Path(os.environ["CHAASHINI_CONFIG"]))
    else:
        here = Path(__file__).resolve().parent.parent
        candidates.append(here / "configs" / "chaashini.yaml")
    merged: dict[str, Any] = {}
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                merged = _deep_merge(merged, yaml.safe_load(f) or {})
            local = c.parent / "local.yaml"
            if local.exists():
                with open(local, "r", encoding="utf-8") as f:
                    merged = _deep_merge(merged, yaml.safe_load(f) or {})
    merged = _env_overrides(merged)
    cfg = Config(**merged)
    cfg.paths = cfg.paths.resolve()
    return cfg


_CFG: Config | None = None


def get_config() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG
