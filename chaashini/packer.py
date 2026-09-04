"""Build Hugging Face `datasets`-compatible parquet shards from the staging directory.

Each shard: data/<lang>/<name>-<lang>-<NNNNN>.parquet with an `audio` column (bytes+path)
declared as an Audio feature, so `load_dataset(repo, lang)` decodes it directly.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("chaashini.packer")

COLUMNS = [
    "id", "audio", "text", "language", "language_name", "language_confidence", "language_mix", "script", "code_mixed",
    "duration_s", "sample_rate", "speaker_id", "source_id", "segment_index", "enhanced",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808", "music_prob", "speech_prob", "noise_prob",
    "snr_db", "rms_dbfs", "peak_dbfs", "clipping_ratio", "bandwidth_hz", "vad_speech_ratio", "speaker_dominance",
    "chars_per_sec", "asr_confidence", "genre", "created_at",
]


def features():
    from datasets import Audio, Features, Value
    return Features({
        "id": Value("string"), "audio": Audio(), "text": Value("string"), "language": Value("string"),
        "language_name": Value("string"), "language_confidence": Value("float32"), "language_mix": Value("string"),
        "script": Value("string"), "code_mixed": Value("bool"), "duration_s": Value("float32"), "sample_rate": Value("int32"),
        "speaker_id": Value("string"), "source_id": Value("string"), "segment_index": Value("int32"), "enhanced": Value("bool"),
        "dnsmos_sig": Value("float32"), "dnsmos_bak": Value("float32"), "dnsmos_ovrl": Value("float32"), "dnsmos_p808": Value("float32"),
        "music_prob": Value("float32"), "speech_prob": Value("float32"), "noise_prob": Value("float32"),
        "snr_db": Value("float32"), "rms_dbfs": Value("float32"), "peak_dbfs": Value("float32"), "clipping_ratio": Value("float32"),
        "bandwidth_hz": Value("float32"), "vad_speech_ratio": Value("float32"), "speaker_dominance": Value("float32"),
        "chars_per_sec": Value("float32"), "asr_confidence": Value("float32"), "genre": Value("string"), "created_at": Value("string"),
    })


def arrow_schema():
    import pyarrow as pa
    f32, i32, s, b = pa.float32(), pa.int32(), pa.string(), pa.bool_()
    return pa.schema([
        ("id", s), ("audio", pa.struct([("bytes", pa.binary()), ("path", s)])), ("text", s), ("language", s), ("language_name", s),
        ("language_confidence", f32), ("language_mix", s), ("script", s), ("code_mixed", b), ("duration_s", f32), ("sample_rate", i32),
        ("speaker_id", s), ("source_id", s), ("segment_index", i32), ("enhanced", b),
        ("dnsmos_sig", f32), ("dnsmos_bak", f32), ("dnsmos_ovrl", f32), ("dnsmos_p808", f32), ("music_prob", f32), ("speech_prob", f32),
        ("noise_prob", f32), ("snr_db", f32), ("rms_dbfs", f32), ("peak_dbfs", f32), ("clipping_ratio", f32), ("bandwidth_hz", f32),
        ("vad_speech_ratio", f32), ("speaker_dominance", f32), ("chars_per_sec", f32), ("asr_confidence", f32), ("genre", s), ("created_at", s),
    ])


def write_parquet(rows: list[dict], path: Path) -> None:
    """Write rows with pyarrow and embed the Hugging Face `features` metadata (Audio feature for `audio`)
    exactly as `datasets` does, without running any audio encoder."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = arrow_schema()
    cols = {name: [r[name] for r in rows] for name in schema.names}
    table = pa.Table.from_pydict(cols, schema=schema)
    meta = {"info": {"features": features().to_dict()}}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"huggingface": json.dumps(meta).encode("utf-8")})
    pq.write_table(table, str(path), compression="zstd", row_group_size=256)


def _row(meta: dict, audio_bytes: bytes, fname: str) -> dict:
    q = meta.get("quality", {})
    lid = meta.get("lid", {})
    return {
        "id": meta["id"], "audio": {"bytes": audio_bytes, "path": fname}, "text": meta.get("text", ""),
        "language": meta.get("lang"), "language_name": meta.get("lang_name"), "language_confidence": float(lid.get("confidence", 0.0)),
        "language_mix": json.dumps(lid.get("composition", {}), ensure_ascii=False), "script": lid.get("script", ""),
        "code_mixed": bool(lid.get("code_mixed", False)), "duration_s": float(meta["duration_s"]), "sample_rate": int(meta["sample_rate"]),
        "speaker_id": meta.get("speaker_id", ""), "source_id": meta.get("source_id", ""), "segment_index": int(meta.get("segment_index", 0)),
        "enhanced": bool(meta.get("enhanced", False)),
        "dnsmos_sig": float(q.get("dnsmos_sig", 0)), "dnsmos_bak": float(q.get("dnsmos_bak", 0)), "dnsmos_ovrl": float(q.get("dnsmos_ovrl", 0)),
        "dnsmos_p808": float(q.get("dnsmos_p808", 0)), "music_prob": float(q.get("music_prob", 0)), "speech_prob": float(q.get("speech_prob", 0)),
        "noise_prob": float(q.get("noise_prob", 0)), "snr_db": float(q.get("snr_db", 0)), "rms_dbfs": float(q.get("rms_dbfs", 0)),
        "peak_dbfs": float(q.get("peak_dbfs", 0)), "clipping_ratio": float(q.get("clipping_ratio", 0)), "bandwidth_hz": float(q.get("bandwidth_hz", 0)),
        "vad_speech_ratio": float(q.get("vad_speech_ratio", 0)), "speaker_dominance": float(q.get("speaker_dominance", 0)),
        "chars_per_sec": float(meta.get("chars_per_sec", 0)), "asr_confidence": float(q.get("asr_conf", 0) or 0), "genre": meta.get("genre", "") or "", "created_at": meta.get("created_at", ""),
    }


def build_shards(staging_dir: Path, shards_dir: Path, lang: str, next_index: int, target_mb: int, name: str = "chaashini",
                 min_seconds: float = 60.0) -> list[dict]:
    """Pack all staged files of `lang` into one or more parquet shards. Returns shard descriptors.
    Staged files are deleted only after the parquet is fully written and fsynced."""
    src = staging_dir / lang
    if not src.exists():
        return []
    files = sorted(src.glob("*.json"))
    if not files:
        return []
    shards: list[dict] = []
    rows: list[dict] = []
    consumed: list[Path] = []
    size = 0
    dur = 0.0
    target = target_mb * 1024 * 1024
    out_dir = shards_dir / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    def flush():
        nonlocal rows, consumed, size, dur, next_index
        if not rows or dur < min_seconds:
            return
        fname = f"{name}-{lang}-{next_index:05d}.parquet"
        path = out_dir / fname
        tmp = out_dir / (fname + ".tmp")
        write_parquet(rows, tmp)
        os.replace(tmp, path)
        with open(path, "rb") as f:
            os.fsync(f.fileno())
        shards.append({"lang": lang, "path": str(path), "hf_path": f"data/{lang}/{fname}", "n_chunks": len(rows),
                       "duration_s": dur, "size_bytes": path.stat().st_size, "index": next_index})
        for p in consumed:
            for q in (p, p.with_suffix(".json")):
                try:
                    q.unlink()
                except OSError:
                    pass
        log.info("built shard %s (%d chunks, %.1f min, %.1f MB)", fname, len(rows), dur / 60, path.stat().st_size / 1e6)
        rows, consumed, size, dur = [], [], 0, 0.0
        next_index += 1

    for js in files:
        try:
            with open(js, encoding="utf-8") as f:
                meta = json.load(f)
            audio_path = js.with_suffix("." + meta.get("format", "flac"))
            if not audio_path.exists():
                js.unlink(missing_ok=True)
                continue
            with open(audio_path, "rb") as f:
                b = f.read()
        except Exception as e:  # noqa: BLE001
            log.warning("skip staged %s: %s", js, e)
            continue
        rows.append(_row(meta, b, audio_path.name))
        consumed.append(audio_path)
        size += len(b)
        dur += float(meta.get("duration_s", 0.0))
        if size >= target:
            flush()
    # final partial shard is kept for the next round unless it is the only material and >= min_seconds
    flush()
    return shards
