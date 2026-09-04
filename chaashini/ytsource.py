"""Discovery and download of publicly available long-form audio via yt-dlp.

Design notes
* Search is done "flat" (cheap): id/title/duration/channel only.  Full metadata is fetched at
  download time where yt-dlp needs it anyway, and `match_filter` rejects unwanted items
  (music category, live, too short/long, wrong original language) BEFORE bytes are pulled.
* Only the ORIGINAL audio track is accepted; auto-dubbed tracks carry language_preference=-1.
* Errors are classified: permanent (private/removed/geo) vs rate-limit/bot-check (global
  cool-down) vs transient (retry).
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import SourceCfg

log = logging.getLogger("chaashini.source")


class SkipVideo(Exception):
    """Permanent, not our fault: unavailable / filtered."""


class RateLimited(Exception):
    """Source is throttling or bot-checking us: back off globally."""


class Transient(Exception):
    """Try again later."""


_PERMANENT = re.compile(
    r"private video|video unavailable|has been removed|not available in your country|members-only|"
    r"join this channel|age.?restricted|sign in to confirm your age|requested format is not available|"
    r"this live event|premieres in|is not available|copyright|account associated|terminated|"
    r"unsupported url|no video formats|skipping|video is unavailable|does not exist", re.I)
_RATELIMIT = re.compile(r"429|too many requests|sign in to confirm you.?re not a bot|bot|rate.?limit|blocked|captcha|403", re.I)


def classify_error(msg: str) -> type[Exception]:
    m = msg or ""
    if _RATELIMIT.search(m) and not _PERMANENT.search(m):
        return RateLimited
    if _PERMANENT.search(m):
        return SkipVideo
    return Transient


def source_hash(video_id: str, salt: str) -> str:
    return hashlib.sha256((salt + ":" + video_id).encode()).hexdigest()[:16]


@dataclass
class Found:
    id: str
    title: str
    duration: float | None
    channel_id: str | None
    channel: str | None
    view_count: int | None
    url: str


def cookie_files(cfg: SourceCfg) -> list[str]:
    """All usable cookie files: the single `cookies_file` plus every *.txt in `cookies_dir`."""
    out: list[str] = []
    if cfg.cookies_file and os.path.exists(cfg.cookies_file):
        out.append(cfg.cookies_file)
    if cfg.cookies_dir and os.path.isdir(cfg.cookies_dir):
        for p in sorted(Path(cfg.cookies_dir).glob("*.txt")):
            if str(p) not in out and p.stat().st_size > 0:
                out.append(str(p))
    return out


def identity(cfg: SourceCfg, index: int) -> tuple[str | None, str | None, str]:
    """(cookie_file, proxy, label) for slot `index`, pairing cookie i with proxy i when both lists exist."""
    files = cookie_files(cfg)
    proxies = [p for p in cfg.proxies if p] or ([cfg.proxy] if cfg.proxy else [])
    n = max(len(files), len(proxies), 1)
    i = index % n
    ck = files[i % len(files)] if files else None
    px = proxies[i % len(proxies)] if proxies else None
    label = (Path(ck).stem if ck else "nocookie") + (f"@{i}" if proxies else "")
    return ck, px, label


def _ydl_base(cfg: SourceCfg, quiet: bool = True, cookies_file: str | None = None, proxy: str | None = None) -> dict:
    opts: dict = {
        "quiet": quiet, "no_warnings": True, "noprogress": True, "ignoreerrors": False,
        "socket_timeout": 30, "retries": 3, "fragment_retries": 5, "extractor_retries": 2,
        "sleep_interval_requests": cfg.sleep_requests,
        "noplaylist": True, "geo_bypass": True, "cachedir": str(Path.home() / ".cache" / "yt-dlp"),
    }
    ck = cookies_file if cookies_file is not None else (cfg.cookies_file or None)
    if ck and os.path.exists(ck):
        opts["cookiefile"] = ck
    px = proxy if proxy is not None else (cfg.proxy or None)
    if px:
        opts["proxy"] = px
    return opts


def _entries(info: dict | None) -> list[dict]:
    if not info:
        return []
    return [e for e in (info.get("entries") or []) if e]


def search(cfg: SourceCfg, query: str, n: int, cookies_file: str | None = None, proxy: str | None = None) -> list[Found]:
    import yt_dlp
    q = " ".join([query] + cfg.negative_terms)
    opts = _ydl_base(cfg, cookies_file=cookies_file, proxy=proxy)
    opts.update({"extract_flat": "in_playlist", "skip_download": True, "playlistend": n})
    out: list[Found] = []
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{n}:{q}", download=False)
    except Exception as e:  # noqa: BLE001
        raise classify_error(str(e))(str(e)) from e
    for e in _entries(info):
        if e.get("ie_key") not in (None, "Youtube") or not e.get("id"):
            continue
        out.append(Found(e["id"], e.get("title") or "", e.get("duration"), e.get("channel_id"), e.get("channel") or e.get("uploader"),
                         e.get("view_count"), e.get("url") or f"https://www.youtube.com/watch?v={e['id']}"))
    return out


def channel_videos(cfg: SourceCfg, channel_id: str, n: int, cookies_file: str | None = None, proxy: str | None = None) -> list[Found]:
    import yt_dlp
    opts = _ydl_base(cfg, cookies_file=cookies_file, proxy=proxy)
    opts.update({"extract_flat": "in_playlist", "skip_download": True, "playlistend": n})
    url = f"https://www.youtube.com/channel/{channel_id}/videos"
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:  # noqa: BLE001
        raise classify_error(str(e))(str(e)) from e
    out: list[Found] = []
    for e in _entries(info):
        if not e.get("id"):
            continue
        out.append(Found(e["id"], e.get("title") or "", e.get("duration"), channel_id, info.get("channel") or info.get("uploader"),
                         e.get("view_count"), f"https://www.youtube.com/watch?v={e['id']}"))
    return out


def playlist_videos(cfg: SourceCfg, url: str, n: int, cookies_file: str | None = None, proxy: str | None = None) -> list[Found]:
    import yt_dlp
    opts = _ydl_base(cfg, cookies_file=cookies_file, proxy=proxy)
    opts.update({"extract_flat": "in_playlist", "skip_download": True, "playlistend": n})
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:  # noqa: BLE001
        raise classify_error(str(e))(str(e)) from e
    out: list[Found] = []
    for e in _entries(info):
        if not e.get("id"):
            continue
        out.append(Found(e["id"], e.get("title") or "", e.get("duration"), e.get("channel_id") or info.get("channel_id"),
                         e.get("channel") or info.get("channel"), e.get("view_count"), f"https://www.youtube.com/watch?v={e['id']}"))
    return out


def select_audio_format(ctx: dict):
    """yt-dlp format selector (callable form).

    Preference order: audio-only DASH (https) over HLS (m3u8 fragments cannot be seeked reliably),
    non-DRC over DRC (loudness-processed) variants, ORIGINAL track when the video carries dubbed
    tracks (original = language_preference >= 10, dubbed = -1; single-track videos also report -1,
    so the rule is only applied when an original exists), then opus over AAC, then bitrate.
    """
    fmts = [f for f in (ctx.get("formats") or []) if f.get("vcodec") in (None, "none") and f.get("acodec") not in (None, "none")]
    if not fmts:
        fmts = [f for f in (ctx.get("formats") or []) if f.get("acodec") not in (None, "none")]
    if not fmts:
        return
    pool = [f for f in fmts if "m3u8" not in (f.get("protocol") or "")] or fmts
    pool = [f for f in pool if "drc" not in (f.get("format_id") or "").lower() and "drc" not in (f.get("format_note") or "").lower()] or pool
    if any((f.get("language_preference") or 0) >= 10 for f in pool):
        pool = [f for f in pool if (f.get("language_preference") or 0) >= 10]
    pool.sort(key=lambda f: ((f.get("acodec") or "").startswith("opus"), f.get("abr") or f.get("tbr") or 0), reverse=True)
    yield pool[0]


@dataclass
class Downloaded:
    path: str
    info: dict
    audio_track_lang: str | None
    orig_lang: str | None
    categories: list[str]
    duration: float
    ext: str
    abr: float | None
    asr: int | None


def _match_filter_factory(cfg: SourceCfg, allowed_langs: set[str] | None, extra_check=None):
    def f(info: dict, *, incomplete: bool = False):
        if extra_check is not None and not incomplete:
            why = extra_check(info)
            if why:
                return why
        d = info.get("duration")
        if d is not None:
            if d < cfg.min_duration_s:
                return f"too short ({d:.0f}s)"
            if d > cfg.max_duration_s:
                return f"too long ({d:.0f}s)"
        if info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
            return "live"
        cats = [c for c in (info.get("categories") or [])]
        for c in cats:
            if c in cfg.skip_categories:
                return f"category:{c}"
        if info.get("age_limit", 0) and info["age_limit"] >= 18:
            return "age-restricted"
        lang = (info.get("language") or "").split("-")[0].lower()
        if allowed_langs and lang and lang not in allowed_langs:
            return f"declared language {lang}"
        return None
    return f


def download(cfg: SourceCfg, video_id: str, out_dir: str | Path, allowed_langs: set[str] | None = None,
             cookies_file: str | None = None, proxy: str | None = None, extra_check=None) -> Downloaded:
    """Download the original-language audio track for `video_id` into out_dir/<id>.<ext>."""
    import yt_dlp
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = _ydl_base(cfg, cookies_file=cookies_file, proxy=proxy)
    opts.update({
        "format": select_audio_format,
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "match_filter": _match_filter_factory(cfg, allowed_langs, extra_check),
        "sleep_interval": cfg.sleep_interval, "max_sleep_interval": cfg.sleep_interval * 2,
        "concurrent_fragment_downloads": 2,
        "overwrites": True, "continuedl": True,
    })
    if cfg.rate_limit:
        opts["ratelimit"] = _parse_rate(cfg.rate_limit)
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise classify_error(str(e))(str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise classify_error(str(e))(str(e)) from e
    if info is None:
        raise SkipVideo("filtered by match_filter")
    reqs = info.get("requested_downloads") or []
    path = None
    for r in reqs:
        p = r.get("filepath") or r.get("_filename")
        if p and os.path.exists(p):
            path = p
            break
    if not path:
        # yt-dlp can skip silently when match_filter rejected; look on disk as a last resort
        cands = sorted(out_dir.glob(f"{video_id}.*"))
        if not cands:
            raise SkipVideo("filtered or no file produced")
        path = str(cands[0])
    fmt = reqs[0] if reqs else info
    track_lang = fmt.get("language") or info.get("language")
    lp = fmt.get("language_preference")
    has_original = any((f.get("language_preference") or 0) >= 10 for f in (info.get("formats") or []))
    if has_original and lp is not None and lp < 0:
        try:
            os.remove(path)
        except OSError:
            pass
        raise SkipVideo("only dubbed audio track available")
    return Downloaded(path=path, info=info, audio_track_lang=track_lang, orig_lang=info.get("language"),
                      categories=list(info.get("categories") or []), duration=float(info.get("duration") or 0.0),
                      ext=fmt.get("ext") or Path(path).suffix.lstrip("."), abr=fmt.get("abr"), asr=fmt.get("asr"))


def _parse_rate(s: str) -> int:
    m = re.match(r"^\s*([\d.]+)\s*([kKmM]?)", s)
    if not m:
        return 0
    v = float(m.group(1))
    return int(v * {"": 1, "k": 1024, "K": 1024, "m": 1024 ** 2, "M": 1024 ** 2}[m.group(2)])


def slim_info(info: dict) -> dict:
    keys = ("id", "title", "channel", "channel_id", "duration", "view_count", "like_count", "upload_date", "categories",
            "tags", "language", "live_status", "age_limit", "availability", "uploader_id", "channel_follower_count",
            "description")
    out = {k: info.get(k) for k in keys if k in info}
    if out.get("description"):
        out["description"] = out["description"][:1500]
    if out.get("tags"):
        out["tags"] = out["tags"][:40]
    return out
