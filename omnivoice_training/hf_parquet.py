#!/usr/bin/env python3
"""Stream rows out of HuggingFace parquet shards over plain HTTP range requests.

`datasets.load_dataset(..., streaming=True)` stalls for minutes on these repos -
the igbo corpus alone has 5,166 files and resolution happens before a single row
is yielded. It also insists on torchcodec to decode the audio column. Reading the
parquet directly avoids both problems, starts producing rows immediately, and
lets us pull only the columns we actually need.
"""
from __future__ import annotations

import io, json, os, re, threading, time

import pyarrow.parquet as pq
import requests

API = "https://huggingface.co"
_TL = threading.local()
TAIL = 256 * 1024


def _sess(token=None):
    s = getattr(_TL, "s", None)
    if s is None:
        s = requests.Session()
        if token:
            s.headers["Authorization"] = f"Bearer {token}"
        s.headers["User-Agent"] = "ohun-prepare/1.0"
        s.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=8, max_retries=3))
        _TL.s = s
    return s


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP ranges, with the parquet footer cached."""

    def __init__(self, url, size, token=None):
        self.url, self.size, self.pos, self.token = url, size, 0, token
        self._toff = max(0, size - TAIL)
        self._tail = self._get(self._toff, size - 1)

    def _get(self, start, end):
        for i in range(5):
            try:
                r = _sess(self.token).get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 120))
                if r.status_code not in (200, 206):
                    raise IOError(f"HTTP {r.status_code}")
                return r.content
            except Exception:
                if i == 4:
                    raise
                time.sleep(1.5 * (i + 1))

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        a, b = self.pos, self.pos + n - 1
        data = (self._tail[a - self._toff: b - self._toff + 1]
                if a >= self._toff else self._get(a, b))
        self.pos += len(data)
        return data


def _cache_path(repo, split):
    d = os.environ.get("OHUN_SHARD_CACHE", "/tmp/ohun_shard_cache")
    os.makedirs(d, exist_ok=True)
    key = f"{repo}_{split}".replace("/", "_")
    return os.path.join(d, key + ".json")


def list_shards(repo, split, token=None, repo_type="datasets", use_cache=True):
    """Return [(path, size)] for one split, newest shard-set only.

    Some repos carry orphaned shard sets from an earlier re-shard (igbo has both
    dev-*-of-00016 and dev-*-of-00041); including both would duplicate rows, so
    keep only the set written by the newest commit.
    """
    # The tree endpoint pages through every file in the repo (6,885 for ohun),
    # and this is called once per window - cache it.
    cp = _cache_path(repo, split)
    if use_cache and os.path.exists(cp):
        try:
            with open(cp) as fh:
                return [tuple(x) for x in json.load(fh)]
        except Exception:
            pass

    files, cursor = [], None
    while True:
        url = f"{API}/api/{repo_type}/{repo}/tree/main?recursive=1&expand=1"
        if cursor:
            url += "&cursor=" + cursor
        r = _sess(token).get(url, timeout=60)
        r.raise_for_status()
        batch = r.json()
        files += batch
        m = re.search(r"cursor=([^&>;]+)", r.headers.get("Link", ""))
        if m and batch:
            cursor = m.group(1)
        else:
            break

    pat = re.compile(rf"^data/(?:[^/]+/)?{re.escape(split)}-(\d{{5}})-of-(\d{{5}})\.parquet$")
    sets = {}
    for e in files:
        if e.get("type") != "file":
            continue
        m = pat.match(e["path"])
        if not m:
            continue
        tot = m.group(2)
        sz = (e.get("lfs") or {}).get("size") or e.get("size", 0)
        date = (e.get("lastCommit") or {}).get("date", "")
        sets.setdefault(tot, {"files": [], "newest": ""})
        sets[tot]["files"].append((e["path"], sz))
        sets[tot]["newest"] = max(sets[tot]["newest"], date)
    if not sets:
        return []
    best = max(sets.values(), key=lambda v: (v["newest"], len(v["files"])))
    out = sorted(best["files"])
    if use_cache:
        try:
            tmp = cp + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(out, fh)
            os.replace(tmp, cp)
        except Exception:
            pass
    return out


def _read_shard(repo, path, size, columns, token, repo_type):
    url = f"{API}/{repo_type}/{repo}/resolve/main/{path}"
    pf = pq.ParquetFile(HttpFile(url, size, token))
    names = pf.schema_arrow.names
    cols = [c for c in (columns or names) if c in names] or None
    out = []
    for rg in range(pf.metadata.num_row_groups):
        out.extend(pf.read_row_group(rg, columns=cols).to_pylist())
    return out


def iter_rows(repo, split, columns=None, token=None, shuffle_seed=None,
              repo_type="datasets", workers=8, skip_shards=0, max_shards=None):
    """Yield row dicts, prefetching several shards concurrently.

    A single sequential reader is latency-bound (~1.8 MB/s here); fetching a
    few shards in parallel and handing rows off through a bounded queue keeps
    the link busy without ever holding more than `workers` shards in memory.
    """
    import queue as _q
    from concurrent.futures import ThreadPoolExecutor

    shards = list_shards(repo, split, token=token, repo_type=repo_type)
    if shuffle_seed is not None:
        import random
        random.Random(shuffle_seed).shuffle(shards)
    shards = shards[skip_shards:]
    if max_shards:
        shards = shards[:max_shards]
    if not shards:
        return

    out = _q.Queue(maxsize=workers)
    sentinel = object()

    def work(item):
        path, size = item
        try:
            out.put(_read_shard(repo, path, size, columns, token, repo_type))
        except Exception as e:
            print(f"    WARN shard {path} unreadable: {e!r}", flush=True)
            out.put([])

    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = [ex.submit(work, s) for s in shards]
        done = 0
        while done < len(futs):
            rows = out.get()
            done += 1
            for row in rows:
                yield row
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
