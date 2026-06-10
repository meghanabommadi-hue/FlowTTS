"""L1 local cache — per-process LRU index over local WAV files.

This layer sits in front of the distributed shared storage.  It tracks which
digests are present on this node's local SSD and maintains an LRU eviction
policy so the cache never grows beyond a configurable byte limit.

Architecture
------------
  Request
    │
    ▼
  LocalCache.get(digest)   ← checks LRU index (dict lookup, O(1))
    │
    ├─ HIT  → return path, promote to MRU end
    │
    └─ MISS → caller checks StorageBackend / generates audio
                  └─ on store: LocalCache.put(digest, size_bytes)

Key design decisions
--------------------
- The LRU is an in-memory *index* over files that already exist on disk.
  It does not store WAV bytes itself — that would double the memory footprint.
- Size tracking is per-file byte count, not entry count, matching the
  operator-facing "max 2 GB" mental model.
- Eviction is synchronous and happens during put() so the cache never
  silently exceeds the limit.  In practice put() runs in a background
  thread (see CacheManager) so it doesn't block synthesis.
- Thread safety: a single threading.Lock guards all mutations.  The lock
  is held only for dict operations, not for disk I/O.
- The legacy flat-directory caches (cached_data_simran/, etc.) can be
  pre-populated into this index at startup via scan_directory().
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterator


class LocalCache:
    """LRU in-memory index with byte-level capacity tracking.

    Parameters
    ----------
    max_bytes     : eviction threshold in bytes (default 2 GB).
    storage       : optional :class:`~flowtts.cache.storage.LocalStorageBackend`
                    used for eviction (delete) and path lookup.  If None,
                    eviction only removes the entry from the in-memory index
                    (the file is left on disk — useful when the backend manages
                    its own lifecycle).
    """

    def __init__(
        self,
        max_bytes: int = 2 * 1024 ** 3,   # 2 GB
        storage: "LocalStorageBackend | None" = None,
    ) -> None:
        from flowtts.cache.storage import LocalStorageBackend
        self._max_bytes = max_bytes
        self._storage: LocalStorageBackend | None = storage

        # OrderedDict preserves insertion order; we use move_to_end() to
        # implement LRU: oldest entry at front, most-recently-used at back.
        self._index: OrderedDict[str, int] = OrderedDict()   # digest → size_bytes
        self._total_bytes: int = 0
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, digest: str) -> bool:
        """Return True and promote to MRU if *digest* is in L1, else False."""
        with self._lock:
            if digest in self._index:
                self._index.move_to_end(digest)
                self._hits += 1
                return True
            self._misses += 1
            return False

    def put(self, digest: str, size_bytes: int) -> None:
        """Register *digest* in the L1 index; evict LRU entries if over budget.

        If *digest* is already present, its position is updated to MRU and
        the size is refreshed (handles re-writes of the same file).
        """
        with self._lock:
            if digest in self._index:
                old_size = self._index[digest]
                self._total_bytes -= old_size
                self._index.move_to_end(digest)
                self._index[digest] = size_bytes
                self._total_bytes += size_bytes
            else:
                self._index[digest] = size_bytes
                self._total_bytes += size_bytes

            self._evict_lru()

    def evict(self, digest: str) -> None:
        """Explicitly remove *digest* from L1 (and optionally from disk)."""
        with self._lock:
            self._evict_one(digest, delete_file=True)

    def scan_directory(self, directory: str | Path, ext: str = ".wav") -> int:
        """Populate the L1 index by scanning an existing flat cache directory.

        Useful for importing the legacy per-voice caches (cached_data_simran/,
        cached_data_tara/, etc.) so their files are visible to LocalCache
        without migrating them to the sharded layout first.

        Returns the number of files registered.
        """
        d = Path(directory)
        if not d.exists():
            return 0
        count = 0
        with self._lock:
            for p in d.iterdir():
                if p.suffix == ext and len(p.stem) == 64:  # SHA256 hex = 64 chars
                    size = p.stat().st_size
                    if p.stem not in self._index:
                        self._index[p.stem] = size
                        self._total_bytes += size
                        count += 1
            self._evict_lru()
        return count

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._index)

    @property
    def hit_count(self) -> int:
        return self._hits

    @property
    def miss_count(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._index),
                "total_bytes": self._total_bytes,
                "max_bytes": self._max_bytes,
                "utilisation_pct": round(100 * self._total_bytes / self._max_bytes, 1),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
            }

    # ------------------------------------------------------------------
    # Internal helpers  (must be called with _lock held)
    # ------------------------------------------------------------------

    def _evict_lru(self) -> None:
        """Evict least-recently-used entries until total_bytes ≤ max_bytes."""
        while self._total_bytes > self._max_bytes and self._index:
            oldest_digest = next(iter(self._index))
            self._evict_one(oldest_digest, delete_file=True)

    def _evict_one(self, digest: str, *, delete_file: bool) -> None:
        """Remove *digest* from index (and optionally from disk).

        Caller must hold _lock.
        """
        size = self._index.pop(digest, 0)
        self._total_bytes = max(0, self._total_bytes - size)
        if delete_file and self._storage is not None:
            # Release lock before disk I/O to avoid holding it during syscalls.
            # Re-acquiring is safe; the entry is already removed from the index.
            self._lock.release()
            try:
                self._storage.delete(digest)
            finally:
                self._lock.acquire()

    def __iter__(self) -> Iterator[str]:
        """Iterate digests from LRU (oldest) to MRU (newest)."""
        with self._lock:
            return iter(list(self._index.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    def __contains__(self, digest: str) -> bool:
        with self._lock:
            return digest in self._index
