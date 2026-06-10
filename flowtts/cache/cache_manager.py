"""CacheManager — orchestrates L1 local cache, distributed lock, and storage.

Full request flow
-----------------

  get(key)
    │
    ├─ L1 LocalCache hit?  ──────────────────────────────→ return bytes (L1_HIT)
    │
    ├─ StorageBackend.exists() ?  ───────────────────────→ read + L1.put + return (L2_HIT)
    │
    ├─ try_acquire(lock_key)
    │     │
    │     ├─ ACQUIRED  → caller generates audio → store_async(key, data)
    │     │                                              (MISS_GENERATE)
    │     │
    │     ├─ WAIT      → wait_for_peer() until cache appears
    │     │    │
    │     │    ├─ peer finished  → StorageBackend.read() → return (PEER_HIT)
    │     │    │
    │     │    └─ timeout        → caller generates audio (MISS_TIMEOUT)
    │     │
    │     └─ DEGRADED  → caller generates audio (MISS_DEGRADED)
    │
    └─ return None  (caller must synthesize)

Non-blocking writes
-------------------
store_async() submits the disk write + Redis metadata update + L1 put to a
ThreadPoolExecutor.  The calling coroutine returns as soon as the audio is
queued — the response is sent to the WebSocket client immediately, with the
background thread handling persistence.  This eliminates cache-write latency
from the critical path.

Tradeoffs:
  + Response latency is unaffected by cache write speed.
  + Write failures are non-fatal (only logged) — the next request re-generates.
  - A server crash between response and write-complete loses the cache entry
    for that request.  Acceptable for TTS audio (cheap to re-generate).

Legacy cache compatibility
--------------------------
The existing flat-layout caches (cached_data_simran/, etc.) use SHA256(raw_text)
as the filename stem without voice/model parameters in the key.  CacheManager
checks the legacy path first before checking the sharded storage, so those files
continue to be served without any migration.

Redis metadata
--------------
Each successfully cached entry stores a small JSON hash in Redis:
  HSET audio:<digest>  path <shard_path>  hits 0  created_at <epoch>

Callers can query Redis for hit counts, LFU eviction candidates, etc.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import time
from pathlib import Path
from typing import Any

from flowtts.cache.cache_key import CacheKey, make_cache_key, legacy_cache_key
from flowtts.cache.distributed_lock import DistributedLock, LockResult
from flowtts.cache.local_cache import LocalCache
from flowtts.cache.storage import StorageBackend, LocalStorageBackend, GCSStorageBackend

logger = logging.getLogger(__name__)

# Module-level singleton — initialised once by _init_cache_manager().
_manager: "CacheManager | None" = None


def get_cache_manager() -> "CacheManager | None":
    """Return the process-level CacheManager singleton (None if not initialised)."""
    return _manager


class GetResult:
    """Result of CacheManager.get()."""

    __slots__ = ("data", "source", "path")

    def __init__(self, data: bytes | None, source: str, path: str = "") -> None:
        self.data = data       # WAV bytes, or None on cache miss
        self.source = source   # "l1_hit", "l2_hit", "peer_hit", "miss", "legacy_hit"
        self.path = path       # storage path (for logging)

    @property
    def is_hit(self) -> bool:
        return self.data is not None


class CacheManager:
    """Orchestrates L1 LocalStorageBackend + L2 GCSStorageBackend + DistributedLock.

    Storage tiers
    -------------
    _storage (L1) : Local SSD — fast, per-server, 2 GB LRU-bounded.
    _l2_storage   : GCS — shared across all TTS servers, durable, unlimited.

    On read:  L1 in-memory LRU index → L1 disk → legacy flat dirs → L2 GCS
    On write: L1 disk (via background thread) + L2 GCS (via background thread)
    On hit:   GCS data is written back to L1 so next request is served locally.

    Parameters
    ----------
    storage          : L1 local storage backend (LocalStorageBackend).
    l2_storage       : L2 shared storage backend (GCSStorageBackend or None).
    lock             : distributed lock instance (None → no coordination).
    l1_cache         : in-process LRU index (None → skip in-memory layer).
    write_executor   : thread pool for background writes (None → new one created).
    legacy_dirs      : list of (voice_id, directory_path) for existing flat caches.
    redis_client     : Redis/Dragonfly client for metadata storage (None → skip).
    model_version    : TTS checkpoint identifier used in cache key computation.
    """

    def __init__(
        self,
        storage: StorageBackend,
        *,
        l2_storage: StorageBackend | None = None,
        lock: DistributedLock | None = None,
        l1_cache: LocalCache | None = None,
        write_executor: concurrent.futures.ThreadPoolExecutor | None = None,
        legacy_dirs: list[tuple[str, Path]] | None = None,
        redis_client: Any | None = None,
        model_version: str = "",
    ) -> None:
        self._storage = storage          # L1: local SSD
        self._l2_storage = l2_storage    # L2: GCS (None if not configured)
        self._lock = lock
        self._l1 = l1_cache
        self._executor = write_executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="cache_write"
        )
        self._legacy_dirs: dict[str, Path] = {}
        for voice_id, path in (legacy_dirs or []):
            if Path(path).exists():
                self._legacy_dirs[voice_id] = Path(path)
        self._redis = redis_client
        self._model_version = model_version

        # Metrics counters (incremented internally; read by metrics module).
        self.stat_l1_hits: int = 0
        self.stat_l2_hits: int = 0
        self.stat_legacy_hits: int = 0
        self.stat_peer_hits: int = 0
        self.stat_misses: int = 0
        self.stat_lock_waits: int = 0
        self.stat_duplicate_prevented: int = 0
        self.stat_write_errors: int = 0

    # ------------------------------------------------------------------
    # Build a CacheKey for this manager's model version
    # ------------------------------------------------------------------

    def make_key(
        self,
        text: str,
        voice_id: str = "",
        speaking_rate: float = 1.0,
        language: str = "",
        extra_params: dict | None = None,
    ) -> CacheKey:
        return make_cache_key(
            text,
            voice_id=voice_id,
            model_version=self._model_version,
            speaking_rate=speaking_rate,
            language=language,
            extra_params=extra_params,
        )

    # ------------------------------------------------------------------
    # Main get path
    # ------------------------------------------------------------------

    async def get(self, key: CacheKey) -> GetResult:
        """Look up audio for *key*.  Returns GetResult with data=None on miss."""

        # 1. L1 in-process check (no I/O)
        if self._l1 is not None and self._l1.get(key.digest):
            data = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._storage.read, key.digest
            )
            if data is not None:
                self.stat_l1_hits += 1
                return GetResult(data, "l1_hit", self._storage.shard_path(key.digest))
            # L1 index was stale (file deleted externally) — remove from index.
            if self._l1 is not None:
                self._l1.evict(key.digest)

        # 2. Legacy flat-directory check (existing 22 GB+ per-voice caches)
        legacy_data = await self._check_legacy(key)
        if legacy_data is not None:
            self.stat_legacy_hits += 1
            # Warm L1 with the legacy entry so next hit is faster.
            if self._l1 is not None:
                self._l1.put(key.digest, len(legacy_data))
            return GetResult(legacy_data, "legacy_hit")

        # 3. L2 shared storage check — GCS when configured, else local sharded tree.
        #    GCS is shared across all servers so a hit here means another server
        #    wrote it (or it was pre-populated).  We write it back to L1 so the
        #    next request on this node is served locally.
        _l2 = self._l2_storage or self._storage
        l2_exists = await asyncio.get_event_loop().run_in_executor(
            self._executor, _l2.exists, key.digest
        )
        if l2_exists:
            data = await asyncio.get_event_loop().run_in_executor(
                self._executor, _l2.read, key.digest
            )
            if data is not None:
                self.stat_l2_hits += 1
                if self._l1 is not None:
                    self._l1.put(key.digest, len(data))
                # Write-back to local L1 storage so future hits are served from disk.
                if self._l2_storage is not None and self._storage is not self._l2_storage:
                    try:
                        self._executor.submit(self._storage.write, key.digest, data)
                    except Exception:
                        pass
                await self._increment_hits(key)
                return GetResult(data, "l2_hit", _l2.shard_path(key.digest))

        # 4. Distributed lock (DragonflyDB / Redis) — stampede prevention.
        #    The `_exists_async` closure checks GCS (or local if no GCS) so that
        #    waiters can detect a peer's upload before the lock is released.
        if self._lock is not None:
            lock_result = await self._lock.try_acquire(key.lock_key)

            if lock_result == LockResult.WAIT:
                self.stat_lock_waits += 1
                logger.info("cache_manager.lock_wait digest=%s", key.digest[:16])

                async def _exists_async() -> bool:
                    return await asyncio.get_event_loop().run_in_executor(
                        self._executor, _l2.exists, key.digest
                    )

                found = await self._lock.wait_for_peer(key.lock_key, _exists_async)
                if found:
                    data = await asyncio.get_event_loop().run_in_executor(
                        self._executor, _l2.read, key.digest
                    )
                    if data is not None:
                        self.stat_peer_hits += 1
                        self.stat_duplicate_prevented += 1
                        if self._l1 is not None:
                            self._l1.put(key.digest, len(data))
                        await self._increment_hits(key)
                        return GetResult(data, "peer_hit", _l2.shard_path(key.digest))
                # Peer timeout or unreadable — fall through to generate.
                self.stat_misses += 1
                return GetResult(None, "miss")

            # ACQUIRED or DEGRADED — caller will generate and call store_async().
            # Lock is released by store_async() after write completes.
            if lock_result == LockResult.ACQUIRED:
                logger.debug("cache_manager.lock_acquired digest=%s", key.digest[:16])

        self.stat_misses += 1
        return GetResult(None, "miss")

    # ------------------------------------------------------------------
    # Non-blocking store path
    # ------------------------------------------------------------------

    def store_async(self, key: CacheKey, data: bytes, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Submit a background write for *data* under *key*.

        Returns immediately — the caller should NOT wait for this.
        The distributed lock (if any) is released after the write completes.
        """
        lp = loop or asyncio.get_event_loop()
        future = self._executor.submit(self._write_sync, key, data)
        future.add_done_callback(
            lambda f: lp.call_soon_threadsafe(self._on_write_done, key, f)
        )

    def _write_sync(self, key: CacheKey, data: bytes) -> None:
        """Synchronous write — runs in the thread pool.

        Writes to L1 (local storage) first, then to L2 (GCS) if configured.
        L1 failure is non-fatal when L2 is present — the data is still durable.
        L2 failure is logged but does not prevent L1 from completing.
        """
        # L1: local SSD
        try:
            self._storage.write(key.digest, data)
            if self._l1 is not None:
                self._l1.put(key.digest, len(data))
            logger.debug("cache_manager.l1_write_ok digest=%s bytes=%d",
                         key.digest[:16], len(data))
        except Exception as exc:
            logger.error("cache_manager.l1_write_error digest=%s: %s", key.digest[:16], exc)
            if self._l2_storage is None:
                raise  # no L2 fallback — propagate

        # L2: GCS (shared across all servers)
        if self._l2_storage is not None:
            try:
                self._l2_storage.write(key.digest, data)
                logger.debug("cache_manager.l2_write_ok digest=%s path=%s",
                             key.digest[:16], self._l2_storage.shard_path(key.digest))
            except Exception as exc:
                logger.error("cache_manager.l2_write_error digest=%s: %s",
                             key.digest[:16], exc)
                # Non-fatal: L1 already has the data; GCS write will be retried
                # on the next cache miss for the same content.

    def _on_write_done(self, key: CacheKey, future: concurrent.futures.Future) -> None:
        """Called on the event loop thread after the background write completes."""
        if future.exception() is not None:
            self.stat_write_errors += 1

        # Release the distributed lock (fire-and-forget coroutine).
        if self._lock is not None:
            asyncio.ensure_future(self._release_and_record(key))
        else:
            asyncio.ensure_future(self._record_metadata(key))

    async def _release_and_record(self, key: CacheKey) -> None:
        await self._lock.release(key.lock_key)
        await self._record_metadata(key)

    async def _record_metadata(self, key: CacheKey) -> None:
        """Write lightweight metadata to Redis (hits counter, path, created_at)."""
        if self._redis is None:
            return
        meta_key = key.meta_key
        shard = self._storage.shard_path(key.digest)
        try:
            await self._redis.hset(meta_key, mapping={
                "path": shard,
                "hits": 0,
                "created_at": str(int(time.time())),
                "voice_id": key.voice_id,
                "model_version": key.model_version,
            })
            # No TTL on metadata — audio is persistent.
        except Exception as exc:
            logger.debug("cache_manager.metadata_error key=%s: %s", meta_key, exc)

    async def _increment_hits(self, key: CacheKey) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.hincrby(key.meta_key, "hits", 1)
            await self._redis.hset(key.meta_key, "last_access", str(int(time.time())))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Legacy flat-directory lookup
    # ------------------------------------------------------------------

    async def _check_legacy(self, key: CacheKey) -> bytes | None:
        """Check per-voice flat-layout legacy directories using legacy key format."""
        voice_dir = self._legacy_dirs.get(key.voice_id)
        if voice_dir is None:
            return None

        leg_hash = legacy_cache_key(key.normalised_text)
        legacy_path = voice_dir / f"{leg_hash}.wav"

        if not legacy_path.exists():
            # Also try with the raw (un-normalised) text in case normalisation
            # changed whitespace that was in the original cache.
            return None

        try:
            data = await asyncio.get_event_loop().run_in_executor(
                self._executor, legacy_path.read_bytes
            )
            return data
        except OSError:
            return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        d: dict = {
            "l1_hits": self.stat_l1_hits,
            "l2_hits": self.stat_l2_hits,
            "legacy_hits": self.stat_legacy_hits,
            "peer_hits": self.stat_peer_hits,
            "misses": self.stat_misses,
            "lock_waits": self.stat_lock_waits,
            "duplicate_prevented": self.stat_duplicate_prevented,
            "write_errors": self.stat_write_errors,
        }
        total_hits = (
            self.stat_l1_hits + self.stat_l2_hits +
            self.stat_legacy_hits + self.stat_peer_hits
        )
        total = total_hits + self.stat_misses
        d["hit_rate"] = round(total_hits / total, 4) if total else 0.0
        if self._l1 is not None:
            d["l1"] = self._l1.stats()
        return d


# ---------------------------------------------------------------------------
# Process-level singleton init
# ---------------------------------------------------------------------------

def init_cache_manager(
    *,
    storage_root: str | Path,
    legacy_dirs: list[tuple[str, Path]] | None = None,
    # DragonflyDB / Redis — distributed locks + metadata
    redis_url: str | None = None,
    lock_ttl_s: float = 60.0,
    wait_timeout_s: float = 30.0,
    # L1 local SSD
    l1_max_bytes: int = 2 * 1024 ** 3,
    # L2 Google Cloud Storage
    gcs_bucket: str | None = None,
    gcs_prefix: str = "tts-cache",
    gcs_project: str | None = None,
    gcs_credentials_file: str | None = None,
    gcs_timeout_s: float = 30.0,
    # Misc
    model_version: str = "",
    write_workers: int = 4,
) -> "CacheManager":
    """Create and register the process-level CacheManager singleton.

    Architecture
    ------------
      L1 : LocalStorageBackend (local SSD, 2 GB LRU)
      L2 : GCSStorageBackend   (gs://<gcs_bucket>/<gcs_prefix>/…) — optional
      Locks: DragonflyDB/Redis (redis_url) — optional

    Call once during server startup.  Subsequent calls return the existing
    singleton unchanged (idempotent).

    Parameters
    ----------
    storage_root          : root directory for the sharded local cache tree.
    legacy_dirs           : [(voice_id, path), ...] for existing flat caches.
    redis_url             : DragonflyDB/Redis URL, e.g. "redis://dragonfly:6379/1".
                            None → no distributed locks.
    lock_ttl_s            : lock auto-expiry (seconds).
    wait_timeout_s        : max wait time for a peer lock holder (seconds).
    l1_max_bytes          : L1 in-memory LRU budget in bytes.
    gcs_bucket            : GCS bucket name.  None → skip GCS (Phase 1 local only).
    gcs_prefix            : object key prefix inside the bucket (default "tts-cache").
    gcs_project           : GCP project ID (None → infer from ADC).
    gcs_credentials_file  : path to service-account JSON key (None → use ADC).
    gcs_timeout_s         : per-GCS-op timeout in seconds.
    model_version         : TTS checkpoint tag baked into every cache key.
    write_workers         : background write thread pool size.
    """
    global _manager
    if _manager is not None:
        return _manager

    # L1: local sharded storage
    storage = LocalStorageBackend(storage_root)

    # L1 in-memory LRU index — optionally pre-populate from legacy flat dirs
    l1 = LocalCache(max_bytes=l1_max_bytes, storage=storage)
    for _, path in (legacy_dirs or []):
        if Path(path).exists():
            count = l1.scan_directory(path)
            logger.info("cache_manager.l1_scan dir=%s entries=%d", path, count)

    # L2: Google Cloud Storage
    l2_storage = None
    if gcs_bucket:
        try:
            credentials = None
            if gcs_credentials_file:
                from google.oauth2 import service_account  # type: ignore[import]
                credentials = service_account.Credentials.from_service_account_file(
                    gcs_credentials_file,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            l2_storage = GCSStorageBackend(
                bucket=gcs_bucket,
                prefix=gcs_prefix,
                project=gcs_project,
                credentials=credentials,
                timeout=gcs_timeout_s,
            )
            logger.info(
                "cache_manager.gcs_connected bucket=%s prefix=%s",
                gcs_bucket, gcs_prefix,
            )
        except ImportError:
            logger.warning(
                "cache_manager: google-cloud-storage package not available — "
                "running without GCS L2 cache.  "
                "Install it with: pip install google-cloud-storage"
            )
        except Exception as exc:
            logger.warning(
                "cache_manager.gcs_connect_error: %s — running without GCS", exc
            )

    # DragonflyDB / Redis — distributed locks + metadata
    redis_client = None
    lock = None
    if redis_url:
        try:
            import redis.asyncio as aioredis  # type: ignore[import]
            redis_client = aioredis.from_url(redis_url, decode_responses=True)
            lock = DistributedLock(
                redis_client,
                lock_ttl_s=lock_ttl_s,
                wait_timeout_s=wait_timeout_s,
            )
            logger.info("cache_manager.dragonfly_connected url=%s", redis_url)
        except ImportError:
            logger.warning(
                "cache_manager: redis package not available — "
                "running without distributed locks.  "
                "Install it with: pip install redis"
            )
        except Exception as exc:
            logger.warning(
                "cache_manager.dragonfly_connect_error: %s — running without locks", exc
            )

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=write_workers,
        thread_name_prefix="cache_write",
    )

    _manager = CacheManager(
        storage,
        l2_storage=l2_storage,
        lock=lock,
        l1_cache=l1,
        write_executor=executor,
        legacy_dirs=legacy_dirs,
        redis_client=redis_client,
        model_version=model_version,
    )
    logger.info(
        "cache_manager.init storage=%s l1_max_gb=%.1f gcs=%s dragonfly=%s",
        storage_root,
        l1_max_bytes / 1024 ** 3,
        f"gs://{gcs_bucket}/{gcs_prefix}" if gcs_bucket else "disabled",
        "yes" if redis_client else "no",
    )
    return _manager
