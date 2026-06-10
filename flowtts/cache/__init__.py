"""Distributed cache layer for FlowTTS audio generation.

Architecture:
  L1 (LocalCache)        — per-process LRU in-memory index over local SSD files
  Distributed lock       — Redis SET NX EX stampede prevention
  StorageBackend         — pluggable file I/O (Local / MinIO / S3)
  CacheManager           — orchestrates all layers, exposes simple get/put API

Phase 1 (current): local filesystem storage + Redis distributed locks.
Phase 2 (future):  swap LocalStorageBackend → MinIOStorageBackend without touching
                   any business logic — only the StorageBackend implementation changes.
"""

from flowtts.cache.cache_key import CacheKey, make_cache_key
from flowtts.cache.storage import StorageBackend, LocalStorageBackend, GCSStorageBackend
from flowtts.cache.local_cache import LocalCache
from flowtts.cache.distributed_lock import DistributedLock, LockResult
from flowtts.cache.cache_manager import CacheManager, get_cache_manager

__all__ = [
    "CacheKey",
    "make_cache_key",
    "StorageBackend",
    "LocalStorageBackend",
    "GCSStorageBackend",
    "LocalCache",
    "DistributedLock",
    "LockResult",
    "CacheManager",
    "get_cache_manager",
]
