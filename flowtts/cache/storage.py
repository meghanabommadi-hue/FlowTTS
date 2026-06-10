"""Storage backend abstraction for TTS audio cache.

Phase 1: LocalStorageBackend — atomic writes to a sharded local SSD directory.
Phase 2: GCSStorageBackend  — Google Cloud Storage shared across all TTS servers.

Architecture
------------
  TTS Servers
      │
      ▼
  L1 LocalStorageBackend  (per-server SSD, 2 GB LRU)
      │  miss
      ▼
  L2 GCSStorageBackend    (gs://bucket/tts-cache/ab/cd/<digest>.wav)
      │  miss
      ▼
  GPU synthesis  →  upload to GCS  →  store locally

GCS Object Layout
-----------------
Objects are keyed as:

  <prefix>/<d0d1>/<d2d3>/<digest>.wav

This mirrors the local sharding layout and avoids flat-bucket performance
problems when the bucket contains millions of objects.

StorageBackend interface
------------------------
  exists(digest)        → bool
  read(digest)          → bytes | None
  write(digest, data)   → None   (atomic for local; GCS PUT is atomic by design)
  delete(digest)        → None
  shard_path(digest)    → str    (debug / metrics)
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Abstract interface for audio blob storage.

    Implementations must be thread-safe.  Async callers should run blocking
    methods in a ThreadPoolExecutor rather than blocking the event loop.
    """

    @abstractmethod
    def exists(self, digest: str) -> bool:
        """Return True if the audio blob for *digest* is available."""

    @abstractmethod
    def read(self, digest: str) -> bytes | None:
        """Return the WAV bytes for *digest*, or None if not found."""

    @abstractmethod
    def write(self, digest: str, data: bytes) -> None:
        """Durably store *data* under *digest*.  Must be atomic."""

    @abstractmethod
    def delete(self, digest: str) -> None:
        """Remove the blob for *digest* (no-op if absent)."""

    @abstractmethod
    def shard_path(self, digest: str) -> str:
        """Return a human-readable location string for logging / metrics."""


# ---------------------------------------------------------------------------
# Local filesystem backend (L1)
# ---------------------------------------------------------------------------

class LocalStorageBackend(StorageBackend):
    """Sharded local-filesystem storage with atomic writes.

    Directory layout: <root>/<d0d1>/<d2d3>/<digest>.wav

    Writes go to a sibling temp file first, then are atomically renamed so
    readers never see a partially-written file even if the process crashes.

    Parameters
    ----------
    root     : base directory for the cache tree.
    ext      : file extension (default ".wav").
    dir_mode : permission bits for auto-created leaf directories.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        ext: str = ".wav",
        dir_mode: int = 0o755,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._ext = ext
        self._dir_mode = dir_mode

    def _leaf_dir(self, digest: str) -> Path:
        return self._root / digest[:2] / digest[2:4]

    def _blob_path(self, digest: str) -> Path:
        return self._leaf_dir(digest) / f"{digest}{self._ext}"

    def exists(self, digest: str) -> bool:
        return self._blob_path(digest).exists()

    def read(self, digest: str) -> bytes | None:
        try:
            return self._blob_path(digest).read_bytes()
        except (FileNotFoundError, OSError):
            return None

    def write(self, digest: str, data: bytes) -> None:
        leaf = self._leaf_dir(digest)
        leaf.mkdir(parents=True, exist_ok=True, mode=self._dir_mode)
        blob = self._blob_path(digest)
        fd, tmp_path = tempfile.mkstemp(dir=leaf, prefix=f".tmp_{digest}_", suffix=self._ext)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, blob)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def delete(self, digest: str) -> None:
        try:
            self._blob_path(digest).unlink()
        except FileNotFoundError:
            pass

    def shard_path(self, digest: str) -> str:
        return str(self._blob_path(digest))

    @property
    def root(self) -> Path:
        return self._root


# ---------------------------------------------------------------------------
# Google Cloud Storage backend (L2)
# ---------------------------------------------------------------------------

class GCSStorageBackend(StorageBackend):
    """Google Cloud Storage backend for shared cross-server audio cache.

    Object key layout mirrors the local sharding layout:

        gs://<bucket>/<prefix>/<d0d1>/<d2d3>/<digest>.wav

    This backend requires the ``google-cloud-storage`` package.
    Install it with:  pip install google-cloud-storage

    Authentication follows the standard Application Default Credentials (ADC)
    lookup chain:
      1. GOOGLE_APPLICATION_CREDENTIALS env var (service-account JSON key)
      2. gcloud CLI credentials   (``gcloud auth application-default login``)
      3. Workload Identity (GKE / Cloud Run metadata server)

    Parameters
    ----------
    bucket         : GCS bucket name, e.g. "kapturecx-ml".
    prefix         : Object key prefix, e.g. "tts-cache".  No trailing slash.
    project        : GCP project ID (required if not inferable from ADC).
    credentials    : ``google.oauth2.credentials.Credentials`` or
                     ``google.oauth2.service_account.Credentials`` — or None
                     to use ADC.
    content_type   : MIME type for stored objects (default "audio/wav").
    timeout        : per-operation HTTP timeout in seconds (default 30).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "tts-cache",
        *,
        project: str | None = None,
        credentials=None,
        content_type: str = "audio/wav",
        timeout: float = 30.0,
    ) -> None:
        try:
            from google.cloud import storage as gcs  # type: ignore[import]
            from google.api_core.exceptions import NotFound  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "GCSStorageBackend requires the 'google-cloud-storage' package.\n"
                "Install it with:  pip install google-cloud-storage"
            ) from exc

        self._client = gcs.Client(project=project, credentials=credentials)
        self._bucket = self._client.bucket(bucket)
        self._prefix = prefix.rstrip("/")
        self._content_type = content_type
        self._timeout = timeout
        self._NotFound = NotFound

    def _object_name(self, digest: str) -> str:
        return f"{self._prefix}/{digest[:2]}/{digest[2:4]}/{digest}.wav"

    def exists(self, digest: str) -> bool:
        blob = self._bucket.blob(self._object_name(digest))
        return blob.exists(timeout=self._timeout)

    def read(self, digest: str) -> bytes | None:
        blob = self._bucket.blob(self._object_name(digest))
        try:
            return blob.download_as_bytes(timeout=self._timeout)
        except self._NotFound:
            return None
        except Exception:
            return None

    def write(self, digest: str, data: bytes) -> None:
        """Upload *data* to GCS.

        GCS PUT is atomic at the object level — the object either appears
        fully or not at all, so no temp-rename dance is needed.
        """
        blob = self._bucket.blob(self._object_name(digest))
        blob.upload_from_string(
            data,
            content_type=self._content_type,
            timeout=self._timeout,
        )

    def delete(self, digest: str) -> None:
        blob = self._bucket.blob(self._object_name(digest))
        try:
            blob.delete(timeout=self._timeout)
        except self._NotFound:
            pass

    def shard_path(self, digest: str) -> str:
        return f"gs://{self._bucket.name}/{self._object_name(digest)}"
