"""Unit tests for the distributed cache layer.

Run with:
    cd /home/ubuntu/FlowTTS
    .venv/bin/python3 -m unittest flowtts.test.test_distributed_cache -v

No Redis or GPU required — all tests use local storage and mock/no-op lock.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from flowtts.cache.cache_key import (
    make_cache_key,
    legacy_cache_key,
    _normalise_text,
)
from flowtts.cache.storage import LocalStorageBackend
from flowtts.cache.local_cache import LocalCache
from flowtts.cache.distributed_lock import DistributedLock, LockResult
from flowtts.cache.cache_manager import CacheManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(n_samples: int = 1600, sample_rate: int = 16000) -> bytes:
    n_channels, bits = 1, 16
    data_bytes = n_samples * n_channels * (bits // 8)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_bytes))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                           sample_rate * n_channels * (bits // 8),
                           n_channels * (bits // 8), bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_bytes))
    buf.write(b"\x00" * data_bytes)
    return buf.getvalue()


DUMMY_WAV = _make_wav()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# cache_key.py
# ===========================================================================

class TestCacheKey(unittest.TestCase):

    def test_basic_roundtrip(self):
        key = make_cache_key("Hello world", voice_id="simran", model_version="v1")
        self.assertEqual(len(key.digest), 64)
        self.assertEqual(key.voice_id, "simran")
        self.assertEqual(key.model_version, "v1")
        self.assertEqual(key.speaking_rate, 1.0)

    def test_normalisation_collapses_whitespace(self):
        k1 = make_cache_key("Hello   world", voice_id="tara")
        k2 = make_cache_key("Hello world",   voice_id="tara")
        self.assertEqual(k1.digest, k2.digest)

    def test_normalisation_strips_outer_whitespace(self):
        k1 = make_cache_key("  Hello world  ", voice_id="tara")
        k2 = make_cache_key("Hello world",     voice_id="tara")
        self.assertEqual(k1.digest, k2.digest)

    def test_different_voice_different_digest(self):
        k1 = make_cache_key("Hello", voice_id="simran")
        k2 = make_cache_key("Hello", voice_id="tara")
        self.assertNotEqual(k1.digest, k2.digest)

    def test_different_model_version_different_digest(self):
        k1 = make_cache_key("Hello", model_version="v1")
        k2 = make_cache_key("Hello", model_version="v2")
        self.assertNotEqual(k1.digest, k2.digest)

    def test_different_speaking_rate_different_digest(self):
        k1 = make_cache_key("Hello", speaking_rate=1.0)
        k2 = make_cache_key("Hello", speaking_rate=0.85)
        self.assertNotEqual(k1.digest, k2.digest)

    def test_speaking_rate_rounding(self):
        k1 = make_cache_key("Hello", speaking_rate=1.0000001)
        k2 = make_cache_key("Hello", speaking_rate=1.0000009)
        self.assertEqual(k1.digest, k2.digest)

    def test_different_language_different_digest(self):
        k1 = make_cache_key("Hello", language="en-US")
        k2 = make_cache_key("Hello", language="hi-IN")
        self.assertNotEqual(k1.digest, k2.digest)

    def test_extra_params_sorted(self):
        k1 = make_cache_key("Hello", extra_params={"a": 1, "b": 2})
        k2 = make_cache_key("Hello", extra_params={"b": 2, "a": 1})
        self.assertEqual(k1.digest, k2.digest)

    def test_field_isolation_via_nul(self):
        k1 = make_cache_key("\x00c", voice_id="ab")
        k2 = make_cache_key("c",     voice_id="ab\x00")
        self.assertNotEqual(k1.digest, k2.digest)

    def test_lock_and_meta_keys(self):
        key = make_cache_key("Hello", voice_id="simran")
        self.assertTrue(key.lock_key.startswith("lock:"))
        self.assertTrue(key.meta_key.startswith("audio:"))
        self.assertIn(key.digest, key.lock_key)
        self.assertIn(key.digest, key.meta_key)

    def test_to_dict_roundtrip(self):
        key = make_cache_key("Hello", voice_id="tara", model_version="v2",
                              speaking_rate=0.9, language="hi-IN",
                              extra_params={"pitch": 0})
        d = key.to_dict()
        self.assertEqual(d["voice_id"], "tara")
        self.assertEqual(d["model_version"], "v2")
        self.assertEqual(d["speaking_rate"], 0.9)

    def test_legacy_key_matches_sha256(self):
        text = "नमस्ते"
        self.assertEqual(
            legacy_cache_key(text),
            hashlib.sha256(text.encode("utf-8")).hexdigest()
        )

    def test_empty_text(self):
        k = make_cache_key("", voice_id="simran")
        self.assertEqual(len(k.digest), 64)
        self.assertEqual(k.normalised_text, "")

    def test_unicode_nfc_normalisation(self):
        import unicodedata
        text_nfd = unicodedata.normalize("NFD", "é")
        text_nfc = unicodedata.normalize("NFC", "é")
        self.assertEqual(
            make_cache_key(text_nfc).digest,
            make_cache_key(text_nfd).digest,
        )

    def test_deterministic_across_calls(self):
        args = dict(voice_id="tara", model_version="v3",
                    speaking_rate=1.1, language="hi-IN")
        k1 = make_cache_key("Hello", **args)
        k2 = make_cache_key("Hello", **args)
        self.assertEqual(k1.digest, k2.digest)

    def test_different_texts_different_digest(self):
        k1 = make_cache_key("Hello")
        k2 = make_cache_key("World")
        self.assertNotEqual(k1.digest, k2.digest)


# ===========================================================================
# storage.py
# ===========================================================================

class TestLocalStorageBackend(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _backend(self):
        return LocalStorageBackend(self.tmpdir)

    def test_write_then_exists(self):
        b = self._backend()
        digest = "a" * 64
        self.assertFalse(b.exists(digest))
        b.write(digest, DUMMY_WAV)
        self.assertTrue(b.exists(digest))

    def test_write_then_read(self):
        b = self._backend()
        digest = "b" * 64
        b.write(digest, DUMMY_WAV)
        self.assertEqual(b.read(digest), DUMMY_WAV)

    def test_sharding_layout(self):
        b = self._backend()
        digest = "abcdef" + "0" * 58
        b.write(digest, DUMMY_WAV)
        expected = Path(self.tmpdir) / "ab" / "cd" / f"{digest}.wav"
        self.assertTrue(expected.exists())

    def test_no_partial_file_visible(self):
        b = self._backend()
        digest = "c" * 64
        b.write(digest, DUMMY_WAV)
        shard_dir = Path(self.tmpdir) / "cc" / "cc"
        files = list(shard_dir.iterdir())
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, f"{digest}.wav")

    def test_read_missing_returns_none(self):
        b = self._backend()
        self.assertIsNone(b.read("f" * 64))

    def test_delete(self):
        b = self._backend()
        digest = "d" * 64
        b.write(digest, DUMMY_WAV)
        b.delete(digest)
        self.assertFalse(b.exists(digest))

    def test_delete_missing_is_noop(self):
        b = self._backend()
        b.delete("e" * 64)  # must not raise

    def test_shard_path_string(self):
        b = self._backend()
        digest = "1234" + "0" * 60
        path = b.shard_path(digest)
        self.assertIn("12", path)
        self.assertIn("34", path)
        self.assertIn(digest, path)

    def test_concurrent_writes_same_digest(self):
        b = self._backend()
        digest = "7" * 64
        errors = []

        def _write():
            try:
                b.write(digest, DUMMY_WAV)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_write) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertFalse(errors)
        self.assertEqual(b.read(digest), DUMMY_WAV)


# ===========================================================================
# local_cache.py
# ===========================================================================

class TestLocalCache(unittest.TestCase):

    def test_put_and_get(self):
        lc = LocalCache(max_bytes=10 * 1024 ** 2)
        self.assertFalse(lc.get("aaa" + "0" * 61))
        lc.put("aaa" + "0" * 61, 100)
        self.assertTrue(lc.get("aaa" + "0" * 61))

    def test_hit_and_miss_counters(self):
        lc = LocalCache()
        d = "x" * 64
        lc.put(d, 50)
        lc.get(d)
        lc.get("y" * 64)
        self.assertEqual(lc.hit_count, 1)
        self.assertEqual(lc.miss_count, 1)

    def test_lru_eviction_order(self):
        lc = LocalCache(max_bytes=300)
        lc.put("a" * 64, 100)
        lc.put("b" * 64, 100)
        lc.put("c" * 64, 100)
        lc.put("d" * 64, 100)   # should evict "a"
        self.assertFalse(lc.get("a" * 64))
        self.assertTrue(lc.get("b" * 64))
        self.assertTrue(lc.get("c" * 64))
        self.assertTrue(lc.get("d" * 64))

    def test_access_promotes_to_mru(self):
        lc = LocalCache(max_bytes=300)
        lc.put("a" * 64, 100)
        lc.put("b" * 64, 100)
        lc.put("c" * 64, 100)
        lc.get("a" * 64)        # promote a → b is now LRU
        lc.put("d" * 64, 100)   # evict b
        self.assertFalse(lc.get("b" * 64))
        self.assertTrue(lc.get("a" * 64))

    def test_total_bytes_tracking(self):
        lc = LocalCache()
        lc.put("p" * 64, 1024)
        self.assertEqual(lc.total_bytes, 1024)
        lc.put("q" * 64, 2048)
        self.assertEqual(lc.total_bytes, 3072)

    def test_scan_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            for i in range(5):
                digest = hashlib.sha256(f"text{i}".encode()).hexdigest()
                (d / f"{digest}.wav").write_bytes(DUMMY_WAV)
            lc = LocalCache()
            count = lc.scan_directory(d)
            self.assertEqual(count, 5)
            self.assertEqual(lc.entry_count, 5)

    def test_scan_directory_skips_non_sha256(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            # Too short — not a valid SHA256 stem
            (d / "short.wav").write_bytes(DUMMY_WAV)
            # Too long
            (d / ("a" * 65 + ".wav")).write_bytes(DUMMY_WAV)
            lc = LocalCache()
            count = lc.scan_directory(d)
            self.assertEqual(count, 0)

    def test_stats_dict(self):
        lc = LocalCache(max_bytes=1024)
        lc.put("z" * 64, 512)
        s = lc.stats()
        self.assertIn("entries", s)
        self.assertIn("total_bytes", s)
        self.assertIn("hit_rate", s)
        self.assertEqual(s["total_bytes"], 512)


# ===========================================================================
# distributed_lock.py
# ===========================================================================

class TestDistributedLockDegraded(unittest.TestCase):

    def test_acquire_returns_degraded(self):
        lock = DistributedLock(None)
        result = run(lock.try_acquire("lock:test"))
        self.assertEqual(result, LockResult.DEGRADED)

    def test_release_returns_false(self):
        lock = DistributedLock(None)
        ok = run(lock.release("lock:test"))
        self.assertFalse(ok)

    def test_wait_returns_false_immediately(self):
        lock = DistributedLock(None)
        result = run(lock.wait_for_peer("lock:test", AsyncMock(return_value=False)))
        self.assertFalse(result)


class TestDistributedLockMockedRedis(unittest.TestCase):

    def _redis(self, *, set_ok=True):
        m = AsyncMock()
        m.set = AsyncMock(return_value=set_ok)
        m.eval = AsyncMock(return_value=1)
        m.exists = AsyncMock(return_value=0)
        return m

    def test_acquire_success(self):
        lock = DistributedLock(self._redis(set_ok=True))
        self.assertEqual(run(lock.try_acquire("lock:a")), LockResult.ACQUIRED)

    def test_acquire_contended(self):
        lock = DistributedLock(self._redis(set_ok=None))
        self.assertEqual(run(lock.try_acquire("lock:a")), LockResult.WAIT)

    def test_release_invokes_lua(self):
        r = self._redis()
        lock = DistributedLock(r)
        run(lock.try_acquire("lock:x"))
        run(lock.release("lock:x"))
        r.eval.assert_awaited_once()

    def test_wait_returns_true_when_exists_fn_returns_true(self):
        r = self._redis()
        r.exists = AsyncMock(return_value=0)
        lock = DistributedLock(r, wait_timeout_s=5.0, poll_base_ms=5.0)
        calls = []

        async def _exists():
            calls.append(1)
            return len(calls) >= 2

        result = run(lock.wait_for_peer("lock:b", _exists))
        self.assertTrue(result)

    def test_acquire_redis_error_returns_degraded(self):
        r = AsyncMock()
        r.set = AsyncMock(side_effect=ConnectionError("down"))
        lock = DistributedLock(r)
        self.assertEqual(run(lock.try_acquire("lock:c")), LockResult.DEGRADED)


# ===========================================================================
# cache_manager.py
# ===========================================================================

class TestCacheManager(unittest.TestCase):

    def _make_manager(self, tmpdir, *, legacy_dirs=None):
        storage = LocalStorageBackend(Path(tmpdir) / "sharded")
        l1 = LocalCache(max_bytes=100 * 1024 ** 2)
        return CacheManager(
            storage,
            lock=None,
            l1_cache=l1,
            legacy_dirs=legacy_dirs,
            redis_client=None,
            model_version="v_test",
        )

    def test_miss_on_empty_cache(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = self._make_manager(td)
            key = mgr.make_key("Hello", voice_id="simran")
            result = run(mgr.get(key))
            self.assertFalse(result.is_hit)
            self.assertEqual(result.source, "miss")

    def test_store_then_l1_hit(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = self._make_manager(td)
            key = mgr.make_key("Hello", voice_id="simran")
            mgr._storage.write(key.digest, DUMMY_WAV)
            mgr._l1.put(key.digest, len(DUMMY_WAV))
            result = run(mgr.get(key))
            self.assertTrue(result.is_hit)
            self.assertEqual(result.source, "l1_hit")
            self.assertEqual(result.data, DUMMY_WAV)

    def test_store_then_l2_hit(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = self._make_manager(td)
            key = mgr.make_key("Hello world", voice_id="tara")
            mgr._storage.write(key.digest, DUMMY_WAV)
            result = run(mgr.get(key))
            self.assertTrue(result.is_hit)
            self.assertEqual(result.source, "l2_hit")
            self.assertEqual(result.data, DUMMY_WAV)

    def test_l2_hit_populates_l1(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = self._make_manager(td)
            key = mgr.make_key("L1 warmup text", voice_id="vikram")
            mgr._storage.write(key.digest, DUMMY_WAV)
            run(mgr.get(key))
            self.assertTrue(mgr._l1.get(key.digest))

    def test_legacy_dir_hit(self):
        with tempfile.TemporaryDirectory() as td:
            voice_dir = Path(td) / "cached_data_simran"
            voice_dir.mkdir()
            text = "नमस्ते"
            leg_hash = legacy_cache_key(_normalise_text(text))
            (voice_dir / f"{leg_hash}.wav").write_bytes(DUMMY_WAV)

            mgr = self._make_manager(td, legacy_dirs=[("simran", voice_dir)])
            key = mgr.make_key(text, voice_id="simran")
            result = run(mgr.get(key))
            self.assertTrue(result.is_hit)
            self.assertEqual(result.source, "legacy_hit")

    def test_store_async_writes_to_storage(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = self._make_manager(td)
            key = mgr.make_key("Async write test")
            mgr.store_async(key, DUMMY_WAV)
            time.sleep(0.3)
            self.assertTrue(mgr._storage.exists(key.digest))

    def test_stats_structure(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = self._make_manager(td)
            s = mgr.stats()
            for field in ("l1_hits", "l2_hits", "legacy_hits", "peer_hits",
                          "misses", "hit_rate"):
                self.assertIn(field, s)

    def test_different_model_version_different_key(self):
        with tempfile.TemporaryDirectory() as td:
            mgr1 = CacheManager(
                LocalStorageBackend(Path(td) / "s1"),
                model_version="v1",
            )
            mgr2 = CacheManager(
                LocalStorageBackend(Path(td) / "s2"),
                model_version="v2",
            )
            k1 = mgr1.make_key("Hello")
            k2 = mgr2.make_key("Hello")
            self.assertNotEqual(k1.digest, k2.digest)

    def test_stampede_protection_peer_hit(self):
        """WAIT → peer writes file → wait_for_peer returns True → peer_hit reported."""
        with tempfile.TemporaryDirectory() as td:
            storage = LocalStorageBackend(Path(td) / "sharded_sp")
            l1 = LocalCache()

            mock_lock = AsyncMock(spec=DistributedLock)
            mock_lock.try_acquire = AsyncMock(return_value=LockResult.WAIT)

            async def _fake_wait(lock_key, exists_fn):
                # Simulate peer writing the file and returning True.
                # Do NOT write ahead of time — write here so L2 check sees nothing
                # until after the lock path is entered.
                storage.write(lock_key.replace("lock:", ""), DUMMY_WAV)
                return True

            mock_lock.wait_for_peer = _fake_wait

            mgr = CacheManager(storage, lock=mock_lock, l1_cache=l1, model_version="v1")
            key = mgr.make_key("Stampede text", voice_id="simran")
            # Do NOT pre-write — we want the L2 check to miss so the lock path is reached.

            result = run(mgr.get(key))
            self.assertTrue(result.is_hit)
            self.assertEqual(result.source, "peer_hit")
            self.assertEqual(mgr.stat_duplicate_prevented, 1)


# ===========================================================================
# GCSStorageBackend — mock-based tests (no real GCS credentials required)
# ===========================================================================

class _MockBlob:
    """Minimal mock of google.cloud.storage.Blob."""

    def __init__(self, name: str, store: dict, NotFound):
        self._name = name
        self._store = store
        self._NotFound = NotFound

    def exists(self, timeout=None) -> bool:
        return self._name in self._store

    def download_as_bytes(self, timeout=None) -> bytes:
        if self._name not in self._store:
            raise self._NotFound("not found")
        return self._store[self._name]

    def upload_from_string(self, data, content_type=None, timeout=None):
        self._store[self._name] = data

    def delete(self, timeout=None):
        if self._name not in self._store:
            raise self._NotFound("not found")
        del self._store[self._name]


class _MockBucket:
    def __init__(self, name, store, NotFound):
        self.name = name
        self._store = store
        self._NotFound = NotFound

    def blob(self, key):
        return _MockBlob(key, self._store, self._NotFound)


class _MockNotFound(Exception):
    pass


def _make_gcs_backend(bucket_name="test-bucket", prefix="tts-cache"):
    """Build a GCSStorageBackend wired to an in-memory dict instead of real GCS."""
    from flowtts.cache.storage import GCSStorageBackend
    store = {}
    backend = GCSStorageBackend.__new__(GCSStorageBackend)
    backend._bucket = _MockBucket(bucket_name, store, _MockNotFound)
    backend._prefix = prefix
    backend._content_type = "audio/wav"
    backend._timeout = 10.0
    backend._NotFound = _MockNotFound
    backend._store = store  # for test inspection
    return backend


class TestGCSStorageBackend(unittest.TestCase):

    def test_write_then_exists(self):
        b = _make_gcs_backend()
        digest = "a" * 64
        self.assertFalse(b.exists(digest))
        b.write(digest, DUMMY_WAV)
        self.assertTrue(b.exists(digest))

    def test_write_then_read(self):
        b = _make_gcs_backend()
        digest = "b" * 64
        b.write(digest, DUMMY_WAV)
        self.assertEqual(b.read(digest), DUMMY_WAV)

    def test_object_key_sharding(self):
        b = _make_gcs_backend(prefix="tts-cache")
        digest = "abcdef" + "0" * 58
        b.write(digest, DUMMY_WAV)
        expected_key = f"tts-cache/ab/cd/{digest}.wav"
        self.assertIn(expected_key, b._store)

    def test_read_missing_returns_none(self):
        b = _make_gcs_backend()
        self.assertIsNone(b.read("f" * 64))

    def test_delete(self):
        b = _make_gcs_backend()
        digest = "d" * 64
        b.write(digest, DUMMY_WAV)
        b.delete(digest)
        self.assertFalse(b.exists(digest))

    def test_delete_missing_is_noop(self):
        b = _make_gcs_backend()
        b.delete("e" * 64)  # must not raise

    def test_shard_path_format(self):
        b = _make_gcs_backend(bucket_name="kapturecx-ml", prefix="tts-cache")
        digest = "1234" + "0" * 60
        path = b.shard_path(digest)
        self.assertTrue(path.startswith("gs://kapturecx-ml/tts-cache/"))
        self.assertIn(digest, path)


# ===========================================================================
# Two-tier L1+GCS integration test
# ===========================================================================

class TestTwoTierCacheManager(unittest.TestCase):
    """Verify the L1 (local) + L2 (GCS) two-tier architecture."""

    def _make_manager(self, tmpdir):
        from flowtts.cache.storage import LocalStorageBackend
        l1_storage = LocalStorageBackend(Path(tmpdir) / "l1")
        l2_storage = _make_gcs_backend()
        l1 = LocalCache(max_bytes=100 * 1024 ** 2)
        return CacheManager(
            l1_storage,
            l2_storage=l2_storage,
            lock=None,
            l1_cache=l1,
            redis_client=None,
            model_version="v_test",
        ), l2_storage

    def test_gcs_hit_returns_l2_hit(self):
        with tempfile.TemporaryDirectory() as td:
            mgr, l2 = self._make_manager(td)
            key = mgr.make_key("GCS text", voice_id="simran")
            # Pre-populate GCS only.
            l2.write(key.digest, DUMMY_WAV)
            result = run(mgr.get(key))
            self.assertTrue(result.is_hit)
            self.assertEqual(result.source, "l2_hit")
            self.assertTrue(result.path.startswith("gs://"))

    def test_gcs_hit_writes_back_to_l1_disk(self):
        with tempfile.TemporaryDirectory() as td:
            mgr, l2 = self._make_manager(td)
            key = mgr.make_key("L1 writeback text", voice_id="tara")
            l2.write(key.digest, DUMMY_WAV)
            run(mgr.get(key))
            # Give write-back executor a moment to finish.
            time.sleep(0.2)
            self.assertTrue(mgr._storage.exists(key.digest))

    def test_gcs_hit_populates_l1_index(self):
        with tempfile.TemporaryDirectory() as td:
            mgr, l2 = self._make_manager(td)
            key = mgr.make_key("L1 index text", voice_id="vikram")
            l2.write(key.digest, DUMMY_WAV)
            run(mgr.get(key))
            self.assertTrue(mgr._l1.get(key.digest))

    def test_store_async_writes_both_tiers(self):
        with tempfile.TemporaryDirectory() as td:
            mgr, l2 = self._make_manager(td)
            key = mgr.make_key("Both-tier write", voice_id="simran")
            mgr.store_async(key, DUMMY_WAV)
            time.sleep(0.3)
            self.assertTrue(mgr._storage.exists(key.digest))   # L1
            self.assertTrue(l2.exists(key.digest))              # L2 (GCS)

    def test_l1_hit_takes_precedence_over_gcs(self):
        """L1 hit must be returned without touching GCS."""
        with tempfile.TemporaryDirectory() as td:
            mgr, l2 = self._make_manager(td)
            key = mgr.make_key("L1 priority text")
            mgr._storage.write(key.digest, DUMMY_WAV)
            mgr._l1.put(key.digest, len(DUMMY_WAV))
            # GCS has different data — if L2 is used, the data would differ.
            l2.write(key.digest, b"wrong data")
            result = run(mgr.get(key))
            self.assertEqual(result.source, "l1_hit")
            self.assertEqual(result.data, DUMMY_WAV)

    def test_gcs_path_format_in_result(self):
        with tempfile.TemporaryDirectory() as td:
            mgr, l2 = self._make_manager(td)
            key = mgr.make_key("Path format", voice_id="daya")
            l2.write(key.digest, DUMMY_WAV)
            result = run(mgr.get(key))
            self.assertTrue(result.path.startswith("gs://"))


if __name__ == "__main__":
    unittest.main()
