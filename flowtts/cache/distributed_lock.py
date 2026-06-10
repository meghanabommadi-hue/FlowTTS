"""Distributed lock for cache stampede prevention.

Problem
-------
When 100 servers all receive the same uncached text simultaneously, without
coordination every server would start a GPU synthesis job for the same audio.
This wastes 99× the GPU resources and generates 99 identical files.

Solution: Redis SET NX EX distributed lock
------------------------------------------
Only the first server that wins the lock generates the audio.
All other servers wait, then re-check the cache and serve the result.

Lock protocol
-------------
  ACQUIRE:  SET lock:<digest> <token> NX EX <ttl_s>
              NX  → only set if key does not exist (atomic)
              EX  → auto-expire so crashed servers never deadlock
              token → random UUID owned by this server instance

  RELEASE:  Lua script: if GET lock:<key> == token → DEL lock:<key>
              Ensures only the lock owner can release (prevents split-brain
              where server A's expired lock is accidentally deleted by server B)

  WAIT:     Exponential back-off with jitter polls for lock release OR for
            the cache entry to appear (whichever comes first).
            Jitter (± 10 ms) prevents all waiters waking at the same instant
            (thundering herd after lock release).

Recovery
--------
- Lock TTL auto-expire: if the generating server crashes, the lock expires
  after `lock_ttl_s` seconds and the next waiter becomes the generator.
- Lock TTL extension: long synthesis tasks call `extend()` before expiry so
  the lock stays alive without requiring a full re-acquire.
- The lock token (random UUID per acquire) prevents a restarted server from
  accidentally releasing another server's lock.

Graceful degradation
--------------------
If Redis is unavailable, DistributedLock degrades silently:
  - acquire() returns LockResult.ACQUIRED (optimistic — this server generates)
  - release() is a no-op
  - wait_for_peer() immediately returns False (caller falls through to generation)
  This means the system behaves exactly like the current code when Redis is down,
  with no extra latency or errors surfaced to callers.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Lua script: atomically release lock only if we own it.
_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

# Lua script: atomically extend TTL only if we still own the lock.
_EXTEND_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("expire", KEYS[1], ARGV[2])
else
    return 0
end
"""


class LockResult(Enum):
    ACQUIRED = auto()    # this server owns the lock — proceed with generation
    WAIT     = auto()    # another server holds the lock — wait for it
    DEGRADED = auto()    # Redis unavailable — optimistic pass-through


class DistributedLock:
    """Async Redis distributed lock with stampede prevention.

    Parameters
    ----------
    redis_client  : an ``async redis.asyncio.Redis`` client (or compatible).
                    Pass None to run in degraded no-op mode.
    lock_ttl_s    : lock auto-expiry in seconds.  Should be > worst-case
                    synthesis time.  Default 60 s.
    wait_timeout_s: max seconds a waiter will poll before giving up and
                    generating its own copy.  Default 30 s.
    poll_base_ms  : base poll interval in ms (doubled each retry up to cap).
    poll_cap_ms   : max poll interval in ms.
    poll_jitter_ms: ± random jitter added to each poll interval.
    """

    def __init__(
        self,
        redis_client=None,
        *,
        lock_ttl_s: float = 60.0,
        wait_timeout_s: float = 30.0,
        poll_base_ms: float = 50.0,
        poll_cap_ms: float = 500.0,
        poll_jitter_ms: float = 10.0,
    ) -> None:
        self._redis = redis_client
        self._lock_ttl_s = int(lock_ttl_s)
        self._wait_timeout_s = wait_timeout_s
        self._poll_base_ms = poll_base_ms
        self._poll_cap_ms = poll_cap_ms
        self._poll_jitter_ms = poll_jitter_ms

        # Map: lock_key → token (only for locks held by this process)
        self._owned: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def try_acquire(self, lock_key: str) -> LockResult:
        """Attempt a single non-blocking acquisition of *lock_key*.

        Returns
        -------
        ACQUIRED  if this server now owns the lock.
        WAIT      if another server holds the lock.
        DEGRADED  if Redis is unavailable.
        """
        if self._redis is None:
            return LockResult.DEGRADED

        token = str(uuid.uuid4())
        try:
            result = await self._redis.set(
                lock_key,
                token,
                nx=True,          # only set if not exists
                ex=self._lock_ttl_s,
            )
        except Exception as exc:
            logger.warning("distributed_lock.acquire_error key=%s: %s", lock_key, exc)
            return LockResult.DEGRADED

        if result:
            self._owned[lock_key] = token
            return LockResult.ACQUIRED
        return LockResult.WAIT

    async def release(self, lock_key: str) -> bool:
        """Release a previously acquired lock.

        Uses a Lua script so only the owner can release.  Returns True if
        the lock was successfully released, False if it had already expired
        or was never owned.
        """
        token = self._owned.pop(lock_key, None)
        if token is None or self._redis is None:
            return False
        try:
            result = await self._redis.eval(_RELEASE_SCRIPT, 1, lock_key, token)
            return bool(result)
        except Exception as exc:
            logger.warning("distributed_lock.release_error key=%s: %s", lock_key, exc)
            return False

    async def extend(self, lock_key: str) -> bool:
        """Extend the TTL of a held lock by *lock_ttl_s* more seconds.

        Call this periodically inside a long synthesis loop to prevent the
        lock from expiring while generation is still in progress.
        """
        token = self._owned.get(lock_key)
        if token is None or self._redis is None:
            return False
        try:
            result = await self._redis.eval(
                _EXTEND_SCRIPT, 1, lock_key, token, str(self._lock_ttl_s)
            )
            return bool(result)
        except Exception as exc:
            logger.warning("distributed_lock.extend_error key=%s: %s", lock_key, exc)
            return False

    async def wait_for_peer(
        self,
        lock_key: str,
        exists_fn,
    ) -> bool:
        """Block until either the lock is gone or *exists_fn()* returns True.

        This is the stampede-prevention wait loop.  A waiter calls this after
        seeing WAIT from try_acquire().  It polls until:
          (a) The lock disappears  → the peer finished; waiter should re-check cache.
          (b) The cache entry appears (exists_fn() → True) → serve directly.
          (c) Timeout → give up waiting and generate independently.

        Parameters
        ----------
        lock_key  : the Redis lock key to watch.
        exists_fn : async callable() → bool  — checks whether the cache entry
                    now exists on shared storage.

        Returns
        -------
        True   if the cached entry is now available (caller should serve it).
        False  if the lock expired or timed out (caller should generate).
        """
        deadline = time.monotonic() + self._wait_timeout_s
        delay_ms = self._poll_base_ms

        while time.monotonic() < deadline:
            # Check cache first — peer may have finished quickly.
            try:
                if await exists_fn():
                    return True
            except Exception:
                pass

            # Check whether the lock still exists.
            try:
                if self._redis is not None:
                    still_locked = await self._redis.exists(lock_key)
                    if not still_locked:
                        # Lock gone — peer finished or crashed.  Re-check cache.
                        try:
                            return bool(await exists_fn())
                        except Exception:
                            return False
            except Exception as exc:
                logger.debug("distributed_lock.wait_poll_error key=%s: %s", lock_key, exc)
                return False  # degraded → caller generates

            # Exponential back-off with bounded jitter.
            jitter = random.uniform(-self._poll_jitter_ms, self._poll_jitter_ms)
            sleep_s = min(delay_ms + jitter, self._poll_cap_ms) / 1000.0
            await asyncio.sleep(max(0.001, sleep_s))
            delay_ms = min(delay_ms * 2, self._poll_cap_ms)

        logger.warning("distributed_lock.wait_timeout key=%s", lock_key)
        return False  # timeout → caller generates
