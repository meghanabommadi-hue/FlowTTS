"""Pipeline position: VOICE SINGLETON — process-wide VoiceStore accessor.

Mirrors the pattern in `flowtts/synthesis/engine.py`: the API layer and the
engine need the same store instance, and neither should construct it.
"""

from __future__ import annotations

import threading

from flowtts.dhvaani.types import EngineNotReady
from flowtts.dhvaani.voices.store import VoiceStore

_store: VoiceStore | None = None
_lock = threading.Lock()


def set_voice_store(store: VoiceStore) -> None:
    global _store
    with _lock:
        _store = store


def get_voice_store(loaded=None, settings=None) -> VoiceStore:
    """Return the process store, constructing it on first use when `loaded` is given."""
    global _store
    if _store is not None:
        return _store
    with _lock:
        if _store is None:
            if loaded is None:
                raise EngineNotReady(
                    "voice store not initialised -- the engine must be started first"
                )
            _store = VoiceStore(loaded, settings)
        return _store


def reset_voice_store() -> None:
    global _store
    with _lock:
        _store = None
