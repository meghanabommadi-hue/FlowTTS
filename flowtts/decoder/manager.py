"""Per-call decoder instance lifecycle and GPU assignment.

Decoder instances subscribe to audio:{call_id}, decode tokens to PCM
(via decoder.py), run processing pipeline, and are torn down when the
WebSocket for that call_id disconnects. This module can assign decoder
GPU ids when running multiple decoder workers.
"""

from __future__ import annotations

from typing import Dict, Optional


class DecoderManager:
    """Manages per-call_id decoder instances and optional GPU assignment."""

    def __init__(self) -> None:
        self._instances: Dict[str, object] = {}  # call_id -> decoder instance if needed

    def acquire(self, call_id: str, gpu_id: Optional[int] = None) -> None:
        """Register or acquire a decoder for this call_id (e.g. assign GPU)."""
        self._instances[call_id] = None  # placeholder; gateway uses shared decoder for now

    def release(self, call_id: str) -> None:
        """Release decoder for this call_id on WebSocket disconnect."""
        self._instances.pop(call_id, None)
