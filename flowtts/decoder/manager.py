"""Pipeline position: DECODER LIFECYCLE — per-call resource tracking.

Role in pipeline:
  Tracks which call_ids currently have an active decoder session.
  Called by the gateway (api/websockets.py) on connect/disconnect:
    connect    → DecoderManager.acquire(call_id)
    disconnect → DecoderManager.release(call_id)

Current state:
  The gateway uses a single shared AudioDecoder instance (decoder/decoder.py)
  for all calls — acquire/release are lightweight bookkeeping only.

Future use:
  When multiple GPU decoders are needed (e.g. one per active call), this
  manager would assign a gpu_id from a pool and return a dedicated
  AudioDecoder instance per call_id, releasing it back on disconnect.
  The codec_server.py in test/ shows how that multi-instance pattern works.
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
