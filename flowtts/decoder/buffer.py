"""Pipeline position: BUFFER — between Redis Pub/Sub and the decoder.

Role in pipeline:
  The worker currently sends all tokens for one utterance in a single
  Redis message (is_final=True immediately). The buffer exists so that if
  the worker is later changed to emit incremental chunks, the gateway
  (api/websockets.py _listen_for_results) can accumulate them per text_id
  and only trigger decoding once the full sequence arrives.

Flow:
  Redis message arrives  {audio_tokens, text_id, is_final}
    → TokenBufferManager.add_chunk(text_id, audio_tokens, is_final)
        if is_final=False → returns None  (keep buffering)
        if is_final=True  → returns full concatenated token string
    → AudioDecoder.decode_to_wav(full_tokens)  [in decoder/decoder.py]

One TokenBufferManager instance is created per call_id in the gateway and
discarded on disconnect (ConnectionManager.token_buffers dict).
"""

from __future__ import annotations

from typing import Dict, List, Optional


class TokenBufferManager:
    """Buffers audio-token chunks per text_id and returns full token sequence when complete.

    Used by the gateway's _listen_for_results: worker publishes token chunks to
    audio:{call_id}; we accumulate by text_id and only decode when we have the
    full sequence (is_final=True).
    """

    def __init__(self) -> None:
        # text_id -> list of token chunk strings (in order)
        self._chunks: Dict[str, List[str]] = {}

    def add_chunk(
        self,
        text_id: str,
        audio_tokens: str,
        is_final: bool,
    ) -> Optional[str]:
        """Add a token chunk for the given text_id.

        Returns the full concatenated token string when is_final is True
        (so the caller can decode once). Otherwise returns None.
        """
        if text_id not in self._chunks:
            self._chunks[text_id] = []
        self._chunks[text_id].append(audio_tokens)

        if not is_final:
            return None

        full_tokens = "".join(self._chunks.pop(text_id, []))
        return full_tokens if full_tokens else None
