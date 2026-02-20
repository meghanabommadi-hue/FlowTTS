"""Token chunk buffer: assembles LLM output token chunks before decoding.

This module lives under decoder/ because the buffer sits BEFORE the decoder
in the flow: Redis token chunks → buffer (reorder/assemble per text_id)
→ decoder (tokens → PCM). Post-decoder audio processing (resample, crossfade)
lives in flowtts.processing.
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
