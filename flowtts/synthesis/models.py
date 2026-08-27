"""Pipeline position: SYNTHESIS — text → waveform (public wrapper).

Role in pipeline:
  The stable facade every transport uses — WebSocket, REST, OpenAI-compatible,
  and the Redis worker. It owns the full request path around the GPU:

      normalize (flowtts.text)
        → chunk    (flowtts.synthesis.chunker)
          → generate, all chunks dispatched at once (OmniVoiceEngine)
            → stitch (flowtts.processing.stitch)
              → waveform / stream of waveforms

  Two entry points:

      synthesize(text, …)         → one np.ndarray (whole utterance)
      synthesize_stream(text, …)  → async generator of StreamChunk

Streaming model (OmniVoice is non-autoregressive → no token-by-token emission):
  Every chunk is dispatched to the engine's batch queue immediately, so chunks
  coalesce with each other and with other requests' chunks; results are yielded
  IN ORDER, so the client plays a continuous stream while later chunks are still
  on the GPU. Time-to-first-byte is therefore the cost of chunk 0 alone, not of
  the utterance.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import structlog

from flowtts.core.config import settings
from flowtts.processing.stitch import StreamStitcher, stitch_all
from flowtts.synthesis.chunker import CLAUSE, SENTENCE, Chunk, split_for_streaming
from flowtts.synthesis.omnivoice_engine import GenParams, OmniVoiceEngine
from flowtts.text import (
    NormalizerConfig,
    detect_language,
    light_sanitize,
    normalize_for_tts,
)

logger = structlog.get_logger(__name__)


@dataclass
class StreamChunk:
    """One stitched, ready-to-encode piece of the output stream."""

    index: int
    audio: np.ndarray        # float32 at engine.sampling_rate
    is_final: bool
    text: str                # the text chunk this audio came from
    boundary: str = "end"    # what caused the split after it (see chunker)


# The normalizer binds the words of a single numeral with a non-breaking space
# so the chunker cannot split "दो हज़ार पाँच सौ" in half. That is a chunking
# hint, not something the model should ever see: OmniVoice's tokenizer has no
# reason to treat U+00A0 the way it treats a space.
_NBSP = "\u00a0"


def for_model(text: str) -> str:
    """Undo the chunker's non-breaking hints just before the GPU sees the text."""
    return text.replace(_NBSP, " ")


def _same_script(a: str, b: str) -> bool:
    """True if two language codes are written in the same script.

    Languages this server has no profile for (Yoruba, Igbo, Nigerian Pidgin —
    all Latin) fall through to the English profile, which is also Latin, so they
    compare equal to detected Latin text and keep their voice's language.
    """
    from flowtts.text import get_profile

    return (get_profile(a).script or "latn") == (get_profile(b).script or "latn")


def _normalizer_config(overrides: dict | None = None) -> NormalizerConfig:
    """Merge the configured normalizer settings with per-request overrides."""
    data = settings.text.model_dump()
    data.pop("default_language", None)
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return NormalizerConfig.from_dict(data)


class OmniVoiceSynthesizer:
    """Loads OmniVoice once; synthesizes text → waveform, whole or streamed."""

    def __init__(self) -> None:
        self.engine = OmniVoiceEngine()

    async def initialize(self) -> None:
        await self.engine.initialize()

    # ------------------------------------------------------------------ metadata
    @property
    def sampling_rate(self) -> int:
        return self.engine.sampling_rate

    @property
    def engine_info(self) -> dict:
        return self.engine.engine_info

    @property
    def registry(self):
        return self.engine.registry

    # ------------------------------------------------------------------ prep
    def resolve_language(self, text: str, language: str | None,
                         voice_id: str | None) -> str | None:
        """Decide the language once, for both normalization and inference.

        Precedence, highest first:

            1. what the caller asked for — pass it, on every request
            2. the voice's own preferred language, stored when it was cloned
            3. the script of the text, only if ``text.detect_language`` is on
               (it is off by default)
            4. the configured default
            5. None — OmniVoice runs language-agnostic

        There is deliberately no detection in the default path. Script detection
        cannot identify a language, only a script, and language changes the
        model's output completely, so a confident wrong guess is worse than
        none.

        Step 2 is not detection — it is metadata recorded when the voice was
        cloned, and it applies only when the caller omitted the parameter. When
        detection is enabled it is additionally required to agree with the
        text's script, so an Assamese voice handed Tamil text still reads Tamil.
        """
        if language:
            return language

        detected = (detect_language(text, default="")
                    if settings.text.detect_language else "")
        preferred = self.registry.language(voice_id) if self.registry else None

        if preferred and (not detected or _same_script(preferred, detected)):
            return preferred

        resolved = detected or settings.text.default_language
        if not resolved:
            # Worth surfacing rather than silently running agnostic: language
            # conditions the model's phonemes, so a caller omitting it is the
            # most likely source of "the voice sounds unstable". The counter
            # shows up in /v1/stats so it can be tracked down.
            self.engine.stats["no_language"] = self.engine.stats.get("no_language", 0) + 1
            logger.warning("request_without_language", voice_id=voice_id,
                           chars=len(text), preview=text[:48])
        return resolved

    def prepare(
        self,
        text: str,
        language: str | None,
        *,
        voice_id: str | None = None,
        normalize: bool | None = None,
        normalizer_overrides: dict | None = None,
    ) -> tuple[str, str | None]:
        """Normalize *text* and resolve the language to synthesize it in."""
        resolved = self.resolve_language(text, language, voice_id)
        if normalize is False:
            return light_sanitize(text), resolved

        cfg = _normalizer_config(normalizer_overrides)
        if normalize is True:
            cfg.enabled = True

        # The normalizer still needs *a* language even when inference will run
        # agnostic, because it has to pick the words a numeral is spelled with —
        # "२,५००" must not come out as English inside a Hindi sentence. That
        # choice never reaches the model: it only decides which characters are
        # written, so inferring it from the script here is safe in a way that
        # inferring the inference language is not.
        clean, _ = normalize_for_tts(text, resolved or detect_language(text), cfg)
        return clean, resolved

    def chunk(self, text: str):
        """Split normalized text into Chunks with the configured budgets."""
        st = settings.streaming
        return split_for_streaming(
            text,
            target_chars=st.target_chars,
            tolerance_chars=st.tolerance_chars,
            first_chunk_chars=st.first_chunk_chars,
            split_on_clause=st.split_on_clause,
            max_chunks=st.max_chunks,
        )

    def _stitch_kwargs(self) -> dict:
        st = settings.streaming
        return {
            "overlap_ms": st.crossfade_ms,
            "edge_fade_ms": st.edge_fade_ms,
            "click_fade_ms": st.click_fade_ms,
            "final_fade_ms": st.final_fade_ms,
            "trim": st.trim_silence,
            "trim_keep_ms": st.trim_keep_ms,
            "level_match": st.level_match,
            "max_gain_db": st.level_match_max_db,
            "gaps_ms": {SENTENCE: st.sentence_gap_ms, CLAUSE: st.clause_gap_ms},
        }

    def _stitcher(self) -> StreamStitcher:
        return StreamStitcher(self.engine.sampling_rate, **self._stitch_kwargs())

    # ------------------------------------------------------------------ synthesize
    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        language: str | None = None,
        instruct: str | None = None,
        prompt=None,
        params: GenParams | None = None,
        normalize: bool | None = None,
        normalizer_overrides: dict | None = None,
        chunked: bool | None = None,
    ) -> np.ndarray:
        """Return the whole waveform for *text* (float32, engine.sampling_rate).

        Long input is still chunked and stitched — one ``generate()`` on a very
        long text is both slower (OmniVoice chunks it internally anyway, serially)
        and more likely to drift in prosody than N batched chunks stitched here.
        """
        clean, lang = self.prepare(text, language, voice_id=voice_id,
                                   normalize=normalize,
                                   normalizer_overrides=normalizer_overrides)
        if not clean:
            return np.zeros(0, dtype=np.float32)

        params = params or GenParams.build()
        # chunked=False means "one generate() for the whole text" — still a
        # Chunk, so the code below has one shape to handle rather than two.
        chunks = self.chunk(clean) if chunked is not False else [Chunk(text=clean)]
        if len(chunks) <= 1:
            return await self.engine.synthesize(
                for_model(chunks[0].text if chunks else clean), voice_id=voice_id,
                language=lang, instruct=instruct, prompt=prompt, params=params,
            )

        stream_params = params.for_streaming()
        waves = await asyncio.gather(*[
            self.engine.synthesize(for_model(chunk.text), voice_id=voice_id,
                                   language=lang, instruct=instruct, prompt=prompt,
                                   params=stream_params)
            for chunk in chunks
        ])
        return stitch_all(
            list(waves), self.engine.sampling_rate,
            boundaries=[c.boundary for c in chunks], **self._stitch_kwargs(),
        )

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        language: str | None = None,
        instruct: str | None = None,
        prompt=None,
        params: GenParams | None = None,
        normalize: bool | None = None,
        normalizer_overrides: dict | None = None,
        cancel_event: asyncio.Event | None = None,
    ):
        """Yield :class:`StreamChunk` objects in order as each chunk completes."""
        clean, lang = self.prepare(text, language, voice_id=voice_id,
                                   normalize=normalize,
                                   normalizer_overrides=normalizer_overrides)
        if not clean:
            return

        chunks = self.chunk(clean)
        if not chunks:
            return

        params = (params or GenParams.build()).for_streaming()

        # Dispatch everything now so the chunks batch together on the GPU; the
        # consumer still receives them strictly in order.
        tasks = [
            asyncio.create_task(self.engine.synthesize(
                for_model(chunk.text), voice_id=voice_id, language=lang,
                instruct=instruct, prompt=prompt, params=params,
            ))
            for chunk in chunks
        ]

        stitcher = self._stitcher()
        emitted = 0
        cancelled = False
        try:
            for i, task in enumerate(tasks):
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                wav = await task
                is_final = i == len(tasks) - 1
                audio = stitcher.push(wav, boundary=chunks[i].boundary,
                                      is_final=is_final)
                if audio.size or is_final:
                    yield StreamChunk(index=emitted, audio=audio, is_final=is_final,
                                      text=for_model(chunks[i].text),
                                      boundary=chunks[i].boundary)
                    emitted += 1

            if cancelled:
                # Flush the held overlap so a cancelled stream ends on a fade
                # rather than mid-sample, which clicks in the client's player.
                tail = stitcher.flush()
                if tail.size:
                    yield StreamChunk(index=emitted, audio=tail, is_final=True, text="")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
