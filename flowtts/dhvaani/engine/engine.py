"""Pipeline position: ENGINE FACADE — the one object the API layer talks to.

Role in pipeline:
  Owns every stage and orchestrates a request end to end.

      text
        -> TextNormalizer      (cached, off the event loop)
        -> SmartChunker        (span schedule: short first, then ramping)
        -> tokenizer           (prompt tokens + span tokens, char level)
        -> FlowScheduler       (continuous-batched Euler ODE)   [GPU]
        -> VocodeStage         (micro-batched Vocos + resample) [GPU]
        -> SpanStitcher        (overlap-add across spans)
        -> AudioChunk          (PCM bytes, in span order)

Ordering guarantee
------------------
Up to `engine.chunk_lookahead` spans of one request are in flight at once, so
span 2's ODE overlaps span 1's vocoding. The scheduler may finish them out of
order (span 2 can land in a smaller bucket than span 1 and retire first), but
this class always emits them in `span_index` order. A client never has to
reassemble anything.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np
import structlog

from flowtts.dhvaani.backends import build_backend
from flowtts.dhvaani.backends.torch_backend import TorchFmBackend
from flowtts.dhvaani.config import MODEL_SAMPLE_RATE, dhv_settings
from flowtts.dhvaani.engine.arena import ArenaPool
from flowtts.dhvaani.engine.memory import VramWatchdog
from flowtts.dhvaani.engine.scheduler import FlowScheduler
from flowtts.dhvaani.engine.stitch import SpanStitcher
from flowtts.dhvaani.engine.vocode import VocodeStage
from flowtts.dhvaani.model.loader import load_model
from flowtts.dhvaani.model.text_encoder import TextEncoder
from flowtts.dhvaani.model.vocoder import VocosVocoder
from flowtts.dhvaani.text import lang as langmod
from flowtts.dhvaani.text.chunker import SmartChunker, add_punctuation
from flowtts.dhvaani.text.normalizer import TextNormalizer
from flowtts.dhvaani.types import (
    AudioChunk,
    EngineNotReady,
    RequestCancelled,
    RequestMetrics,
    SpanRequest,
    SynthParams,
    TextTooLong,
    VoicePrompt,
    new_request_id,
)
from flowtts.dhvaani.voices.registry import set_voice_store
from flowtts.dhvaani.voices.store import VoiceStore

logger = structlog.get_logger(__name__)


def pcm_to_bytes(pcm: np.ndarray, encoding: str) -> bytes:
    if encoding == "pcm_float32":
        return np.asarray(pcm, dtype=np.float32).tobytes()
    clipped = np.clip(pcm, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


class DhvaaniEngine:
    """Process-wide TTS engine. Construct once, `start()` once."""

    def __init__(self, settings=None):
        self._s = settings or dhv_settings
        self._ready = False
        self.loaded = None
        self.voices: VoiceStore | None = None
        self.normalizer = TextNormalizer(self._s)
        self.chunker = SmartChunker(self._s)
        self._scheduler: FlowScheduler | None = None
        self._vocode: VocodeStage | None = None
        self._watchdog: VramWatchdog | None = None
        self._arenas: ArenaPool | None = None
        self._backend = None
        self._fallback = None
        self._started_at = 0.0
        self._requests = 0
        self._errors = 0

    # -- lifecycle -----------------------------------------------------------
    async def start(self) -> None:
        if self._ready:
            return
        t0 = time.perf_counter()
        logger.info("dhvaani_engine_starting", backend=self._s.backend.kind)

        self.loaded = load_model(self._s)
        self.text_encoder = TextEncoder(self.loaded, self._s)
        self.vocoder = VocosVocoder(self.loaded, self._s)

        self._arenas = ArenaPool(self._s, self.loaded.device, self.loaded.dtype)
        self._watchdog = VramWatchdog(self._s)

        self._backend = build_backend(self.loaded, self._s)
        # A non-torch backend may not cover every bucket; keep a torch backend
        # around so an uncovered shape degrades in latency rather than failing.
        self._fallback = (
            self._backend
            if isinstance(self._backend, TorchFmBackend)
            else TorchFmBackend(self.loaded, self._s)
        )

        self._scheduler = FlowScheduler(
            self._backend, self.text_encoder, self._arenas, self._watchdog,
            self._s, fallback_backend=self._fallback,
        )
        self._vocode = VocodeStage(self.vocoder, self._s)

        self.voices = VoiceStore(self.loaded, self._s)
        set_voice_store(self.voices)

        await self._scheduler.start()
        await self._vocode.start()
        await self._watchdog.start()

        if self._s.server.warmup_enabled:
            await self._warmup()

        self._ready = True
        self._started_at = time.perf_counter()
        logger.info(
            "dhvaani_engine_ready",
            startup_s=round(time.perf_counter() - t0, 2),
            backend=getattr(self._backend, "name", "?"),
            voices=len(self.voices.list()),
            arenas=self._arenas.stats()["total_mib"],
        )

    async def stop(self) -> None:
        self._ready = False
        for comp in (self._scheduler, self._vocode, self._watchdog):
            if comp is not None:
                try:
                    await comp.stop()
                except Exception:
                    pass
        if self._backend is not None:
            self._backend.close()
        if self._fallback is not None and self._fallback is not self._backend:
            self._fallback.close()
        self.normalizer.close()
        logger.info("dhvaani_engine_stopped")

    @property
    def ready(self) -> bool:
        return self._ready

    async def _warmup(self) -> None:
        """Prime CUDA graphs, cuDNN algorithm choice and TRT contexts.

        Done before any port is bound, because the first call on a cold shape
        can be an order of magnitude slower than the steady state and would
        otherwise show up as a latency spike on real traffic.
        """
        s = self._s
        # Cover the whole range a span can land in, not just the first few
        # buckets: a request that hits an uncaptured shape pays a full graph
        # capture inline, which shows up as a multi-hundred-millisecond spike.
        buckets = [b for b in s.buckets.buckets if b <= 896]
        # Keep warmup batches modest: a warmup forward at batch 64 x 896 frames
        # allocates attention scores of O(batch * frames^2) and leaves that peak
        # reserved in the caching allocator. Large batches are compute-bound and
        # need no priming beyond what cuDNN already picked at batch 32.
        batches = sorted({1, 4, 16, min(32, s.engine.max_batch_size)})
        t0 = time.perf_counter()
        try:
            self._backend.warmup(buckets, batches)
            if self._fallback is not self._backend:
                self._fallback.warmup(buckets, [batches[0]])
            self.vocoder.warmup(buckets[:2], [1, min(8, s.engine.max_batch_size)])
        except Exception as e:
            logger.warning("warmup_failed", error=str(e)[:300])
        # Warmup's peak activations are far larger than steady state; hand the
        # reserved-but-unused blocks back before traffic arrives.
        if self._watchdog is not None:
            self._watchdog.maybe_collect(force=True)
        logger.info(
            "warmup_done",
            buckets=buckets,
            batches=batches,
            elapsed_s=round(time.perf_counter() - t0, 2),
            vram=self._watchdog.snapshot() if self._watchdog else {},
        )

    # -- request path --------------------------------------------------------
    def _prepare(
        self, text: str, voice_id: str | None, language: str | None, params
    ) -> tuple[str, str, VoicePrompt, SynthParams]:
        if not self._ready:
            raise EngineNotReady("engine is still starting")
        if len(text) > self._s.server.max_text_chars:
            raise TextTooLong(
                f"input is {len(text)} characters; the limit is "
                f"{self._s.server.max_text_chars}"
            )
        voice = self.voices.resolve(voice_id)
        lang = langmod.resolve(language or voice.language, text, self._s.text.default_language)
        p = params or SynthParams.from_settings(self._s)
        return text, lang, voice, p

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str | None = None,
        language: str | None = None,
        params: SynthParams | None = None,
        request_id: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AudioChunk]:
        rid = request_id or new_request_id()
        t_start = time.perf_counter()
        text, lang, voice, p = self._prepare(text, voice_id, language, params)

        metrics = RequestMetrics(
            request_id=rid, voice_id=voice.voice_id, language=lang,
            n_chars=len(text), n_spans=0,
        )
        self._requests += 1

        t0 = time.perf_counter()
        norm = await self.normalizer.normalize_async(text, lang)
        metrics.normalize_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        spans = self.chunker.split(norm, voice, p.speed)
        metrics.n_spans = len(spans)
        if not spans:
            metrics.total_ms = (time.perf_counter() - t_start) * 1000
            yield AudioChunk(rid, 0, b"", p.output_sample_rate,
                             self._s.audio.encoding, True, norm,
                             meta={"metrics": metrics.__dict__})
            return

        prompt_ids = voice.token_ids
        requests: list[SpanRequest] = []
        for sp in spans:
            ids = self.loaded.token_ids(add_punctuation(sp.text))
            requests.append(
                SpanRequest(
                    request_id=rid, span_index=sp.index, n_spans=len(spans),
                    text=sp.text, token_ids=ids, voice=voice, params=p,
                    is_final=sp.is_final,
                )
            )
        metrics.tokenize_ms = (time.perf_counter() - t0) * 1000

        stitcher = SpanStitcher(
            p.output_sample_rate,
            self._s.audio.crossfade_seconds,
            self._s.audio.final_fade_seconds,
            self._s.audio.trim_edge_silence,
            self._s.audio.silence_threshold_db,
        )

        lookahead = max(1, self._s.engine.chunk_lookahead)
        pending: dict[int, asyncio.Future] = {}
        next_submit = 0
        next_emit = 0
        chunk_index = 0
        first_byte_at = 0.0
        failed = False
        emitted_final = False
        bytes_per_sample = 4 if self._s.audio.encoding == "pcm_float32" else 2

        async def _submit_upto(limit: int) -> None:
            nonlocal next_submit
            while next_submit < len(requests) and len(pending) < limit:
                req = requests[next_submit]
                pending[req.span_index] = await self._scheduler.submit(req)
                next_submit += 1

        try:
            await _submit_upto(lookahead)

            while next_emit < len(requests):
                if cancel_event is not None and cancel_event.is_set():
                    self._scheduler.cancel(rid)
                    raise RequestCancelled("cancelled by client")

                fut = pending.pop(next_emit)
                t_flow = time.perf_counter()
                mel = await fut
                span_flow_ms = (time.perf_counter() - t_flow) * 1000
                metrics.flow_ms += span_flow_ms

                # Refill the window as soon as a slot frees, so span N+2 starts
                # its ODE while span N is still vocoding.
                await _submit_upto(lookahead)

                t_voc = time.perf_counter()
                pcm = await self._vocode.submit(
                    mel, int(mel.shape[0]), voice.prompt_rms,
                    self._s.flow.target_rms, p.output_sample_rate,
                )
                metrics.vocode_ms += (time.perf_counter() - t_voc) * 1000

                req = requests[next_emit]
                emit = stitcher.push(pcm, is_final=req.is_final)
                next_emit += 1
                metrics.steps_total += req.params.num_step

                slices = self._slice(emit, req.is_final, p.output_sample_rate)
                for i, (payload, final) in enumerate(slices):
                    if not payload and not final:
                        continue
                    if first_byte_at == 0.0 and payload:
                        first_byte_at = time.perf_counter()
                        metrics.ttfb_ms = (first_byte_at - t_start) * 1000
                    metrics.audio_s += len(payload) / bytes_per_sample / p.output_sample_rate

                    # The LAST chunk of the LAST span carries the request's
                    # metrics. Callers (engine.synthesize, the WS gateway, the
                    # REST headers) read them from there, so they have to be
                    # finalised here rather than in the `finally` below -- by
                    # then there is nothing left to attach them to.
                    is_last = req.is_final and i == len(slices) - 1
                    meta = {"span_index": req.span_index,
                            "flow_ms": round(span_flow_ms, 2)}
                    if is_last:
                        metrics.total_ms = (time.perf_counter() - t_start) * 1000
                        meta["metrics"] = dict(metrics.__dict__)
                        emitted_final = True

                    yield AudioChunk(
                        request_id=rid,
                        chunk_index=chunk_index,
                        audio=payload,
                        sample_rate=p.output_sample_rate,
                        encoding=self._s.audio.encoding,
                        is_final=is_last,
                        text=req.text,
                        meta=meta,
                    )
                    chunk_index += 1
        except RequestCancelled:
            raise
        except Exception as e:
            failed = True
            self._errors += 1
            metrics.error = str(e)
            self._scheduler.cancel(rid)
            logger.error("synthesize_failed", request_id=rid, error=str(e)[:300])
            raise
        finally:
            for f in pending.values():
                f.cancel()
            if self._watchdog is not None:
                self._watchdog.note_request_done()
            if not metrics.total_ms:
                metrics.total_ms = (time.perf_counter() - t_start) * 1000
            if not failed:
                from flowtts.dhvaani.monitoring.metrics import record_request

                record_request(metrics)

        # Safety net: `push(is_final=True)` already drains the stitcher, so this
        # only fires if the stream ended without a final span (an early stop).
        tail = stitcher.flush()
        if tail.size or not emitted_final:
            metrics.total_ms = (time.perf_counter() - t_start) * 1000
            yield AudioChunk(
                request_id=rid, chunk_index=chunk_index,
                audio=pcm_to_bytes(tail, self._s.audio.encoding) if tail.size else b"",
                sample_rate=p.output_sample_rate, encoding=self._s.audio.encoding,
                is_final=True, meta={"metrics": dict(metrics.__dict__)},
            )

    def _slice(self, pcm: np.ndarray, is_final: bool, sr: int) -> list:
        """Split a span's PCM into wire chunks, honouring audio.emit_slice_ms."""
        if pcm.size == 0:
            return [(b"", True)] if is_final else []
        ms = self._s.audio.emit_slice_ms
        enc = self._s.audio.encoding
        if ms <= 0:
            return [(pcm_to_bytes(pcm, enc), True)]
        step = max(1, int(sr * ms / 1000.0))
        out = []
        for i in range(0, pcm.size, step):
            piece = pcm[i : i + step]
            out.append((pcm_to_bytes(piece, enc), i + step >= pcm.size))
        return out

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        language: str | None = None,
        params: SynthParams | None = None,
        request_id: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> tuple[bytes, RequestMetrics]:
        """Non-streaming convenience: the whole utterance as one buffer."""
        parts: list[bytes] = []
        metrics: RequestMetrics | None = None
        async for chunk in self.synthesize_stream(
            text, voice_id, language, params, request_id, cancel_event
        ):
            parts.append(chunk.audio)
            m = chunk.meta.get("metrics")
            if m is not None:
                metrics = RequestMetrics(**m)
        if metrics is None:
            metrics = RequestMetrics(
                request_id=request_id or "", voice_id=voice_id or "",
                language=language or "", n_chars=len(text), n_spans=0,
            )
        return b"".join(parts), metrics

    # -- observability -------------------------------------------------------
    def stats(self) -> dict:
        return {
            "ready": self._ready,
            "uptime_s": round(time.perf_counter() - self._started_at, 1) if self._ready else 0,
            "requests": self._requests,
            "errors": self._errors,
            "backend": getattr(self._backend, "name", None),
            "sample_rate": MODEL_SAMPLE_RATE,
            "scheduler": self._scheduler.stats() if self._scheduler else {},
            "vocode": self._vocode.stats() if self._vocode else {},
            "backend_detail": self._backend.stats() if self._backend else {},
            "vram": self._watchdog.snapshot() if self._watchdog else {},
            "voices": self.voices.stats() if self.voices else {},
            "text_cache": self.normalizer.cache_stats(),
        }
