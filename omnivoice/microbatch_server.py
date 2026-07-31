"""
Async microbatching queue for OmniVoice, aimed at serving >20 concurrent
requests with sub-1s time-to-first-audio.

Why this shape (see the analysis in the accompanying discussion):

  - OmniVoice.generate() is a synchronous, single-call batch API with no
    token-by-token streaming: it runs `num_step` iterative-unmasking
    forward passes over the WHOLE batch, then decodes, then returns. There
    is no partial/early output — "TTFT" for this model means "wall clock
    of one generate() call".
  - Cost is ~O(2B * max_c_len * num_step), not O(B) — padding every item in
    a batch to the longest item's length means one long request taxes every
    short request sharing its batch. So naive "throw all pending requests
    into one big batch" is wrong; requests should be bucketed by estimated
    length before batching.
  - The codec decode step (`audio_tokenizer.decode`) is the one place the
    stock code loops per-sample instead of batching; `batched_decode.py`
    already patches that and is correctness-verified in
    `test_batched_decode.py` (numerically equivalent, faster). This queue
    always applies that patch.

Strategy implemented here: a short debounce window (default 15ms) collects
concurrent incoming requests, buckets them by a cheap length estimate
(word count) into same-size-class groups, caps each group at
`max_batch_size`, and dispatches each group as one `generate()` call. This
keeps batches large enough for GPU efficiency while avoiding the
worst-case padding blowup from mixing very short and very long requests.

To hit <1s under sustained load, tune (in order of impact):
  1. `num_step` in OmniVoiceGenerationConfig — linear in latency. 32 is the
     model default; 8-16 is a reasonable first cut for a hard 1s budget.
  2. `max_batch_size` — bigger batches amortize the fixed per-step forward
     pass cost better, but only up to whatever GPU memory / max_c_len^2
     attention cost allows before compute-bound scaling kicks in. Bench it.
  3. `debounce_seconds` — how long to wait for more requests to join a
     batch before dispatching. Larger = better batching, worse floor
     latency for the unlucky first request in a window.
  4. Reference audio length — cap/trim ref_audio to a few seconds; it
     inflates max_c_len (and therefore compute) for every request batched
     alongside it (see OmniVoice.create_voice_clone_prompt's own 20s
     warning in omnivoice.py).

This module owns queuing/batching only. It does not implement an HTTP/gRPC
front end -- wrap `MicroBatchServer.submit()` in whatever transport you
need (FastAPI endpoint, websocket handler, etc).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from omnivoice import OmniVoice

from batched_decode import enable_batched_decode


@dataclass
class TTSRequest:
    text: str
    language: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    instruct: Optional[str] = None
    speed: Optional[float] = None


@dataclass
class _PendingRequest:
    request: TTSRequest
    submitted_at: float
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


@dataclass
class TTSResult:
    audio: np.ndarray
    sample_rate: int
    queue_wait_s: float
    generate_s: float
    total_s: float
    batch_size: int


class MicroBatchServer:
    """Collects concurrent TTSRequests, batches them by size-bucket, and
    dispatches each bucket through a single (batched-decode-patched)
    OmniVoice.generate() call.

    Usage:
        server = MicroBatchServer(model)
        await server.start()
        result = await server.submit(TTSRequest(text="...", ...))
        ...
        await server.stop()
    """

    def __init__(
        self,
        model: OmniVoice,
        *,
        debounce_seconds: float = 0.015,
        max_batch_size: int = 24,
        num_step: int = 16,
        length_bucket_words: int = 8,
        generation_kwargs: Optional[dict] = None,
    ):
        """
        Args:
            model: A loaded OmniVoice model. Will be patched in-place with
                `enable_batched_decode` if not already patched.
            debounce_seconds: How long to hold a bucket open collecting
                more same-size-class requests before dispatching it.
            max_batch_size: Hard cap on requests per generate() call.
            num_step: Iterative-unmasking steps. Overrides the model's
                default (32) -- lower is faster/lower-quality. Passed
                straight into OmniVoiceGenerationConfig via generate()'s
                **kwargs.
            length_bucket_words: Requests are grouped into buckets by
                `len(text.split()) // length_bucket_words`, so requests of
                similar length get batched together and avoid padding a
                short request out to a long one's `max_c_len`.
            generation_kwargs: Extra kwargs forwarded to `model.generate()`
                (e.g. guidance_scale, audio_chunk_threshold). `num_step` is
                set separately above and merged in.
        """
        self.model = model
        self.debounce_seconds = debounce_seconds
        self.max_batch_size = max_batch_size
        self.length_bucket_words = length_bucket_words
        self.generation_kwargs = {"num_step": num_step, **(generation_kwargs or {})}

        if not getattr(model, "_batched_decode_enabled", False):
            enable_batched_decode(model)
            model._batched_decode_enabled = True

        self._buckets: dict[int, list[_PendingRequest]] = {}
        self._bucket_tasks: dict[int, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        async with self._lock:
            tasks = list(self._bucket_tasks.values())
        for t in tasks:
            t.cancel()

    def _bucket_key(self, request: TTSRequest) -> int:
        return len(request.text.split()) // self.length_bucket_words

    async def submit(self, request: TTSRequest) -> TTSResult:
        """Enqueue a request and await its audio. Safe to call concurrently
        from many asyncio tasks -- each call joins (or opens) the debounce
        window for its length bucket and returns once that bucket's
        generate() call has produced this request's audio.
        """
        if not self._running:
            raise RuntimeError("MicroBatchServer is not started; call start() first.")

        pending = _PendingRequest(request=request, submitted_at=time.perf_counter())
        key = self._bucket_key(request)

        async with self._lock:
            bucket = self._buckets.setdefault(key, [])
            bucket.append(pending)
            is_first_in_bucket = len(bucket) == 1
            if is_first_in_bucket:
                self._bucket_tasks[key] = asyncio.create_task(self._run_bucket(key))
            elif len(bucket) >= self.max_batch_size:
                # Bucket is full -- dispatch immediately instead of waiting
                # out the rest of the debounce window.
                task = self._bucket_tasks.get(key)
                if task is not None:
                    task.cancel()
                self._bucket_tasks[key] = asyncio.create_task(
                    self._dispatch_bucket(key)
                )

        return await pending.future

    async def _run_bucket(self, key: int) -> None:
        try:
            await asyncio.sleep(self.debounce_seconds)
        except asyncio.CancelledError:
            # Cancelled because the bucket filled up and is being
            # dispatched immediately by submit() instead -- nothing to do.
            return
        await self._dispatch_bucket(key)

    async def _dispatch_bucket(self, key: int) -> None:
        async with self._lock:
            batch = self._buckets.pop(key, [])
            self._bucket_tasks.pop(key, None)
        if not batch:
            return

        # A bucket can exceed max_batch_size if many requests land in the
        # same debounce window before the "full" check in submit() fires;
        # split defensively so no single generate() call gets too large.
        for i in range(0, len(batch), self.max_batch_size):
            await self._generate_batch(batch[i : i + self.max_batch_size])

    async def _generate_batch(self, batch: list[_PendingRequest]) -> None:
        n = len(batch)
        texts = [p.request.text for p in batch]
        languages = [p.request.language for p in batch]
        ref_audios = [p.request.ref_audio for p in batch]
        ref_texts = [p.request.ref_text for p in batch]
        # OmniVoice._preprocess_all/_estimate_target_tokens special-cases a
        # bare `None` for speed/instruct (skip entirely) but not a list of
        # all-None (it indexes in and does e.g. `speed > 0`, which throws
        # on None). Only forward these as per-item lists when at least one
        # request actually sets a value; otherwise omit them like the
        # single-request call in inference.py does.
        instructs = [p.request.instruct for p in batch]
        if not any(v is not None for v in instructs):
            instructs = None
        speeds = [p.request.speed for p in batch]
        if not any(v is not None for v in speeds):
            speeds = None

        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        try:
            audios = await loop.run_in_executor(
                None,
                lambda: self._run_generate(
                    texts, languages, ref_audios, ref_texts, instructs, speeds
                ),
            )
        except Exception as exc:  # surface the failure to every waiter in this batch
            for p in batch:
                if not p.future.done():
                    p.future.set_exception(exc)
            return
        generate_s = time.perf_counter() - start

        sample_rate = self.model.sampling_rate
        for p, audio in zip(batch, audios):
            total_s = time.perf_counter() - p.submitted_at
            if not p.future.done():
                p.future.set_result(
                    TTSResult(
                        audio=audio,
                        sample_rate=sample_rate,
                        queue_wait_s=start - p.submitted_at,
                        generate_s=generate_s,
                        total_s=total_s,
                        batch_size=n,
                    )
                )

    def _run_generate(self, texts, languages, ref_audios, ref_texts, instructs, speeds):
        # Runs in a worker thread (via run_in_executor) so concurrent
        # asyncio.sleep() debounce windows for other buckets aren't blocked
        # by this batch's GPU-bound generate() call.
        with torch.inference_mode():
            return self.model.generate(
                text=texts,
                language=languages,
                ref_audio=ref_audios,
                ref_text=ref_texts,
                instruct=instructs,
                speed=speeds,
                **self.generation_kwargs,
            )
