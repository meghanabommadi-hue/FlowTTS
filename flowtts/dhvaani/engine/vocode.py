"""Pipeline position: VOCODE STAGE — batched mel -> PCM, with resampling.

Role in pipeline:
  Sits between the flow scheduler and the stitcher. Coalesces the mels that
  finish at roughly the same moment into one vocoder call, applies the per-voice
  loudness restore, resamples to the client's requested rate, and hands back
  float32 numpy.

Micro-batching
--------------
The scheduler retires spans continuously, so without coalescing the vocoder
would run at batch 1 hundreds of times a second. The batching loop here is the
same pattern `flowtts/decoder/ncodec/codec.py` uses for the MiraTTS codec: block
for the first item, then keep draining the queue until either `max_batch` items
have accumulated or a short timeout expires, then dispatch once.

The timeout is deliberately small. Vocos is only a few percent of total GPU
time, so waiting long to fill a batch trades real time-to-first-byte for a
marginal throughput gain.

Resampling
----------
Done on the GPU, before the device-to-host copy. Beyond being faster than a CPU
resample, converting 24 kHz to 8 kHz telephony audio first cuts the PCIe
transfer by 3x.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np
import structlog

from flowtts.dhvaani.config import MODEL_SAMPLE_RATE, dhv_settings

logger = structlog.get_logger(__name__)


@dataclass
class _Job:
    mel: object            # torch.Tensor (frames, 100), already / feat_scale
    frames: int
    prompt_rms: float
    target_rms: float
    out_sample_rate: int
    future: asyncio.Future
    queued_at: float = field(default_factory=time.perf_counter)


class VocodeStage:
    """Async micro-batching front end for the Vocos vocoder."""

    def __init__(self, vocoder, settings=None):
        self._s = settings or dhv_settings
        self._voc = vocoder
        self._queue: asyncio.Queue[_Job] | None = None
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._resamplers: dict[tuple[int, int], object] = {}
        self._batches = 0
        self._items = 0
        self._wait_ns = 0.0
        self._run_ns = 0.0

        # Vocos is cheap relative to the flow decoder, so a large batch buys
        # little. Cap it and keep the collection window short.
        self._max_batch = max(1, min(self._s.engine.max_batch_size, 32))
        self._timeout_s = 0.0015

    # -- lifecycle -----------------------------------------------------------
    async def start(self) -> None:
        self._stopping = False
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="dhvaani-vocode")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # -- submission ----------------------------------------------------------
    async def submit(
        self,
        mel,
        frames: int,
        prompt_rms: float,
        target_rms: float,
        out_sample_rate: int,
    ) -> np.ndarray:
        if self._queue is None:
            await self.start()
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put(
            _Job(mel, frames, prompt_rms, target_rms, out_sample_rate, fut)
        )
        return await fut

    # -- batching loop -------------------------------------------------------
    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stopping:
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

            batch = [first]
            deadline = loop.time() + self._timeout_s
            while len(batch) < self._max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    return

            try:
                self._run_batch(batch)
            except Exception as e:
                logger.exception("vocode_batch_failed", n=len(batch), error=str(e))
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(e)

    @staticmethod
    def _resample(pcm, src: int, dst: int):
        """Resample on the GPU, before the device-to-host copy.

        Kernels are cached inside model/audio_compat, which also provides the
        torchaudio-free path needed on NGC containers.
        """
        from flowtts.dhvaani.model.audio_compat import resample

        return resample(pcm, src, dst)

    def _run_batch(self, batch: list[_Job]) -> None:
        import torch

        t0 = time.perf_counter()
        self._batches += 1
        self._items += len(batch)
        for job in batch:
            self._wait_ns += t0 - job.queued_at

        # Pad to the batch max; each output is cut to its own length inside the
        # vocoder, so padding never bleeds into a shorter item's audio.
        max_f = max(j.frames for j in batch)
        n_mels = batch[0].mel.shape[-1]
        dev = batch[0].mel.device
        dtype = batch[0].mel.dtype
        mel = torch.zeros((len(batch), max_f, n_mels), device=dev, dtype=dtype)
        for i, job in enumerate(batch):
            f = min(job.frames, job.mel.shape[0])
            mel[i, :f] = job.mel[:f]

        # Vocos wants (B, n_mels, T).
        wavs = self._voc.decode(mel.permute(0, 2, 1), [j.frames for j in batch])

        for job, wav in zip(batch, wavs):
            try:
                pcm = wav
                # Restore the speaker's real loudness. Upstream scales the
                # prompt UP to target_rms before mel extraction, so the model
                # generates at that level; this undoes it.
                if job.prompt_rms < job.target_rms and job.prompt_rms > 0:
                    pcm = pcm * (job.prompt_rms / job.target_rms)

                if job.out_sample_rate != MODEL_SAMPLE_RATE:
                    pcm = self._resample(pcm, MODEL_SAMPLE_RATE, job.out_sample_rate)

                arr = pcm.detach().to("cpu", dtype=torch.float32).numpy()
                if not job.future.done():
                    job.future.set_result(arr)
            except Exception as e:
                if not job.future.done():
                    job.future.set_exception(e)

        self._run_ns += time.perf_counter() - t0

    def stats(self) -> dict:
        return {
            "batches": self._batches,
            "items": self._items,
            "mean_batch": round(self._items / self._batches, 2) if self._batches else 0.0,
            "mean_wait_ms": round(self._wait_ns / self._items * 1000, 3) if self._items else 0.0,
            "mean_run_ms": round(self._run_ns / self._batches * 1000, 3) if self._batches else 0.0,
            "queue_depth": self._queue.qsize() if self._queue else 0,
            "vocoder": self._voc.stats() if hasattr(self._voc, "stats") else {},
        }
