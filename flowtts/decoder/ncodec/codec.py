import gc
import asyncio
import torch
import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from flowtts.decoder.ncodec.model import AudioDecoder
from flowtts.decoder.ncodec.encoder import AudioEncoder
from huggingface_hub import snapshot_download


class TTSCodec:

    def __init__(
        self,
        max_batch_size: int = 128,
        batch_timeout_ms: float = 10.0,
        gpu_chunk_size: int = 50,
        onnx_workers: int = 2,
        use_trt: bool = False,
    ):
        d_path = snapshot_download("YatharthS/MiraTTS")
        d_path = f"{d_path}/decoders"
        self.audio_decoder = AudioDecoder(
            d_path,
            gpu_chunk_size=gpu_chunk_size,
            onnx_workers=onnx_workers,
            use_trt=use_trt,
        )
        self.audio_encoder = AudioEncoder(d_path)

        self._max_batch    = max_batch_size
        self._batch_timeout = batch_timeout_ms / 1000.0  # seconds

        # Single worker: the batch processor runs GPU work on one thread so
        # that PyTorch / ONNX don't fight over stream ordering.
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

        # These are initialised lazily on the first decode_async call so that
        # the asyncio event loop is already running when we create the task.
        self._queue: asyncio.Queue | None = None
        self._batch_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Batch queue internals
    # ------------------------------------------------------------------

    def _ensure_batch_loop(self) -> None:
        """Start the background batch-processing task if not already running."""
        if self._batch_task is None or self._batch_task.done():
            self._queue = asyncio.Queue()
            self._batch_task = asyncio.get_event_loop().create_task(
                self._batch_loop()
            )

    async def _batch_loop(self) -> None:
        """
        Continuously drains the request queue and dispatches batches to the
        thread pool for GPU inference.

        Batching strategy
        -----------------
        • Block until at least one request arrives.
        • Then keep pulling from the queue for up to `batch_timeout_ms` ms OR
          until `max_batch_size` requests have accumulated, whichever comes
          first.
        • Dispatch the whole batch as a single GPU call and resolve each
          caller's Future individually.
        """
        loop = asyncio.get_event_loop()
        while True:
            batch: list[tuple[str, str, asyncio.Future]] = []

            # Wait for the first request (with a generous idle timeout so the
            # task doesn't spin forever when there is no traffic).
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                batch.append(item)
            except asyncio.TimeoutError:
                continue

            # Fill the batch within the collection window.
            deadline = loop.time() + self._batch_timeout
            while len(batch) < self._max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # Unpack and dispatch.
            requests       = [(ctx, spch) for spch, ctx, _ in batch]
            caller_futures = [f for _, _, f in batch]

            try:
                results = await loop.run_in_executor(
                    self._executor,
                    self.audio_decoder.detokenize_batch,
                    requests,
                )
                for fut, res in zip(caller_futures, results):
                    if not fut.done():
                        fut.set_result(res)
            except Exception as exc:
                for fut in caller_futures:
                    if not fut.done():
                        fut.set_exception(exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, audio, encode_semantic=False, duration=8):
        if encode_semantic:
            speech_tokens, context_tokens = self.audio_encoder.encode(audio, True, duration=duration)
            return speech_tokens, context_tokens
        else:
            context_tokens = self.audio_encoder.encode(audio, False, duration=duration)
            return context_tokens

    def process_audio(self, wav, wav2):
        wav = wav.cpu().numpy()

        weight_1, weight_2 = self.weight_1, self.weight_2
        mixed_audio = (wav * weight_1) + (wav2 * weight_2)
        return mixed_audio

    def format_prompt(self, text, context_tokens, extra_tokens, semantic_tokens=None, transcript=None):
        if semantic_tokens:
            prompt = f"<|task_tts|><|start_text|>{text}<|end_text|><|context_audio_start|>{context_tokens}<|context_audio_end|><|prompt_speech_start|>{semantic_tokens}"
        else:
            prompt = f"<|task_tts|><|start_text|>{text}<|end_text|><|context_audio_start|>{context_tokens}<|context_audio_end|><|prompt_speech_start|>"
        return prompt

    def c_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

    def decode(self, speech_tokens, context_tokens, test_var=None):
        wav = self.audio_decoder.detokenize(
            context_tokens,
            speech_tokens,
        )
        return wav

    async def decode_async(self, speech_tokens: str, context_tokens: str):
        """
        Non-blocking decode.  Requests are automatically coalesced into
        batches and processed in a single GPU forward pass.
        """
        self._ensure_batch_loop()
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((speech_tokens, context_tokens, future))
        return await future
