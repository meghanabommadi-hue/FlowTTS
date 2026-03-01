import re
import threading
import torch
import numpy as np
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
from flowtts.decoder.ncodec.model_utils import AudioTokenizer


class AudioDecoder:

    def __init__(
        self,
        decoder_paths: str,
        gpu_chunk_size: int = 50,
        onnx_workers: int = 2,
        use_trt: bool = False,
    ):
        self._decoder_paths = decoder_paths
        self._providers = [("CUDAExecutionProvider", {"device_id": 0})]
        # Max items per GPU forward pass.  Keeps peak activation memory bounded.
        self._gpu_chunk_size = gpu_chunk_size
        self._onnx_workers   = onnx_workers

        # Main-thread session — used by the synchronous single-item detokenize().
        self.processor_detokenizer = self._make_session()

        # Each thread that calls _run_onnx_chunk() gets its own ONNX session
        # via this thread-local.  The CUDAExecutionProvider binds its internal
        # CUDA context to the thread that calls InferenceSession(), so session
        # reuse across threads causes the Reshape/{0,6} error.
        self._thread_local = threading.local()

        # Persistent pool used to parallelise the ONNX serial loop.
        # Each worker thread gets its own session (created lazily on first use).
        self._onnx_executor = ThreadPoolExecutor(max_workers=onnx_workers)
        # Pre-warm: force GPU session creation in every pool thread now so
        # the first real batch doesn't pay the cold-start cost.
        futs = [self._onnx_executor.submit(self._get_session) for _ in range(onnx_workers)]
        for f in futs:
            f.result()

        self.audio_detokenizer = AudioTokenizer(
            f'{decoder_paths}/detokenizer.safetensors',
            use_trt=use_trt,
            gpu_chunk_size=gpu_chunk_size,
        )

    # ------------------------------------------------------------------ helpers

    def _make_session(self, providers=None) -> ort.InferenceSession:
        """Create a fresh ONNX session with the given (or default GPU) providers."""
        if providers is None:
            providers = self._providers
        opts = ort.SessionOptions()
        return ort.InferenceSession(
            f"{self._decoder_paths}/processer.onnx",
            opts,
            providers,
        )

    def _get_session(self) -> ort.InferenceSession:
        """Return a thread-local GPU ONNX session (main thread / single-item path)."""
        if not hasattr(self._thread_local, "session"):
            self._thread_local.session = self._make_session()
        return self._thread_local.session

    @staticmethod
    def _parse_tokens(ctx_str: str, spch_str: str):
        spch = np.array(
            [int(t) for t in re.findall(r"speech_token_(\d+)", spch_str)],
            dtype=np.int64,
        ).reshape(1, -1)
        ctx = np.array(
            [int(t) for t in re.findall(r"context_token_(\d+)", ctx_str)],
            dtype=np.int32,
        ).reshape(1, 1, -1)
        if spch.shape[1] == 0:
            raise ValueError(
                f"No speech tokens found in LLM output. "
                f"spch_str preview: {spch_str[:200]!r}"
            )
        if ctx.shape[2] == 0:
            raise ValueError(
                f"No context tokens found. "
                f"ctx_str preview: {ctx_str[:200]!r}"
            )
        return ctx, spch

    def _run_onnx_chunk(self, chunk: list[tuple[str, str]]) -> list:
        """Run ONNX on a sub-list of requests as a single batched call."""
        sess = self._get_session()
        parsed = [self._parse_tokens(ctx, spch) for ctx, spch in chunk]

        max_spch = max(p[1].shape[1] for p in parsed)
        max_ctx  = max(p[0].shape[2] for p in parsed)
        B = len(parsed)

        ctx_batch  = np.zeros((B, 1, max_ctx), dtype=np.int32)
        spch_batch = np.zeros((B, max_spch),   dtype=np.int64)
        for i, (ctx, spch) in enumerate(parsed):
            ctx_batch[i,  :, :ctx.shape[2]]  = ctx[0]
            spch_batch[i, :spch.shape[1]]    = spch[0]

        out = sess.run(
            ["preprocessed_output"],
            {"context_tokens": ctx_batch, "speech_tokens": spch_batch},
        )[0]  # [B, C, T]

        return [out[i:i+1] for i in range(B)]

    # ------------------------------------------------------------------ public

    @torch.inference_mode()
    def detokenize(self, context_tokens: str, speech_tokens: str):
        """Single-item decode (called from the main thread)."""
        spch = (
            torch.tensor([int(t) for t in re.findall(r"speech_token_(\d+)", speech_tokens)])
            .long().unsqueeze(0)
        ).numpy()
        ctx = (
            torch.tensor([int(t) for t in re.findall(r"context_token_(\d+)", context_tokens)])
            .long().unsqueeze(0).unsqueeze(0)
        ).numpy().astype(np.int32)

        x = self.processor_detokenizer.run(
            ["preprocessed_output"],
            {"context_tokens": ctx, "speech_tokens": spch},
        )

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x_t   = torch.from_numpy(x[0]).to("cuda:0")
            lowres = self.audio_detokenizer.decode(x_t).squeeze(0)
        stream.synchronize()
        return lowres.cpu()

    def detokenize_batch(self, requests: list[tuple[str, str]]) -> list:
        """
        Process B requests in three phases:

        Phase 1 — Parallel ONNX  (onnx_workers threads, each ~B/W items)
        Phase 2 — Chunked GPU decoder  (gpu_chunk_size items per forward pass)
        """
        B = len(requests)

        # --- Phase 1: split requests across ONNX worker threads ---
        # Ceiling-divide so every chunk is as equal as possible.
        csize  = max(1, -(-B // self._onnx_workers))   # ceiling division
        chunks = [requests[i : i + csize] for i in range(0, B, csize)]

        futures  = [self._onnx_executor.submit(self._run_onnx_chunk, c) for c in chunks]
        # Collect in submission order to preserve request ordering.
        onnx_outs = [out for fut in futures for out in fut.result()]

        # Stack into [B, C, T], zero-padding if token counts differ.
        C     = onnx_outs[0].shape[1]
        max_T = max(o.shape[2] for o in onnx_outs)
        if all(o.shape[2] == max_T for o in onnx_outs):
            x_batch = np.concatenate(onnx_outs, axis=0)
        else:
            x_batch = np.zeros((B, C, max_T), dtype=onnx_outs[0].dtype)
            for i, o in enumerate(onnx_outs):
                x_batch[i, :, : o.shape[2]] = o[0]

        # --- Phase 2: chunked GPU forward (prevents OOM on large batches) ---
        chunk   = self._gpu_chunk_size
        results: list = [None] * B
        with torch.inference_mode():
            stream = torch.cuda.Stream()
            for start in range(0, B, chunk):
                end = min(start + chunk, B)
                with torch.cuda.stream(stream):
                    x_t   = torch.from_numpy(x_batch[start:end]).to("cuda:0")
                    lowres = self.audio_detokenizer.decode(x_t)
                    for i, gi in enumerate(range(start, end)):
                        results[gi] = lowres[i].squeeze(0).cpu()
            stream.synchronize()
        return results
