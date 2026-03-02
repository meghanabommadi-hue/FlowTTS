import re
import threading
import torch
import numpy as np
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
from flowtts.decoder.ncodec.model_utils import AudioTokenizer

_RE_SPEECH  = re.compile(r"speech_token_(\d+)")
_RE_CONTEXT = re.compile(r"context_token_(\d+)")


def _trim_trailing_silence(
    wav: torch.Tensor,
    *,
    silence_threshold: float = 1e-3,
    min_tail_samples: int = 3200,
    window_samples: int = 1600,
) -> torch.Tensor:
    """Trim the repeating-frame noise tail produced by feature padding.

    Padding the feature batch with the last real frame causes the decoder
    to emit a periodic noise burst with high RMS (~0.07). Real speech always
    ends with a brief near-zero gap (breath/pause, RMS < 0.001). We scan
    backwards in windows to find that gap, then cut there + a small safety tail.
    """
    if wav.numel() == 0:
        return wav
    x = wav.view(-1)
    N = x.numel()

    # Walk backwards in windows looking for the near-zero gap at speech end.
    for end in range(N, 0, -window_samples):
        start = max(0, end - window_samples)
        rms = x[start:end].float().pow(2).mean().sqrt().item()
        if rms < silence_threshold:
            # Found the gap — keep up to this point + safety tail
            cut = min(N, end + min_tail_samples)
            if cut < N:
                return x[:cut]
            return wav  # gap is near the end, nothing significant to trim

    return wav  # no near-zero gap found — leave as-is


def _trim_by_token_ratio(
    wav: torch.Tensor,
    this_len: int,
    max_len: int,
    *,
    safety_margin: float = 0.95,
) -> torch.Tensor:
    """Trim tail proportionally to the relative speech token length.

    In batched decode we pad speech tokens up to max_len before ONNX.
    Shorter sequences therefore get extra frames at the end which can
    manifest as beeps.  We approximate the true end of each utterance by
    scaling the full waveform length by (this_len / max_len).
    """
    if wav.numel() == 0 or max_len <= 0 or this_len <= 0:
        return wav
    ratio = min(1.0, float(this_len) / float(max_len) * safety_margin)
    if ratio >= 1.0:
        return wav
    x = wav.view(-1)
    keep = max(1, int(x.numel() * ratio))
    return x[:keep]


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

        # Persistent CUDA stream — avoids allocating a new stream per batch call.
        self._stream = torch.cuda.Stream()

    # ------------------------------------------------------------------ helpers

    def _make_session(self, providers=None) -> ort.InferenceSession:
        """Create a fresh ONNX session with the given (or default GPU) providers."""
        if providers is None:
            providers = self._providers
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = False  # not needed for GPU inference
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
        spch = np.fromiter(_RE_SPEECH.findall(spch_str),  dtype=np.int64).reshape(1, -1)
        ctx  = np.fromiter(_RE_CONTEXT.findall(ctx_str),  dtype=np.int32).reshape(1, 1, -1)
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
        """Run ONNX on a sub-list of requests as a single batched call.

        Returns list of (tensor [1,C,T], real_token_count) tuples.
        real_token_count is used by the caller to cut exactly the right
        amount of audio — 1 token = 320 audio samples at 16kHz.
        """
        sess = self._get_session()
        parsed = [self._parse_tokens(ctx, spch) for ctx, spch in chunk]

        real_lengths = [p[1].shape[1] for p in parsed]  # actual token count per item
        max_spch = max(real_lengths)
        max_ctx  = max(p[0].shape[2] for p in parsed)
        B = len(parsed)

        ctx_batch  = np.zeros((B, 1, max_ctx), dtype=np.int32)
        spch_batch = np.zeros((B, max_spch),   dtype=np.int64)
        for i, (ctx, spch) in enumerate(parsed):
            ctx_batch[i, 0, :ctx.shape[2]] = ctx[0, 0]
            T = spch.shape[1]
            spch_batch[i, :T] = spch[0]
            if T < max_spch:
                spch_batch[i, T:] = spch[0, -1]  # pad with last real token, not 0

        out = sess.run(
            ["preprocessed_output"],
            {"context_tokens": ctx_batch, "speech_tokens": spch_batch},
        )
        batch_out = torch.from_numpy(out[0]).to("cuda:0")  # [B, C, T_out]
        return [(batch_out[i:i+1], real_lengths[i]) for i in range(B)]

    # ------------------------------------------------------------------ public

    @torch.inference_mode()
    def detokenize(self, context_tokens: str, speech_tokens: str):
        """Single-item decode (called from the main thread)."""
        spch = np.fromiter(_RE_SPEECH.findall(speech_tokens),   dtype=np.int64).reshape(1, -1)
        ctx  = np.fromiter(_RE_CONTEXT.findall(context_tokens), dtype=np.int32).reshape(1, 1, -1)

        x = self.processor_detokenizer.run(
            ["preprocessed_output"],
            {"context_tokens": ctx, "speech_tokens": spch},
        )

        with torch.cuda.stream(self._stream):
            x_t   = torch.from_numpy(x[0]).to("cuda:0")
            lowres = self.audio_detokenizer.decode(x_t).squeeze(0)
        self._stream.synchronize()
        return lowres.cpu()

    def detokenize_batch(self, requests: list[tuple[str, str]]) -> list:
        """
        Process B requests in three phases:

        Phase 1 — Parallel ONNX  (onnx_workers threads, each ~B/W items)
        Phase 2 — Chunked GPU decoder  (gpu_chunk_size items per forward pass)
        """
        import time as _time
        _t0 = _time.perf_counter()

        B = len(requests)

        # --- Phase 1: split requests across ONNX worker threads ---
        # Ceiling-divide so every chunk is as equal as possible.
        csize  = max(1, -(-B // self._onnx_workers))   # ceiling division
        chunks = [requests[i : i + csize] for i in range(0, B, csize)]

        futures  = [self._onnx_executor.submit(self._run_onnx_chunk, c) for c in chunks]
        # Collect in submission order; each entry is (tensor, real_token_count).
        onnx_outs = [item for fut in futures for item in fut.result()]
        _t1 = _time.perf_counter()

        # --- Phase 2: group by T, batch each group — no cross-item padding ---
        # 1 speech token = 320 audio samples at 16 kHz (upsample factor 8*5*4*2).
        # We know real_tokens per item, so we cut exactly real_tokens*320 + tail.
        _UPSAMPLE    = 320
        _TAIL_SAMPS  = 1600  # 100ms safety tail after last real token
        _TRIM_TOKENS = 3     # remove this many tokens from the end to cancel noise onset

        from collections import defaultdict as _dd
        groups: dict = _dd(list)  # T_feat -> [(orig_idx, tensor, real_tokens)]
        for i, (o, real_tok) in enumerate(onnx_outs):
            groups[o.shape[2]].append((i, o, real_tok))

        chunk    = self._gpu_chunk_size
        gpu_outs: list = [None] * B
        with torch.inference_mode():
            for T_val, items in groups.items():
                for sub_start in range(0, len(items), chunk):
                    sub = items[sub_start : sub_start + chunk]
                    indices   = [idx      for idx, _, _  in sub]
                    real_toks = [rt       for _,   _, rt in sub]
                    x_batch   = torch.cat([o for _, o, _ in sub], dim=0)
                    with torch.cuda.stream(self._stream):
                        lowres = self.audio_detokenizer.decode(x_batch)
                        for li, gi in enumerate(indices):
                            wav = lowres[li].squeeze(0)
                            # Cut to exact token length + safety tail, minus a few tokens of noise onset.
                            keep = min(wav.numel(), (real_toks[li] - _TRIM_TOKENS) * _UPSAMPLE + _TAIL_SAMPS)
                            gpu_outs[gi] = wav[:keep]
            self._stream.synchronize()
        _t2 = _time.perf_counter()

        # Move to CPU after sync so transfers don't block the GPU stream.
        result = [t.cpu() for t in gpu_outs]
        _t3 = _time.perf_counter()

        print(
            f"[decoder] B={B}  T_groups={len(groups)}"
            f"  onnx={(_t1-_t0)*1000:.1f}ms"
            f"  gpu={(_t2-_t1)*1000:.1f}ms"
            f"  cpu_xfer={(_t3-_t2)*1000:.1f}ms"
            f"  total={(_t3-_t0)*1000:.1f}ms",
            flush=True,
        )
        return result
