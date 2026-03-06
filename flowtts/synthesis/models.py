"""Pipeline position: SYNTHESIS — text → audio token string.

Role in pipeline:
  The GPU-heavy core of the pipeline. Takes a plain-text utterance and
  returns a speech token string that encodes the audio at the token level.
  The downstream decoder (flowtts/decoder/decoder.py) converts tokens → PCM.

Sequence inside FlowTtsSynthesizer.synthesize():
  1. Format prompt:
       TTSCodec.format_prompt(text, context_tokens, ref_speech_tokens)
       → "<|task_tts|><|start_text|>{text}…<|prompt_speech_start|>"
  2. LLM inference (via dedicated lmdeploy thread):
       lmdeploy pipeline(batch_of_prompts, gen_config, do_preprocess=False)
       → raw output text  "<|speech_token_42|><|speech_token_7|>…"
  3. Return token string to caller (worker.py or server.py).

Initialisation (done once, lazy):
  • Downloads model via HuggingFace Hub (ncodec.TTSCodec).
  • Encodes reference audio (cfg.ref_audio) to get:
      - context_tokens  — speaker timbre/style descriptor.
      - ref_speech_tokens — optional speech prefix for the LLM prompt.
  • Loads lmdeploy TurbomindEngine with the configured model_dir, dtype, cache settings.

Threading model:
  lmdeploy runs in a DEDICATED background thread (not asyncio's executor).
  This eliminates GIL contention between the asyncio event loop and lmdeploy:
  - While lmdeploy is running inference, asyncio is idle (all requests await fut)
  - While asyncio is processing requests, lmdeploy thread sleeps (collecting batch)
  The thread uses stdlib queue.Queue (not asyncio.Queue) and time.sleep() for
  the 1ms batch-collection window — neither blocks the event loop.

Performance notes:
  • lmdeploy pipeline holds the GPU for the lifetime of the process.
  • temperature=0.0 (greedy) by default — deterministic output, fastest.
  • max_new_tokens limits utterance length; raise for longer sentences.
"""

from __future__ import annotations

import asyncio
import os
import queue as stdlib_queue
import threading
import time
from pathlib import Path
from typing import NamedTuple

import structlog

from flowtts.core.config import settings

logger = structlog.get_logger(__name__)


class _BatchItem(NamedTuple):
    prompt: str
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future


class FlowTtsSynthesizer:
    """Loads lmdeploy pipeline + TTSCodec once; synthesizes text → audio token string.

    Concurrent synthesize() calls are batched by a dedicated background thread
    that collects items for llm_batch_timeout_ms, then flushes as one pipe() call.
    """

    def __init__(self) -> None:
        self._pipe = None
        self._tts_codec = None
        self._context_tokens: str = ""
        self._ref_speech_tokens = None
        self._gen_config = None
        self._thread_queue: stdlib_queue.Queue[_BatchItem | None] | None = None
        self._lmdeploy_thread: threading.Thread | None = None
        self._running: bool = False

    async def initialize(self) -> None:
        """Load model. Mirrors main() in TTSIntegration/ws_server.py."""
        if self._pipe is not None:
            return  # already initialized — don't reload

        cfg = settings.tts_model
        dec = settings.decoder
        model_path = cfg.model_dir

        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info("loading_codec_and_ref_audio", ref_audio=cfg.ref_audio)
        from flowtts.decoder.ncodec.codec import TTSCodec
        tts_codec = TTSCodec(
            max_batch_size=dec.max_batch,
            batch_timeout_ms=dec.batch_timeout_ms,
            gpu_chunk_size=dec.gpu_chunk_size,
            onnx_workers=dec.onnx_workers,
            use_trt=dec.use_trt,
        )

        # Encode reference audio to get context tokens + ref speech tokens
        ref_path = cfg.ref_audio
        context_tokens: str
        ref_speech_tokens = None
        if ref_path and os.path.isfile(ref_path):
            try:
                ref_enc = tts_codec.encode(ref_path)
                if isinstance(ref_enc, tuple) and len(ref_enc) == 2:
                    ref_speech_tokens, context_tokens = ref_enc[0], ref_enc[1]
                else:
                    context_tokens = ref_enc
                logger.info("ref_audio_loaded", path=ref_path)
            except Exception as e:
                logger.warning("ref_audio_failed", error=str(e), using="default_context")
                context_tokens = self._default_context()
        else:
            logger.warning("ref_audio_not_found", path=ref_path, using="default_context")
            context_tokens = self._default_context()

        logger.info("loading_lmdeploy_pipeline", model=model_path)
        from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

        backend_config = TurbomindEngineConfig(
            cache_max_entry_count=cfg.cache_max_entry_count,
            tp=cfg.tp,
            dtype=cfg.dtype,
            enable_prefix_caching=cfg.enable_prefix_caching,
            max_batch_size=cfg.max_batch_size,
        )
        pipe = pipeline(model_path, backend_config=backend_config)

        gen_config = GenerationConfig(
            max_new_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            min_p=cfg.min_p,
            do_sample=cfg.temperature > 0.0,
            skip_special_tokens=False,
            stop_words=[],
        )

        self._tts_codec = tts_codec
        self._context_tokens = context_tokens
        self._ref_speech_tokens = ref_speech_tokens
        self._pipe = pipe
        self._gen_config = gen_config

        # Start dedicated lmdeploy thread
        self._thread_queue = stdlib_queue.Queue()
        self._running = True
        self._lmdeploy_thread = threading.Thread(
            target=self._lmdeploy_worker,
            name="lmdeploy-worker",
            daemon=True,
        )
        self._lmdeploy_thread.start()

        logger.info("synthesizer_ready")

        # Log key runtime info
        ctx_token_count = context_tokens.count("<|context_token_")
        ref_speech_present = ref_speech_tokens is not None and bool(ref_speech_tokens)
        ref_source = cfg.ref_audio if (ref_path and os.path.isfile(ref_path)) else "default_context (hardcoded)"

        print("\n" + "=" * 60, flush=True)
        print("  FlowTTS — lmdeploy Engine runtime stats", flush=True)
        print("=" * 60, flush=True)
        print(f"  tp                   : {cfg.tp}", flush=True)
        print(f"  dtype                : {cfg.dtype}", flush=True)
        print(f"  cache_max_entry_count: {cfg.cache_max_entry_count}", flush=True)
        print(f"  enable_prefix_caching: {cfg.enable_prefix_caching}", flush=True)
        print(f"  max_batch_size       : {cfg.max_batch_size}", flush=True)
        print(f"  llm_batch_timeout_ms : {cfg.llm_batch_timeout_ms}", flush=True)
        print(f"  ref_audio source     : {ref_source}", flush=True)
        print(f"  context_tokens       : {ctx_token_count} tokens encoded", flush=True)
        print(f"  ref_speech_tokens    : {'present' if ref_speech_present else 'absent'}", flush=True)
        print("=" * 60 + "\n", flush=True)

    def _lmdeploy_worker(self) -> None:
        """Dedicated thread: collect batch → call pipe() → deliver results.

        Runs entirely outside asyncio — no GIL contention with the event loop.
        While lmdeploy is doing GPU inference, asyncio is idle (all requests
        are awaiting their futures). While asyncio is processing new requests,
        this thread is sleeping (collecting the next batch).
        """
        cfg = settings.tts_model
        max_batch = cfg.max_batch_size
        timeout_s = cfg.llm_batch_timeout_ms / 1000.0

        assert self._thread_queue is not None

        while self._running:
            # Block until the first request arrives
            try:
                first = self._thread_queue.get(timeout=1.0)
            except stdlib_queue.Empty:
                continue

            if first is None:  # shutdown sentinel
                break

            # Sleep briefly so concurrent synthesize() calls can queue up.
            # This sleep is in a background thread — it does NOT block the
            # asyncio event loop. During this window, all concurrent requests
            # finish their format_prompt + queue.put() calls.
            time.sleep(timeout_s)

            items: list[_BatchItem] = [first]
            while len(items) < max_batch:
                try:
                    item = self._thread_queue.get_nowait()
                    if item is None:
                        break
                    items.append(item)
                except stdlib_queue.Empty:
                    break

            prompts = [item.prompt for item in items]
            print(f"[lmdeploy_thread] batch_size={len(prompts)}", flush=True)
            logger.info("llm_batch_dispatch", batch_size=len(prompts))
            t0 = time.monotonic()

            try:
                responses = self._pipe(
                    prompts, gen_config=self._gen_config, do_preprocess=False
                )
                elapsed = time.monotonic() - t0
                print(
                    f"[lmdeploy_thread] done  batch={len(prompts)}  elapsed={elapsed:.3f}s",
                    flush=True,
                )
                for item, resp in zip(items, responses):
                    item.loop.call_soon_threadsafe(item.future.set_result, resp.text)
            except Exception as e:
                for item in items:
                    item.loop.call_soon_threadsafe(item.future.set_exception, e)

    async def synthesize(self, text: str) -> str:
        """Return full audio token string for the given text.

        The returned string looks like "<|speech_token_0|><|speech_token_1|>..."
        which the decoder (ncodec TTSCodec.decode) converts to PCM.
        """
        if self._pipe is None or self._tts_codec is None or self._thread_queue is None:
            raise RuntimeError("FlowTtsSynthesizer not initialized")

        prompt = self._tts_codec.format_prompt(
            text, self._context_tokens, self._ref_speech_tokens
        )

        t0 = time.monotonic()
        logger.info("llm_call_start", text_preview=text[:40], prompt_len=len(prompt))

        loop = asyncio.get_event_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self._thread_queue.put(_BatchItem(prompt=prompt, loop=loop, future=fut))
        full_text = await fut

        duration = time.monotonic() - t0
        logger.info(
            "llm_call_end",
            text_preview=text[:40],
            duration_seconds=round(duration, 4),
            token_len=len(full_text),
        )
        return full_text

    @staticmethod
    def _default_context() -> str:
        return (
            "<|context_token_3991|><|context_token_1250|><|context_token_2828|>"
            "<|context_token_3303|><|context_token_1187|><|context_token_3021|>"
            "<|context_token_355|><|context_token_3767|><|context_token_3663|>"
            "<|context_token_837|><|context_token_731|><|context_token_3656|>"
            "<|context_token_757|><|context_token_3360|><|context_token_3250|>"
            "<|context_token_3626|><|context_token_1244|><|context_token_526|>"
            "<|context_token_3829|><|context_token_205|><|context_token_1619|>"
            "<|context_token_268|><|context_token_4024|><|context_token_3375|>"
            "<|context_token_3032|><|context_token_2180|><|context_token_3278|>"
            "<|context_token_1609|><|context_token_3685|><|context_token_1359|>"
            "<|context_token_2817|><|context_token_3999|>"
        )
