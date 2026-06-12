"""Pipeline position: SYNTHESIS — Mira/sglang backend.

Implements BaseSynthesizer for the MiraITS model family (Hindi / Telugu).

Sequence inside synthesize():
  1. TTSCodec.format_prompt(text, context_tokens, ref_speech_tokens)
     → "<|task_tts|><|start_text|>{text}…<|prompt_speech_start|>"
  2. sgl.Engine.async_generate(prompt, sampling_params)
     → "<|speech_token_42|><|speech_token_7|>…"  (discrete token string)
  3. TTSCodec.decode_async(tokens, context_tokens)
     → WAV tensor at 16 kHz
  4. tensor_to_wav(tensor)  →  WAV bytes

synthesize_stream() batches tokens in rolling windows of
settings.streaming.chunk_tokens, decoding + yielding each window as a
separate WAV chunk (time-to-first-chunk optimisation).

Initialised once via MiraSynthesizer.initialize().
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import structlog

from flowtts.core.config import settings
from flowtts.decoder.decoder import tensor_to_wav, SAMPLE_RATE
from flowtts.synthesis.base import BaseSynthesizer, SynthChunk, SynthResult

logger = structlog.get_logger(__name__)

_RE_SPEECH = re.compile(r"<\|speech_token_\d+\|>", re.ASCII)
_OVERLAP_TOKENS = 4  # prepend to each chunk for codec conv context


class MiraSynthesizer(BaseSynthesizer):
    """MiraITS / sglang TTS backend.

    Produces discrete speech tokens via sglang, then decodes them with
    ncodec TTSCodec (ONNX + PyTorch on GPU).
    """

    def __init__(self) -> None:
        self._engine = None
        self._tts_codec = None
        self._context_tokens: str = ""
        self._ref_speech_tokens = None
        self._sampling_params: dict = {}

    # ------------------------------------------------------------------
    # BaseSynthesizer interface
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._engine is not None:
            return

        cfg = settings.tts_model
        dec = settings.decoder
        model_path = cfg.model_dir

        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Mira model not found: {model_path}")

        logger.info("mira_loading_codec", ref_audio=cfg.ref_audio)
        from flowtts.decoder.ncodec.codec import TTSCodec  # noqa: PLC0415

        tts_codec = TTSCodec(
            max_batch_size=dec.max_batch,
            batch_timeout_ms=dec.batch_timeout_ms,
            gpu_chunk_size=dec.gpu_chunk_size,
            onnx_workers=dec.onnx_workers,
            use_trt=dec.use_trt,
        )

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
                logger.info("mira_ref_audio_loaded", path=ref_path)
            except Exception as e:
                logger.warning("mira_ref_audio_failed", error=str(e))
                context_tokens = self._default_context()
        else:
            logger.warning("mira_ref_audio_not_found", path=ref_path)
            context_tokens = self._default_context()

        logger.info("mira_loading_engine", model=model_path)
        import sglang as sgl  # noqa: PLC0415

        engine = sgl.Engine(
            model_path=model_path,
            tokenizer_path=model_path,
            mem_fraction_static=cfg.mem_fraction_static,
            trust_remote_code=True,
            dtype=cfg.dtype,
            attention_backend=cfg.attention_backend,
            chunked_prefill_size=cfg.chunked_prefill_size,
            max_running_requests=cfg.max_running_requests,
            schedule_policy=cfg.schedule_policy,
            cuda_graph_max_bs=cfg.cuda_graph_max_bs,
            disable_radix_cache=cfg.disable_radix_cache,
            num_continuous_decode_steps=cfg.num_continuous_decode_steps,
        )

        self._sampling_params = {
            "max_new_tokens":     cfg.max_tokens,
            "temperature":        cfg.temperature,
            "top_p":              cfg.top_p,
            "top_k":              cfg.top_k,
            "repetition_penalty": cfg.repetition_penalty,
            "min_p":              cfg.min_p,
            "ignore_eos":         False,
            "skip_special_tokens": False,
        }

        self._tts_codec = tts_codec
        self._context_tokens = context_tokens
        self._ref_speech_tokens = ref_speech_tokens
        self._engine = engine

        # Log runtime stats
        si = getattr(engine, "scheduler_info", {}) or {}
        sa = engine.server_args
        resolved_attn = getattr(sa, "attention_backend", "n/a")
        mem = {}
        try:
            srv_info = engine.get_server_info()
            internal = srv_info.get("internal_states", [{}])
            mem = internal[0].get("memory_usage", {}) if internal else {}
            resolved_attn = srv_info.get("attention_backend") or resolved_attn
        except Exception:
            pass
        ctx_token_count = context_tokens.count("<|context_token_")
        ref_source = ref_path if (ref_path and os.path.isfile(ref_path)) else "default_context"

        print("\n" + "=" * 60, flush=True)
        print("  MiraSynthesizer — ready", flush=True)
        print("=" * 60, flush=True)
        print(f"  model             : {model_path}", flush=True)
        print(f"  tp_size           : {sa.tp_size}", flush=True)
        print(f"  attention_backend : {resolved_attn}", flush=True)
        if mem:
            print(f"  mem weight (GB)   : {mem.get('weight', 'n/a')}", flush=True)
            print(f"  mem kvcache (GB)  : {mem.get('kvcache', 'n/a')}", flush=True)
        print(f"  ref_audio         : {ref_source}", flush=True)
        print(f"  context_tokens    : {ctx_token_count} tokens", flush=True)
        print("=" * 60 + "\n", flush=True)
        logger.info("mira_ready")

    async def synthesize(self, text: str, voice_id: str | None = None) -> SynthResult:
        if self._engine is None:
            raise RuntimeError("MiraSynthesizer not initialized")

        prompt = self._tts_codec.format_prompt(text, self._context_tokens, self._ref_speech_tokens)

        t0 = time.perf_counter()
        result = await self._engine.async_generate(prompt, self._sampling_params)
        llm_s = time.perf_counter() - t0
        token_str: str = result["text"]
        n_tokens = token_str.count("<|speech_token_")

        td = time.perf_counter()
        wav_tensor = await self._tts_codec.decode_async(token_str, self._context_tokens)
        decode_s = time.perf_counter() - td

        decoded = tensor_to_wav(wav_tensor)
        return SynthResult(
            wav_bytes=decoded.wav_bytes,
            sample_rate=SAMPLE_RATE,
            n_tokens=n_tokens,
            llm_s=round(llm_s, 4),
            decode_s=round(decode_s, 4),
        )

    async def synthesize_stream(self, text: str, voice_id: str | None = None) -> AsyncGenerator[SynthChunk, None]:
        if self._engine is None:
            raise RuntimeError("MiraSynthesizer not initialized")

        codec = self._tts_codec
        ctx   = self._context_tokens
        chunk_tokens_target = settings.streaming.chunk_tokens

        buffer: str = ""
        token_buf: list[str] = []
        overlap_tokens: list[str] = []
        chunk_index = 0
        total_tokens = 0
        decode_total = 0.0

        prompt = self._tts_codec.format_prompt(text, self._context_tokens, self._ref_speech_tokens)
        generator = await self._engine.async_generate(prompt, self._sampling_params, stream=True)

        async def _flush(is_final: bool):
            nonlocal chunk_index, total_tokens, decode_total, overlap_tokens
            if not token_buf:
                return

            real_tokens   = list(token_buf)
            token_buf.clear()
            decode_tokens = overlap_tokens + real_tokens
            n_overlap     = len(overlap_tokens)
            overlap_tokens = real_tokens[-_OVERLAP_TOKENS:]

            td = time.perf_counter()
            wav_tensor = await codec.decode_async("".join(decode_tokens), ctx)
            decode_total += time.perf_counter() - td

            pcm = np.asarray(wav_tensor, dtype=np.float32).squeeze()
            if n_overlap > 0:
                pcm = pcm[n_overlap * 320:]   # 1 token = 320 samples @ 16 kHz

            decoded = tensor_to_wav(pcm)
            n_tok = len(real_tokens)
            total_tokens += n_tok
            chunk_index  += 1

            meta = {}
            if is_final:
                meta = {"decode_s": round(decode_total, 4), "n_tokens": total_tokens}

            yield SynthChunk(
                wav_bytes=decoded.wav_bytes,
                is_final=is_final,
                sample_rate=SAMPLE_RATE,
                n_tokens=n_tok,
                meta=meta,
            )

        prev_len = 0
        async for chunk in generator:
            full_so_far: str = chunk.get("text", "")
            delta = full_so_far[prev_len:]
            prev_len = len(full_so_far)
            if not delta:
                continue

            buffer += delta
            last_end = 0
            for m in _RE_SPEECH.finditer(buffer):
                token_buf.append(m.group())
                last_end = m.end()
            if last_end:
                buffer = buffer[last_end:]

            while len(token_buf) >= chunk_tokens_target:
                async for sc in _flush(is_final=False):
                    yield sc

        # EOS — flush remainder
        async for sc in _flush(is_final=True):
            yield sc

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
