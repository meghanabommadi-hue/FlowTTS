"""Pipeline position: SYNTHESIS — text → audio token string.

Role in pipeline:
  The GPU-heavy core of the pipeline. Takes a plain-text utterance and
  returns a speech token string that encodes the audio at the token level.
  The downstream decoder (flowtts/decoder/decoder.py) converts tokens → PCM.

Sequence inside FlowTtsSynthesizer.synthesize():
  1. Format prompt:
       TTSCodec.format_prompt(text, context_tokens, ref_speech_tokens)
       → "<|task_tts|><|start_text|>{text}…<|prompt_speech_start|>"
  2. LLM inference:
       sgl.Engine.async_generate(prompt, sampling_params)
       → raw output text  "<|speech_token_42|><|speech_token_7|>…"
  3. Return token string to caller (worker.py or server.py).

Initialisation (done once, lazy):
  • Downloads model via HuggingFace Hub (ncodec.TTSCodec).
  • Encodes reference audio (cfg.ref_audio) to get:
      - context_tokens  — speaker timbre/style descriptor.
      - ref_speech_tokens — optional speech prefix for the LLM prompt.
  • Loads sglang Engine with the configured model_dir, dtype, mem_fraction.

Performance notes:
  • sglang Engine holds the GPU for the lifetime of the process.
  • temperature=0.0 (greedy) by default — deterministic output, fastest.
  • max_new_tokens=512 limits utterance length; raise for longer sentences.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import structlog

from flowtts.core.config import settings

logger = structlog.get_logger(__name__)


class FlowTtsSynthesizer:
    """Loads sgl.Engine + TTSCodec once; synthesizes text → audio token string."""

    def __init__(self) -> None:
        self._engine = None
        self._tts_codec = None
        self._context_tokens: str = ""
        self._ref_speech_tokens = None
        self._sampling_params: dict = {}

    async def initialize(self) -> None:
        """Load model. Mirrors main() in TTSIntegration/ws_server.py."""
        if self._engine is not None:
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

        logger.info("loading_sglang_engine", model=model_path)
        import sglang as sgl

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

        sampling_params = {
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
        self._sampling_params = sampling_params
        logger.info("synthesizer_ready")

        # Inspect live engine objects — not config.py values.
        # get_server_info() round-trips to the scheduler subprocess and returns
        # global_server_args_dict, which contains the attention_backend actually
        # selected by model_runner.init_attention_backend() after model load.
        si = getattr(engine, "scheduler_info", {}) or {}
        sa = engine.server_args
        # Read attention_backend directly from server_args (set by sglang after check_server_args())
        # then confirm via get_server_info() which merges asdict(server_args) into the response.
        resolved_attn = getattr(sa, "attention_backend", None) or "n/a"
        mem = {}
        try:
            srv_info = engine.get_server_info()
            internal = srv_info.get("internal_states", [{}])
            mem = internal[0].get("memory_usage", {}) if internal else {}
            # get_server_info includes asdict(server_args) so attention_backend is a top-level key
            resolved_attn = srv_info.get("attention_backend") or resolved_attn
        except Exception:
            pass

        # ref_audio: inspect what we actually encoded
        ctx_token_count = context_tokens.count("<|context_token_")
        ref_speech_present = ref_speech_tokens is not None and bool(ref_speech_tokens)
        ref_source = cfg.ref_audio if (ref_path and os.path.isfile(ref_path)) else "default_context (hardcoded)"

        print("\n" + "=" * 60, flush=True)
        print("  FlowTTS — Engine runtime stats (from model)", flush=True)
        print("=" * 60, flush=True)
        print(f"  tp_size              : {sa.tp_size}", flush=True)
        print(f"  attention_backend    : {resolved_attn}", flush=True)
        print(f"  max_total_num_tokens : {si.get('max_total_num_tokens', 'n/a')}", flush=True)
        print(f"  max_req_input_len    : {si.get('max_req_input_len', 'n/a')}", flush=True)
        if mem:
            print(f"  mem weight (GB)      : {mem.get('weight', 'n/a')}", flush=True)
            print(f"  mem kvcache (GB)     : {mem.get('kvcache', 'n/a')}", flush=True)
            print(f"  mem graph (GB)       : {mem.get('graph', 'n/a')}", flush=True)
        print(f"  ref_audio source     : {ref_source}", flush=True)
        print(f"  context_tokens       : {ctx_token_count} tokens encoded", flush=True)
        print(f"  ref_speech_tokens    : {'present' if ref_speech_present else 'absent'}", flush=True)
        print("=" * 60 + "\n", flush=True)

    async def synthesize(self, text: str) -> str:
        """Return full audio token string for the given text."""
        if self._engine is None or self._tts_codec is None:
            raise RuntimeError("FlowTtsSynthesizer not initialized")

        prompt = self._tts_codec.format_prompt(
            text, self._context_tokens, self._ref_speech_tokens
        )

        t0 = time.monotonic()
        logger.info("llm_call_start", text_preview=text[:40], prompt_len=len(prompt))

        result = await self._engine.async_generate(prompt, self._sampling_params)
        full_text = result["text"]

        duration = time.monotonic() - t0
        logger.info(
            "llm_call_end",
            text_preview=text[:40],
            duration_seconds=round(duration, 4),
            token_len=len(full_text),
        )
        return full_text

    async def synthesize_stream(self, text: str):
        """Async generator yielding incremental speech token strings as the LLM produces them.

        Each yielded value is a string fragment like "<|speech_token_42|><|speech_token_7|>..."
        The final yield is an empty string signalling EOS.
        """
        if self._engine is None or self._tts_codec is None:
            raise RuntimeError("FlowTtsSynthesizer not initialized")

        prompt = self._tts_codec.format_prompt(
            text, self._context_tokens, self._ref_speech_tokens
        )

        stream_params = {**self._sampling_params}
        generator = await self._engine.async_generate(prompt, stream_params, stream=True)

        prev_len = 0
        async for chunk in generator:
            full_so_far: str = chunk.get("text", "")
            delta = full_so_far[prev_len:]
            prev_len = len(full_so_far)
            if delta:
                yield delta
        # signal end
        yield ""

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
