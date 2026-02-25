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
            disable_radix_cahce=cfg.disable_radix_cahce
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

    async def synthesize(self, text: str) -> str:
        """Return full audio token string for the given text.

        The returned string looks like "<|speech_token_0|><|speech_token_1|>..."
        which the decoder (ncodec TTSCodec.decode) converts to PCM.
        """
        if self._engine is None or self._tts_codec is None:
            raise RuntimeError("FlowTtsSynthesizer not initialized")

        prompt = self._tts_codec.format_prompt(
            text, self._context_tokens, self._ref_speech_tokens
        )

        result = await self._engine.async_generate(prompt, self._sampling_params)
        full_text = result["text"]

        logger.debug("synthesis_done", text_preview=text[:40], token_len=len(full_text))
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
