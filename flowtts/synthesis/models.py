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
import re
import time
from pathlib import Path

import structlog

from flowtts.core.config import settings

logger = structlog.get_logger(__name__)


class FlowTtsSynthesizer:
    """Loads sgl.Engine + TTSCodec once; synthesizes text → audio token string."""

    # Silence-token filtering constants
    _SILENCE_TOKEN  = "<|speech_token_2973|>"
    _SILENCE_STREAK = 30  # consecutive silence tokens → early stop (30 × 20ms = 600ms)

    def __init__(self) -> None:
        self._engine = None
        self._tts_codec = None
        self._context_tokens: str = ""
        self._ref_speech_tokens = None
        self._sampling_params: dict = {}
        # Per-voice encoded tokens: {voice_id: (context_tokens, ref_speech_tokens)}
        self._voice_tokens: dict[str, tuple[str, object]] = {}
        self._lora_map: dict = {}
        # LoRA swap tracking: counts how many times the active adapter changed
        self._lora_swap_count: int = 0
        self._last_lora_path: object = object()  # sentinel — never equals None or a string

    async def initialize(self) -> None:
        """Load model. Mirrors main() in TTSIntegration/ws_server.py."""
        if self._engine is not None:
            return  # already initialized — don't reload

        cfg = settings.tts_model
        dec = settings.decoder
        model_path = cfg.model_dir

        # Accept either a local directory path or a HuggingFace repo ID (owner/name).
        # Local paths are validated; HF repo IDs are resolved by sglang at engine init.
        if "/" in model_path and not Path(model_path).exists():
            # Looks like a HF repo ID — sglang will download/use cache
            logger.info("model_path_is_hf_repo", repo=model_path)
        elif not Path(model_path).is_dir():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info("loading_codec_and_ref_audio", ref_audio=cfg.ref_audio)
        from flowtts.decoder.ncodec.codec import TTSCodec
        from flowtts.core.config import VOICE_REF_AUDIO
        tts_codec = TTSCodec(
            max_batch_size=dec.max_batch,
            batch_timeout_ms=dec.batch_timeout_ms,
            gpu_chunk_size=dec.gpu_chunk_size,
            onnx_workers=dec.onnx_workers,
            use_trt=dec.use_trt,
        )

        def _encode_ref(ref_path: str) -> tuple[str, object]:
            """Encode a ref audio → (context_tokens, ref_speech_tokens)."""
            if ref_path and os.path.isfile(ref_path):
                try:
                    ref_enc = tts_codec.encode(ref_path)
                    if isinstance(ref_enc, tuple) and len(ref_enc) == 2:
                        return ref_enc[1], ref_enc[0]  # context_tokens, ref_speech_tokens
                    return ref_enc, None
                except Exception as e:
                    logger.warning("ref_audio_failed", path=ref_path, error=str(e))
            else:
                logger.warning("ref_audio_not_found", path=ref_path)
            return self._default_context(), None

        # Encode default voice (used as fallback)
        context_tokens, ref_speech_tokens = _encode_ref(cfg.ref_audio)
        logger.info("ref_audio_loaded", path=cfg.ref_audio)

        # Encode all named voices for per-request switching
        voice_tokens: dict[str, tuple[str, object]] = {}
        for voice_id, v_path in VOICE_REF_AUDIO.items():
            v_ctx, v_ref = _encode_ref(v_path)
            voice_tokens[voice_id] = (v_ctx, v_ref)
            logger.info("voice_loaded", voice_id=voice_id, path=v_path)

        lora_map = cfg.language_lora_map or {}
        logger.info(
            "loading_sglang_engine",
            model=model_path,
            lora_languages=list(lora_map.keys()),
        )
        import sglang as sgl

        engine = sgl.Engine(
            model_path=model_path,
            tokenizer_path=model_path,
            mem_fraction_static=cfg.mem_fraction_static,
            trust_remote_code=True,
            dtype=cfg.dtype,
            attention_backend=cfg.attention_backend,
            chunked_prefill_size=cfg.chunked_prefill_size,
            disable_radix_cache=cfg.disable_radix_cache,
            disable_cuda_graph=cfg.disable_cuda_graph,
            disable_overlap_schedule=True,
            lora_paths=lora_map,
            max_loras_per_batch=len(lora_map) + 1,  # +1 for base-model slot (en, lora_path=None)
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
        self._voice_tokens = voice_tokens
        self._engine = engine
        self._sampling_params = sampling_params
        self._lora_map = lora_map
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

        # Store for external consumption (e.g. Prometheus registration in server.py)
        self.engine_info = {
            "tp_size":           getattr(sa, "tp_size", 1),
            "attention_backend": resolved_attn,
            "mem_weight_gb":     mem.get("weight", "n/a"),
            "mem_kvcache_gb":    mem.get("kvcache", "n/a"),
        }

        # ref_audio: inspect what we actually encoded
        ctx_token_count = context_tokens.count("<|context_token_")
        ref_speech_present = ref_speech_tokens is not None and bool(ref_speech_tokens)
        ref_source = cfg.ref_audio if os.path.isfile(cfg.ref_audio) else "default_context (hardcoded)"

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
        if lora_map:
            print(f"  lora_languages       : {list(lora_map.keys())}", flush=True)
            print(f"  default_language     : {cfg.default_language}", flush=True)
        print("=" * 60 + "\n", flush=True)

    def _tokens_for_voice(self, voice_id: str | None) -> tuple[str, object]:
        """Return (context_tokens, ref_speech_tokens) for the given voice, falling back to default."""
        if voice_id and voice_id in self._voice_tokens:
            return self._voice_tokens[voice_id]
        return self._context_tokens, self._ref_speech_tokens

    @staticmethod
    def _strip_trailing_silence(token_str: str) -> str:
        """Remove trailing <|speech_token_2973|> (silence) from a full token string."""
        silence = "<|speech_token_2973|>"
        while token_str.endswith(silence):
            token_str = token_str[: -len(silence)]
        return token_str

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Remove characters outside English (ASCII) and Hindi (Devanagari) scripts.

        Keeps: ASCII printable (English letters, digits, punctuation, spaces),
               Devanagari (U+0900–U+097F) for Hindi,
               Tamil (U+0B80–U+0BFF),
               Telugu (U+0C00–U+0C7F),
               Kannada (U+0C80–U+0CFF),
               Malayalam (U+0D00–U+0D7F).
        Removes: Urdu/Arabic, Chinese, Japanese, Korean, emoji, and all other scripts.
        """
        import re
        return re.sub(r'[^\x00-\x7Fऀ-ॿ஀-௿ఀ-౿ಀ-೿ഀ-ൿ]', '', text).strip()

    async def synthesize(self, text: str, voice_id: str | None = None, language: str | None = None) -> str:
        """Return full audio token string for the given text."""
        if self._engine is None or self._tts_codec is None:
            raise RuntimeError("FlowTtsSynthesizer not initialized")

        text = self._normalize_text(text)
        language = self._resolve_language(language)
        ctx, ref = self._tokens_for_voice(voice_id)
        prompt = self._tts_codec.format_prompt(text, ctx, ref)

        lora_path = None if language == "en" else language
        if lora_path != self._last_lora_path:
            self._lora_swap_count += 1
            self._last_lora_path = lora_path

        t0 = time.monotonic()
        logger.info("llm_call_start", text_preview=text[:40], prompt_len=len(prompt), voice_id=voice_id, language=language)

        result = await self._engine.async_generate(
            prompt, self._sampling_params, lora_path=lora_path
        )
        full_text = self._strip_trailing_silence(result["text"])

        duration = time.monotonic() - t0
        logger.info(
            "llm_call_end",
            text_preview=text[:40],
            duration_seconds=round(duration, 4),
            token_len=len(full_text),
            language=language,
        )
        return full_text

    async def synthesize_stream(self, text: str, voice_id: str | None = None, language: str | None = None):
        """Async generator yielding incremental speech token strings as the LLM produces them.

        Each yielded value is a string fragment like "<|speech_token_42|><|speech_token_7|>..."
        The final yield is an empty string signalling EOS.
        Silence tokens (2973) are buffered; 4 consecutive triggers early stop.
        """
        if self._engine is None or self._tts_codec is None:
            raise RuntimeError("FlowTtsSynthesizer not initialized")

        text = self._normalize_text(text)
        language = self._resolve_language(language)
        ctx, ref = self._tokens_for_voice(voice_id)
        prompt = self._tts_codec.format_prompt(text, ctx, ref)

        lora_path = None if language == "en" else language
        if lora_path != self._last_lora_path:
            self._lora_swap_count += 1
            self._last_lora_path = lora_path

        stream_params = {**self._sampling_params}
        generator = await self._engine.async_generate(
            prompt, stream_params, stream=True, lora_path=lora_path
        )

        prev_len       = 0
        silence_streak = 0
        pending_silence: list[str] = []  # silence tokens held until a real token follows

        async for chunk in generator:
            full_so_far: str = chunk.get("text", "")
            delta = full_so_far[prev_len:]
            prev_len = len(full_so_far)
            if not delta:
                continue

            tokens = re.findall(r"<\|speech_token_\d+\|>", delta)
            to_yield: list[str] = []
            for tok in tokens:
                if tok == self._SILENCE_TOKEN:
                    silence_streak += 1
                    pending_silence.append(tok)
                    if silence_streak >= self._SILENCE_STREAK:
                        logger.info("silence_early_stop", text_preview=text[:40], streak=silence_streak)
                        pending_silence.clear()
                        yield ""
                        return
                else:
                    if pending_silence:
                        to_yield.extend(pending_silence)
                        pending_silence.clear()
                    silence_streak = 0
                    to_yield.append(tok)

            if to_yield:
                yield "".join(to_yield)

        # EOS — discard trailing silence, don't flush pending_silence
        yield ""

    def _resolve_language(self, language: str | None) -> str:
        """Return a validated language tag, falling back to the configured default.

        "en" is always valid and uses the base model with no LoRA adapter.
        """
        lang = language or settings.tts_model.default_language
        if lang == "en":
            return "en"
        if self._lora_map and lang not in self._lora_map:
            raise ValueError(
                f"Unknown language '{lang}'. Supported: en (base model), {list(self._lora_map.keys())}"
            )
        return lang

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
