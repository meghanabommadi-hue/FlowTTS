"""
Model loader / singleton for sglang TTS inference.

Provides a process-level singleton session.  infer.py imports and calls
``get_session()`` which loads everything on first access and caches it for
the lifetime of that process.

NOTE: running model_server.py and infer.py as separate commands does NOT
share state — each is its own process with its own singleton.  The model
always loads when infer.py starts.  Use infer.py directly; it loads once
then fires all parallel requests against the same in-process session.

TTSCodec is always loaded (needed for format_prompt + ref audio encoding).
``skip_decode=True`` runs the full LLM with proper prompts but skips
decode_async and WAV writing — useful for batch concurrency benchmarking.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]          # .../FlowTTS
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flowtts.core.config import settings   # noqa: E402


# ---------------------------------------------------------------------------
# Default context tokens
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ModelSession — holds every loaded object
# ---------------------------------------------------------------------------

@dataclass
class ModelSession:
    engine: object
    tokenizer: object
    tts_codec: object               # always loaded — needed for format_prompt
    context_tokens: str
    ref_speech_tokens: object | None
    skip_decode: bool = False       # True → skip decode_async / WAV writing
    _sampling_params: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    def get_sampling_params(self) -> dict:
        if not self._sampling_params:
            cfg = settings.tts_model
            self._sampling_params = {
                "max_new_tokens":      cfg.max_tokens,
                "temperature":         cfg.temperature,
                "top_p":               cfg.top_p,
                "top_k":               cfg.top_k,
                "repetition_penalty":  cfg.repetition_penalty,
                "min_p":               cfg.min_p,
                "ignore_eos":          False,
                "skip_special_tokens": False,
            }
        return self._sampling_params

    # ------------------------------------------------------------------
    def format_prompt(self, text: str) -> str:
        return self.tts_codec.format_prompt(text, self.context_tokens, self.ref_speech_tokens)

    # ------------------------------------------------------------------
    async def async_generate(self, input_ids: list[int]) -> str:
        result = await self.engine.async_generate(
            input_ids=input_ids,
            sampling_params=self.get_sampling_params(),
        )
        return result["text"]

    # ------------------------------------------------------------------
    async def decode_async(self, speech_token_str: str) -> object:
        return await self.tts_codec.decode_async(speech_token_str, self.context_tokens)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_engine(cfg, model_path: str):
    print(f"[model_server] loading sglang engine from {model_path} …", flush=True)
    import sglang as sgl
    return sgl.Engine(
        model_path=model_path,
        tokenizer_path=model_path,
        mem_fraction_static=cfg.mem_fraction_static,
        trust_remote_code=True,
        dtype=cfg.dtype,
        attention_backend=cfg.attention_backend,
        chunked_prefill_size=cfg.chunked_prefill_size,
        schedule_policy=cfg.schedule_policy,
        cuda_graph_max_bs=cfg.cuda_graph_max_bs,
        disable_radix_cache=cfg.disable_radix_cache,
        num_continuous_decode_steps=cfg.num_continuous_decode_steps,
    )


def _load_tokenizer(model_path: str):
    print(f"[model_server] loading HF tokenizer …", flush=True)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_session(skip_decode: bool = False) -> ModelSession:
    """Load all components and return a ``ModelSession``.

    TTSCodec is always loaded so prompt formatting and ref audio encoding
    work correctly.  When ``skip_decode=True`` the session skips
    decode_async / WAV writing — useful for pure LLM benchmarking.
    """
    cfg = settings.tts_model
    dec = settings.decoder
    model_path = cfg.model_dir

    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    context_tokens = _default_context()
    ref_speech_tokens = None

    print(f"[model_server] loading TTSCodec …", flush=True)
    from flowtts.decoder.ncodec.codec import TTSCodec
    tts_codec = TTSCodec(
        max_batch_size=dec.max_batch,
        batch_timeout_ms=dec.batch_timeout_ms,
        gpu_chunk_size=dec.gpu_chunk_size,
        onnx_workers=dec.onnx_workers,
        use_trt=dec.use_trt,
    )

    ref_path = cfg.ref_audio
    if ref_path and os.path.isfile(ref_path):
        print(f"[model_server] encoding ref audio: {ref_path}", flush=True)
        ref_enc = tts_codec.encode(ref_path)
        if isinstance(ref_enc, tuple) and len(ref_enc) == 2:
            ref_speech_tokens, context_tokens = ref_enc[0], ref_enc[1]
        else:
            context_tokens = ref_enc
        print(f"[model_server] ref audio encoded  ctx_tokens={context_tokens.count('<|context_token_')}", flush=True)
    else:
        print(f"[model_server] ref audio not found ({ref_path}), using default context", flush=True)

    engine    = _load_engine(cfg, model_path)
    tokenizer = _load_tokenizer(model_path)

    mode = "LLM + codec (decode skipped)" if skip_decode else "full pipeline"
    print(f"[model_server] ready  mode={mode}\n", flush=True)

    return ModelSession(
        engine=engine,
        tokenizer=tokenizer,
        tts_codec=tts_codec,
        context_tokens=context_tokens,
        ref_speech_tokens=ref_speech_tokens,
        skip_decode=skip_decode,
    )


# ---------------------------------------------------------------------------
# Module-level singleton — lazily initialised on first get_session() call
# ---------------------------------------------------------------------------

_SESSION: ModelSession | None = None


def get_session(skip_decode: bool = False) -> ModelSession:
    """Return the cached session, loading it on first call."""
    global _SESSION
    if _SESSION is None:
        _SESSION = load_session(skip_decode=skip_decode)
    return _SESSION


# ---------------------------------------------------------------------------
# __main__ — keep the session alive so infer.py can import this module
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test model loading (session is not shared with other processes)"
    )
    parser.add_argument("--llm-only", action="store_true",
                        help="Load codec for prompts but skip decode_async / WAV writing")
    args = parser.parse_args()

    session = get_session(skip_decode=args.llm_only)
    print("[model_server] session loaded OK — exiting (use infer.py to run requests)", flush=True)
