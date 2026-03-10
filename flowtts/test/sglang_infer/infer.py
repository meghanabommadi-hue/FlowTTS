"""
Direct sglang TTS inference — no WebSocket, no Redis, no server.

Model loading is handled by model_server.py.  This script only sends
requests to the already-loaded session and saves WAV output.

Usage
-----
    # single sentence
    python -m flowtts.test.sglang_infer.infer "नमस्ते, कैसे हैं आप?"

    # multiple sentences
    python -m flowtts.test.sglang_infer.infer \
        "नमस्ते, कैसे हैं आप?" \
        "मैं ठीक हूं, धन्यवाद।" \
        "आज मौसम बहुत अच्छा है।"

    # concurrency stress: send N copies in parallel
    python -m flowtts.test.sglang_infer.infer --parallel 10 "नमस्ते, कैसे हैं आप?"

    # read sentences from a text file (one per line)
    python -m flowtts.test.sglang_infer.infer --file sentences.txt

    # skip decoder — LLM-only mode for batch concurrency benchmarking
    python -m flowtts.test.sglang_infer.infer --llm-only --parallel 20 "नमस्ते, कैसे हैं आप?"

Output WAV files are written to ./sglang_infer_out/ by default.
Use --outdir to override.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]          # .../FlowTTS
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flowtts.core.config import settings                         # noqa: E402
from flowtts.test.sglang_infer.model_server import get_session  # noqa: E402


# ---------------------------------------------------------------------------
# Per-sentence inference
# ---------------------------------------------------------------------------

async def _infer_one(
    idx: int,
    text: str,
    outdir: Path,
    tok_executor,
    llm_only: bool,
) -> dict:
    """Run one sentence through the loaded session; return timing dict."""
    session = get_session(skip_decode=llm_only)
    loop = asyncio.get_event_loop()
    t_total_start = time.perf_counter()

    # 1. Format prompt (always uses tts_codec.format_prompt)
    prompt = session.format_prompt(text)

    # 2. Tokenize in thread (non-blocking)
    t_tok0 = time.perf_counter()
    input_ids: list[int] = await loop.run_in_executor(
        tok_executor,
        lambda: session.tokenizer(prompt, return_tensors=None)["input_ids"],
    )
    t_tok = time.perf_counter() - t_tok0

    # 3. LLM inference
    t_llm0 = time.perf_counter()
    speech_token_str = await session.async_generate(input_ids)
    t_llm = time.perf_counter() - t_llm0

    speech_token_count = speech_token_str.count("<|speech_token_")
    sample_rate = settings.decoder.sample_rate
    duration_s = speech_token_count * 320 / sample_rate   # 1 token = 320 samples @ 16kHz

    t_dec = 0.0
    t_wav = 0.0
    out_name = "(skipped)"

    if llm_only:
        print(
            f"[{idx:03d}] done  "
            f"tok={t_tok*1000:.1f}ms  llm={t_llm:.3f}s  "
            f"[decoder=skipped]  "
            f"speech_tokens={speech_token_count}  audio≈{duration_s:.2f}s  "
            f"RTF={t_llm/max(duration_s, 1e-6):.2f}",
            flush=True,
        )
    else:
        # 4. Decode tokens → waveform
        t_dec0 = time.perf_counter()
        wav_tensor = await session.decode_async(speech_token_str)
        t_dec = time.perf_counter() - t_dec0

        # 5. Save WAV
        t_wav0 = time.perf_counter()
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / f"out_{idx:03d}.wav"
        out_name = out_path.name
        pcm = wav_tensor.numpy()
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
        t_wav = time.perf_counter() - t_wav0

        print(
            f"[{idx:03d}] done  "
            f"tok={t_tok*1000:.1f}ms  llm={t_llm:.3f}s  dec={t_dec:.3f}s  "
            f"wav={t_wav*1000:.1f}ms  total={time.perf_counter()-t_total_start:.3f}s  "
            f"speech_tokens={speech_token_count}  audio≈{duration_s:.2f}s  "
            f"RTF={t_llm/max(duration_s, 1e-6):.2f}  "
            f"→ {out_name}",
            flush=True,
        )

    t_total = time.perf_counter() - t_total_start
    return {
        "idx": idx,
        "text": text,
        "tok_ms":  round(t_tok * 1000, 2),
        "llm_s":   round(t_llm, 4),
        "dec_s":   round(t_dec, 4),
        "wav_ms":  round(t_wav * 1000, 2),
        "total_s": round(t_total, 4),
        "speech_tokens": speech_token_count,
        "audio_s": round(duration_s, 3),
        "rtf":     round(t_llm / max(duration_s, 1e-6), 3),
        "out_path": out_name,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run(sentences: list[str], outdir: Path, parallel: bool, llm_only: bool = False) -> None:
    import concurrent.futures as cf

    # Ensure session is loaded before firing requests
    get_session(skip_decode=llm_only)

    tok_executor = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="tok")
    mode_tag = "LLM + codec prompts (decode skipped)" if llm_only else "full pipeline"
    print(f"Running {len(sentences)} sentence(s)  parallel={parallel}  mode={mode_tag}\n", flush=True)
    t0 = time.perf_counter()

    if parallel:
        tasks = [
            _infer_one(i, text, outdir, tok_executor, llm_only)
            for i, text in enumerate(sentences)
        ]
        results = await asyncio.gather(*tasks)
    else:
        results = []
        for i, text in enumerate(sentences):
            results.append(await _infer_one(i, text, outdir, tok_executor, llm_only))

    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60, flush=True)
    print(f"  sentences  : {len(results)}", flush=True)
    print(f"  wall time  : {elapsed:.3f}s  (parallel={'yes' if parallel else 'no'})", flush=True)
    if results:
        llms = [r["llm_s"]  for r in results]
        decs = [r["dec_s"]  for r in results]
        tots = [r["total_s"] for r in results]
        toks = [r["tok_ms"] for r in results]
        print(f"  tok_ms     : min={min(toks):.1f}  avg={sum(toks)/len(toks):.1f}  max={max(toks):.1f}", flush=True)
        print(f"  llm_s      : min={min(llms):.3f}  avg={sum(llms)/len(llms):.3f}  max={max(llms):.3f}", flush=True)
        if not llm_only:
            print(f"  dec_s      : min={min(decs):.3f}  avg={sum(decs)/len(decs):.3f}  max={max(decs):.3f}", flush=True)
        print(f"  total_s    : min={min(tots):.3f}  avg={sum(tots)/len(tots):.3f}  max={max(tots):.3f}", flush=True)
        if parallel and len(llms) > 1 and min(llms) > 0:
            ratio = max(llms) / min(llms)
            eff   = round(sum(llms) / max(llms), 1)
            print(f"  llm max/min: {ratio:.2f}x  effective_parallel≈{eff}/{len(llms)}", flush=True)
    if not llm_only:
        print(f"  output dir : {outdir}", flush=True)
    print("=" * 60, flush=True)

    tok_executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="sglang TTS inference (model loaded via model_server)")
    parser.add_argument("sentences", nargs="*", help="Text sentences to synthesize")
    parser.add_argument("--file", "-f", help="Text file with one sentence per line")
    parser.add_argument("--parallel", "-p", type=int, default=1,
                        help="Repeat a single sentence N times in parallel (stress test)")
    parser.add_argument("--outdir", "-o", default="sglang_infer_out",
                        help="Output directory for WAV files (default: sglang_infer_out)")
    parser.add_argument("--llm-only", action="store_true",
                        help="Use codec for prompt formatting but skip decode_async and WAV writing")
    args = parser.parse_args()

    sentences: list[str] = []

    if args.file:
        with open(args.file) as fh:
            sentences = [ln.strip() for ln in fh if ln.strip()]

    sentences.extend(args.sentences)

    if not sentences:
        sentences = [
            "नमस्ते. मैं बजाज finance से वाणी बोल रही हूं.",
            "क्या मैं customer से बात कर सकती हूं?",
            "आपकी EMI की अगली तारीख पांच तारीख है.",
        ]
        print(f"[info] no sentences provided, using {len(sentences)} demo sentences", flush=True)

    if args.parallel > 1:
        if len(sentences) == 1:
            sentences = sentences * args.parallel
        else:
            print(f"[info] --parallel ignored when multiple sentences are provided", flush=True)

    parallel_mode = len(sentences) > 1
    asyncio.run(run(sentences, Path(args.outdir), parallel_mode, llm_only=args.llm_only))


if __name__ == "__main__":
    main()
