#!/usr/bin/env python3
"""Benchmark harness — measures what actually determines whether this hits
200 RPS at sub-200 ms TTFB.

    # 1. per-stage latency breakdown at several text lengths
    python -m flowtts.dhvaani.test.bench latency --voice simran

    # 2. throughput sweep: ramp concurrency, report RPS / TTFB percentiles
    python -m flowtts.dhvaani.test.bench throughput --voice simran --max-concurrency 128

    # 3. flow-step microbenchmark against the analytic roofline
    python -m flowtts.dhvaani.test.bench step

    # 4. what the profiles cost each other
    python -m flowtts.dhvaani.test.bench profiles --voice simran

    # 5. capacity model: what config reaches a target RPS
    python -m flowtts.dhvaani.test.bench capacity --target-rps 200 --utterance-s 3

`step` and `capacity` need no voice and no text; the others synthesise for real.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

FRAME_RATE = 93.75
N_MELS = 100
FM_DIM = 512
FM_FF = 1536

# Sentences of increasing length, mixed script, representative of IVR traffic.
TEXTS = {
    "short": "नमस्ते, आपकी कैसे मदद करूं?",
    "medium": "आपकी ईएमआई दो हज़ार पांच सौ रुपये बकाया है, कृपया आज ही भुगतान करें।",
    "long": (
        "नमस्ते, मैं बजाज फाइनेंस से बोल रही हूं। आपकी ईएमआई दो हज़ार पांच सौ रुपये "
        "बकाया है। कृपया आज ही भुगतान करें, अन्यथा विलंब शुल्क लग सकता है। आप हमारे "
        "मोबाइल ऐप से, या यूपीआई, एनईएफटी और आईएमपीएस के माध्यम से भुगतान कर सकते हैं।"
    ),
    "english": "Your payment of one thousand two hundred rupees is due today, please pay now.",
    "tamil": "வணக்கம், உங்கள் கட்டணம் இரண்டாயிரம் ரூபாய் நிலுவையில் உள்ளது.",
}


# ---------------------------------------------------------------------------
# Analytic cost model (documented in docs/DHVAANI.md)
# ---------------------------------------------------------------------------
def flops_per_forward(frames: int) -> float:
    """FLOPs for ONE fm_decoder evaluation over `frames` frames.

    The U-net stacks are layers [2,2,4,4,4] at downsampling [1,2,4,2,1], so the
    effective work is ~10x `frames` token-layers. Each Zipformer2EncoderLayer is
    3 feedforwards, 2 attention blocks and 2 conv modules at width 512.
    """
    total = 0.0
    for nlayers, ds in ((2, 1), (2, 2), (4, 4), (4, 2), (4, 1)):
        t = -(-frames // ds)
        ff = 3 * 2 * 2 * FM_DIM * FM_FF
        attn_proj = 2 * 2 * FM_DIM * (4 * 32 * 2 + 4 * 12 * 2 + 4 * 4)
        conv = 2 * 2 * FM_DIM * 31
        total += nlayers * t * (ff + attn_proj + conv)
        total += nlayers * 2 * 2 * 2 * (t * t) * (4 * 32)   # attention matmuls
    total += 2 * frames * (300 * FM_DIM + FM_DIM * N_MELS)
    return total


def span_flops(prompt_s: float, gen_s: float, num_step: int, cfg: bool) -> float:
    frames = int(round((prompt_s + gen_s) * FRAME_RATE))
    return flops_per_forward(frames) * num_step * (2 if cfg else 1)


def span_plan(utterance_s: float, first_s: float, steady_s: float) -> list[float]:
    spans, rem = [], utterance_s
    s = min(first_s, rem)
    spans.append(s)
    rem -= s
    while rem > 0.05:
        s = min(steady_s, rem)
        spans.append(s)
        rem -= s
    return spans


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
async def _engine(args):
    from flowtts.dhvaani.config import apply_profile, dhv_settings
    from flowtts.dhvaani.engine.engine import DhvaaniEngine

    s = dhv_settings
    if args.profile:
        apply_profile(s, args.profile)
    if args.backend:
        s.backend.kind = args.backend
    eng = DhvaaniEngine(s)
    await eng.start()
    return eng, s


async def mode_latency(args) -> int:
    eng, s = await _engine(args)
    print(f"\nper-stage latency  (voice={args.voice}, profile={args.profile or 'default'}, "
          f"backend={eng.stats()['backend']})")
    print(f"{'text':10s} {'chars':>6s} {'spans':>6s} {'TTFB':>8s} {'total':>8s} "
          f"{'flow':>8s} {'vocode':>8s} {'norm':>7s} {'audio':>7s} {'RTF':>6s}")
    print("-" * 88)
    for name, text in TEXTS.items():
        for _ in range(args.warmup):
            await eng.synthesize(text, args.voice)
        rows = []
        for _ in range(args.iters):
            _pcm, m = await eng.synthesize(text, args.voice)
            rows.append(m)
        med = lambda f: statistics.median(getattr(r, f) for r in rows)  # noqa: E731
        print(f"{name:10s} {len(text):6d} {rows[0].n_spans:6d} "
              f"{med('ttfb_ms'):7.1f}m {med('total_ms'):7.1f}m {med('flow_ms'):7.1f}m "
              f"{med('vocode_ms'):7.1f}m {med('normalize_ms'):6.2f}m "
              f"{med('audio_s'):6.2f}s {statistics.median(r.rtf for r in rows):6.3f}")
    await eng.stop()
    return 0


async def mode_throughput(args) -> int:
    eng, s = await _engine(args)
    text = TEXTS[args.text]
    print(f"\nthroughput sweep  (text={args.text}, {len(text)} chars, "
          f"profile={args.profile or 'default'}, backend={eng.stats()['backend']})")
    print(f"{'conc':>5s} {'reqs':>6s} {'RPS':>7s} {'ttfb p50':>9s} {'p90':>7s} {'p99':>7s} "
          f"{'total p50':>10s} {'p99':>8s} {'RTF':>6s} {'batch':>6s} {'VRAM GiB':>9s}")
    print("-" * 96)

    conc = 1
    while conc <= args.max_concurrency:
        for _ in range(2):
            await eng.synthesize(text, args.voice)

        n = max(conc * args.rounds, conc)
        ttfb, total, audio = [], [], 0.0
        sem = asyncio.Semaphore(conc)

        async def one():
            async with sem:
                _p, m = await eng.synthesize(text, args.voice)
                ttfb.append(m.ttfb_ms)
                total.append(m.total_ms)
                return m.audio_s

        t0 = time.perf_counter()
        audio = sum(await asyncio.gather(*[one() for _ in range(n)]))
        elapsed = time.perf_counter() - t0

        st = eng.stats()
        vram = st.get("vram", {}).get("allocated", 0) / 2**30
        print(f"{conc:5d} {n:6d} {n / elapsed:7.1f} {pct(ttfb, 50):8.1f}m "
              f"{pct(ttfb, 90):6.1f}m {pct(ttfb, 99):6.1f}m "
              f"{pct(total, 50):9.1f}m {pct(total, 99):7.1f}m "
              f"{elapsed / audio * conc:6.3f} "
              f"{st['scheduler'].get('mean_batch', 0):6.1f} {vram:9.2f}")
        conc *= 2
    await eng.stop()
    return 0


async def mode_step(args) -> int:
    """Time the flow step alone across the (bucket x batch) grid."""
    import torch

    from flowtts.dhvaani.backends import build_backend
    from flowtts.dhvaani.config import dhv_settings
    from flowtts.dhvaani.model.loader import load_model

    s = dhv_settings
    if args.backend:
        s.backend.kind = args.backend
    loaded = load_model(s)
    be = build_backend(loaded, s)
    dev, dt = loaded.device, loaded.dtype

    buckets = [b for b in s.buckets.buckets if b in (128, 256, 384, 512, 768, 1024)]
    batches = [1, 4, 8, 16, 32, 64]
    peak = args.peak_tflops

    print(f"\nflow-step microbenchmark  (backend={be.name}, dtype={dt})")
    print(f"{'frames':>7s} {'batch':>6s} {'ms':>8s} {'TFLOPS':>8s} {'% roofline':>11s} "
          f"{'spans/s @8step':>15s}")
    print("-" * 64)
    for T in buckets:
        for B in batches:
            if not be.supports_bucket(B, T):
                continue
            x = torch.randn(B, T, N_MELS, device=dev, dtype=dt)
            t = torch.rand(B, device=dev)
            mask = torch.zeros(B, T, device=dev, dtype=torch.bool)
            for _ in range(3):
                be.fm_step(x, x, x, t, mask)
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                be.fm_step(x, x, x, t, mask)
            torch.cuda.synchronize(dev)
            ms = (time.perf_counter() - t0) / args.iters * 1000
            fl = flops_per_forward(T) * B
            tflops = fl / (ms / 1000) / 1e12
            # One span at 8 steps with CFG = 16 forwards of this shape.
            spans_s = B / (ms / 1000 * 16)
            print(f"{T:7d} {B:6d} {ms:8.3f} {tflops:8.1f} {tflops / peak * 100:10.1f}% "
                  f"{spans_s:15.1f}")
    be.close()
    return 0


async def mode_profiles(args) -> int:
    from flowtts.dhvaani.config import PROFILES

    text = TEXTS[args.text]
    print(f"\nprofile comparison  (text={args.text}, voice={args.voice})")
    print(f"{'profile':10s} {'steps':>6s} {'cfg':>5s} {'TTFB':>8s} {'total':>8s} "
          f"{'RTF':>6s} {'audio':>7s}")
    print("-" * 60)
    for name in ("fast", "balanced", "quality"):
        args.profile = name
        eng, s = await _engine(args)
        for _ in range(2):
            await eng.synthesize(text, args.voice)
        rows = [(await eng.synthesize(text, args.voice))[1] for _ in range(args.iters)]
        med = lambda f: statistics.median(getattr(r, f) for r in rows)  # noqa: E731
        print(f"{name:10s} {s.flow.num_step:6d} {s.flow.guidance_scale:5.1f} "
              f"{med('ttfb_ms'):7.1f}m {med('total_ms'):7.1f}m "
              f"{statistics.median(r.rtf for r in rows):6.3f} {med('audio_s'):6.2f}s")
        await eng.stop()
    return 0


def mode_capacity(args) -> int:
    """Analytic capacity model. No GPU required -- use it to pick a config
    before burning L40S time, then confirm with `throughput`."""
    peak = args.peak_tflops
    mfu = args.mfu
    budget = peak * 1e12 * mfu

    print(f"\ncapacity model   device peak {peak:.0f} TFLOPS fp16, assumed MFU {mfu:.0%}"
          f"  -> {budget / 1e12:.0f} effective TFLOPS")
    print(f"target: {args.target_rps} RPS of {args.utterance_s}s utterances\n")
    print(f"{'config':<38s} {'prompt':>7s} {'spans':>6s} {'GFLOP/req':>10s} "
          f"{'max RPS':>8s} {'verdict':>9s}")
    print("-" * 86)

    configs = [
        ("quality   16 step, CFG",  16, True,  1.5, 6.0),
        ("balanced   8 step, CFG",   8, True,  1.2, 4.5),
        ("balanced   8 step, no CFG", 8, False, 1.2, 4.5),
        ("fast       4 step, CFG",   4, True,  1.0, 5.0),
        ("fast       4 step, no CFG", 4, False, 1.0, 5.0),
    ]
    for prompt_s in args.prompt_seconds:
        for label, steps, cfg, first_s, steady_s in configs:
            spans = span_plan(args.utterance_s, first_s, steady_s)
            gf = sum(span_flops(prompt_s, g, steps, cfg) for g in spans)
            rps = budget / gf
            verdict = "OK" if rps >= args.target_rps else "short"
            print(f"{label:<38s} {prompt_s:6.1f}s {len(spans):6d} {gf / 1e9:10.1f} "
                  f"{rps:8.1f} {verdict:>9s}")
        print()

    print("Notes")
    print("  * CFG doubles the flow FLOPs exactly -- it runs the batch twice.")
    print("  * The prompt's frames are re-rendered on EVERY span, so prompt length")
    print("    multiplies through the whole request. Trimming 3s -> 2s is a ~20%")
    print("    throughput win at no quality cost for most speakers.")
    print("  * These are compute-roofline numbers. Real MFU depends on batch size;")
    print("    run `bench step` to measure it, then re-run with --mfu <measured>.")
    print("  * FP8 on Ada (L40S) roughly doubles peak; pass --peak-tflops 362.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="DhVaani benchmarks")
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p, needs_voice=True):
        if needs_voice:
            p.add_argument("--voice", default=None, help="voice_id (default: store default)")
        p.add_argument("--profile", default=None, choices=["fast", "balanced", "quality"])
        p.add_argument("--backend", default=None, choices=["torch", "trt", "triton"])
        p.add_argument("--iters", type=int, default=10)
        p.add_argument("--warmup", type=int, default=3)

    p = sub.add_parser("latency"); common(p)
    p = sub.add_parser("throughput"); common(p)
    p.add_argument("--max-concurrency", type=int, default=64)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--text", default="medium", choices=sorted(TEXTS))
    p = sub.add_parser("step"); common(p, needs_voice=False)
    p.add_argument("--peak-tflops", type=float, default=181.0)
    p = sub.add_parser("profiles"); common(p)
    p.add_argument("--text", default="medium", choices=sorted(TEXTS))
    p = sub.add_parser("capacity")
    p.add_argument("--target-rps", type=float, default=200.0)
    p.add_argument("--utterance-s", type=float, default=3.0)
    p.add_argument("--peak-tflops", type=float, default=181.0,
                   help="181 = L40S fp16 dense; 362 = fp8 or fp16 w/ sparsity")
    p.add_argument("--mfu", type=float, default=0.45)
    p.add_argument("--prompt-seconds", type=float, nargs="+", default=[3.0, 2.0])

    args = ap.parse_args()
    if args.mode == "capacity":
        return mode_capacity(args)
    fn = {"latency": mode_latency, "throughput": mode_throughput,
          "step": mode_step, "profiles": mode_profiles}[args.mode]
    return asyncio.run(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
