"""Gradio dashboard for FlowTTS Prometheus metrics.

Reads from:
  - monitoring/calls.jsonl  (per-call log)
  - prometheus_client registry (live in-process counters, if running embedded)

Launch standalone:
    python -m flowtts.monitoring.dashboard

Or import and call launch() to embed in the server process.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from collections import deque
from datetime import datetime

import gradio as gr

_CALLS_LOG = Path(__file__).parents[2] / "monitoring" / "calls.jsonl"

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_calls(n: int = 500) -> list[dict]:
    """Return the last *n* entries from calls.jsonl."""
    if not _CALLS_LOG.exists():
        return []
    buf: deque[dict] = deque(maxlen=n)
    with _CALLS_LOG.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    buf.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return list(buf)


def _fmt(val: float, unit: str = "") -> str:
    if unit == "ms":
        return f"{val * 1000:.0f} ms"
    if unit == "s":
        return f"{val:.3f} s"
    return str(val)


# ── refresh callback ──────────────────────────────────────────────────────────

def refresh():
    calls = _load_calls(500)

    if not calls:
        empty = "No data yet — run some TTS requests first."
        return (
            empty,          # summary_md
            [],             # recent_table
            [],             # latency_plot
            [],             # tokens_plot
            [],             # voice_table
        )

    total = len(calls)
    cache_hits = sum(1 for c in calls if c.get("cache_hit"))
    avg_llm   = sum(c["llm_s"]   for c in calls) / total
    avg_dec   = sum(c["decode_s"] for c in calls) / total
    avg_e2e   = sum(c["total_s"]  for c in calls) / total
    p95_e2e   = sorted(c["total_s"] for c in calls)[int(total * 0.95)]
    avg_tok   = sum(c["token_count"] for c in calls) / total
    total_wav_mb = sum(c["wav_bytes"] for c in calls) / 1e6

    summary_md = f"""
| Metric | Value |
|--------|-------|
| Requests (last 500) | **{total}** |
| Cache hits | **{cache_hits}** ({100*cache_hits/total:.1f}%) |
| Avg LLM latency | **{avg_llm*1000:.0f} ms** |
| Avg decode latency | **{avg_dec*1000:.0f} ms** |
| Avg end-to-end | **{avg_e2e*1000:.0f} ms** |
| p95 end-to-end | **{p95_e2e*1000:.0f} ms** |
| Avg tokens / req | **{avg_tok:.1f}** |
| Total audio generated | **{total_wav_mb:.2f} MB** |
"""

    # Recent calls table (newest first, last 50)
    recent = list(reversed(calls[-50:]))
    recent_table = [
        [
            c.get("ts", ""),
            c.get("voice_id") or "—",
            c.get("text", "")[:60],
            c.get("token_count", 0),
            f"{c['llm_s']*1000:.0f}",
            f"{c['decode_s']*1000:.0f}",
            f"{c['total_s']*1000:.0f}",
            "✓" if c.get("cache_hit") else "",
        ]
        for c in recent
    ]

    # Latency over time (last 200 calls) — plot data as list of [index, llm_ms, dec_ms, e2e_ms]
    window = calls[-200:]
    latency_plot = [
        [i, c["llm_s"] * 1000, c["decode_s"] * 1000, c["total_s"] * 1000]
        for i, c in enumerate(window)
    ]

    # Tokens over time (last 200 calls)
    tokens_plot = [
        [i, c["token_count"]]
        for i, c in enumerate(window)
    ]

    # Per-voice breakdown
    voice_stats: dict[str, dict] = {}
    for c in calls:
        v = c.get("voice_id") or "unknown"
        if v not in voice_stats:
            voice_stats[v] = {"count": 0, "llm_sum": 0.0, "e2e_sum": 0.0, "tok_sum": 0}
        s = voice_stats[v]
        s["count"] += 1
        s["llm_sum"] += c["llm_s"]
        s["e2e_sum"] += c["total_s"]
        s["tok_sum"] += c["token_count"]

    voice_table = [
        [
            v,
            s["count"],
            f"{s['llm_sum']/s['count']*1000:.0f} ms",
            f"{s['e2e_sum']/s['count']*1000:.0f} ms",
            f"{s['tok_sum']/s['count']:.1f}",
        ]
        for v, s in sorted(voice_stats.items(), key=lambda x: -x[1]["count"])
    ]

    return summary_md, recent_table, latency_plot, tokens_plot, voice_table


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="FlowTTS Metrics") as demo:
        gr.Markdown("# FlowTTS Live Metrics Dashboard")

        with gr.Row():
            refresh_btn = gr.Button("Refresh", variant="primary")
            auto_refresh = gr.Checkbox(label="Auto-refresh (5s)", value=False)

        summary_md = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Latency over time"):
                latency_plot = gr.LinePlot(
                    x="index",
                    y=["llm_ms", "dec_ms", "e2e_ms"],
                    label="Latency (ms)",
                    x_title="Request #",
                    y_title="ms",
                    height=350,
                )

            with gr.Tab("Tokens over time"):
                tokens_plot = gr.LinePlot(
                    x="index",
                    y=["tokens"],
                    label="Tokens per request",
                    x_title="Request #",
                    y_title="tokens",
                    height=350,
                )

            with gr.Tab("Per-voice breakdown"):
                voice_table = gr.Dataframe(
                    headers=["Voice", "Requests", "Avg LLM", "Avg E2E", "Avg Tokens"],
                    datatype=["str", "number", "str", "str", "str"],
                )

            with gr.Tab("Recent calls"):
                recent_table = gr.Dataframe(
                    headers=["Time", "Voice", "Text", "Tokens", "LLM ms", "Dec ms", "E2E ms", "Cache"],
                    datatype=["str", "str", "str", "number", "str", "str", "str", "str"],
                    wrap=True,
                )

        # wire refresh button
        outputs = [summary_md, recent_table, latency_plot, tokens_plot, voice_table]

        def _do_refresh():
            summary, recent, lat, tok, voice = refresh()
            import pandas as pd

            lat_df = pd.DataFrame(lat, columns=["index", "llm_ms", "dec_ms", "e2e_ms"]) if lat else pd.DataFrame(columns=["index", "llm_ms", "dec_ms", "e2e_ms"])
            tok_df = pd.DataFrame(tok, columns=["index", "tokens"]) if tok else pd.DataFrame(columns=["index", "tokens"])

            return summary, recent, lat_df, tok_df, voice

        refresh_btn.click(_do_refresh, outputs=outputs)

        # auto-refresh via timer
        timer = gr.Timer(value=5)
        timer.tick(_do_refresh, outputs=outputs)

        # trigger initial load on page load
        demo.load(_do_refresh, outputs=outputs)

    return demo


def launch(server_port: int = 7860, share: bool = False) -> None:
    demo = build_ui()
    demo.launch(server_port=server_port, share=share, show_api=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FlowTTS Gradio metrics dashboard")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    launch(server_port=args.port, share=args.share)
