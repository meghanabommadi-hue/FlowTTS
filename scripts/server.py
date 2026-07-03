#!/usr/bin/env python3
"""
Local proxy server for TTS GPU Monitor dashboard.
- Serves dashboard.html and node.html
- Proxies /metrics requests to GPU nodes (bypasses browser CORS)
- Scrapes all nodes every 15s and stores snapshots in an in-memory cache
- Exposes /history?host=<ip>&since=<unix_ms> for time-range queries

Usage:
    python3 server.py
Then open: http://localhost:8080
"""

import asyncio
import time
import json
import logging
import ssl
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))
from pathlib import Path
from aiohttp import web, ClientSession, ClientTimeout

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

METRICS_PORT = 8764
METRICS_PATH = "/metrics"
SERVE_PORT   = 8080
SCRAPE_INTERVAL_S = 15
MAX_SNAPSHOTS     = 1440  # ~6 hours at 15s interval per node
PERSIST_INTERVAL  = 10    # write to disk every N scrapes (~2.5 min)
DATA_DIR          = Path(__file__).parent.parent / "data" / "history"

# NODES stores "ip:port" keys — allows same IP on multiple ports
NODES: list[str] = []       # main cluster
EXT2_NODES: list[str] = []  # secondary cluster (legacy)
MULTI_NODES: list[str] = [] # multilingual cluster

# cache["ip:port"] = deque of {"ts": unix_ms, "metrics": {name: [{labels, value}]}}
cache: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_SNAPSHOTS))

# daily_baseline["ip:port"] = {"date": "YYYY-MM-DD", "metrics": {...}} — first snapshot of the day
daily_baseline: dict[str, dict] = {}

# scrape counter per node for persist throttling
scrape_count: dict[str, int] = defaultdict(int)

# concurrent request history: deque of {ts, bucketx_avg, multi_avg}
MAX_CONCURRENT_HISTORY = 2880  # ~12 hours at 15s
concurrent_history: deque = deque(maxlen=MAX_CONCURRENT_HISTORY)

# ── Ext nginx config (Bajaj FlowTTS) ─────────────────────────────────────────
EXT_METRICS_URL = "https://models.kapturecrm.com/bajaj-flowtts/metrics"
EXT_NODE_COUNT  = 7


# ── Prometheus parser ────────────────────────────────────────────────────────

def parse_prometheus(text: str) -> dict:
    result: dict[str, list] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        name_labels, value_str = parts
        try:
            value = float(value_str)
        except ValueError:
            continue
        name = name_labels.split("{")[0]
        labels: dict[str, str] = {}
        lb_start = name_labels.find("{")
        if lb_start != -1:
            lb_str = name_labels[lb_start + 1:name_labels.rfind("}")]
            for pair in lb_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')
        if name not in result:
            result[name] = []
        result[name].append({"labels": labels, "value": value})
    return result


# ── Persistence helpers ───────────────────────────────────────────────────────

def node_history_file(node: str) -> Path:
    safe = node.replace(":", "_").replace("/", "_")
    return DATA_DIR / f"{safe}.jsonl"

def persist_snapshot(node: str, snapshot: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    m = snapshot["metrics"]
    def named_sum(name):
        return sum(e["value"] for e in m.get(name, []) if e["labels"].get("voice", ""))
    reqs = named_sum("tts_requests_total")
    record = {
        "ts":         snapshot["ts"],
        "e2e_ms":     round(named_sum("tts_e2e_ms_total") / reqs, 2) if reqs else 0,
        "decode_ms":  round(named_sum("tts_decode_ms_total") / reqs, 2) if reqs else 0,
        "cache_hits": sum(e["value"] for e in m.get("tts_cache_hits_total", [])),
        "reqs":       reqs,
    }
    with open(node_history_file(node), "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Background scraper ───────────────────────────────────────────────────────

async def scrape_node(session: ClientSession, node: str):
    url = f"http://{node}{METRICS_PATH}"
    try:
        async with session.get(url, timeout=ClientTimeout(total=5)) as resp:
            text = await resp.text()
            metrics = parse_prometheus(text)
            snapshot = {
                "ts": int(time.time() * 1000),
                "metrics": metrics,
            }
            cache[node].append(snapshot)

            today = datetime.now(IST).strftime("%Y-%m-%d")
            if node not in daily_baseline or daily_baseline[node]["date"] != today:
                daily_baseline[node] = {"date": today, "metrics": metrics}

            scrape_count[node] += 1
            if scrape_count[node] % PERSIST_INTERVAL == 0:
                persist_snapshot(node, snapshot)
    except Exception as e:
        logging.warning(f"scrape {node}: {e}")


def cluster_avg_concurrent(nodes: list) -> float:
    vals = []
    for node in nodes:
        snaps = list(cache.get(node, []))
        if not snaps:
            continue
        m = snaps[-1]["metrics"]
        running = sum(e["value"] for e in m.get("vllm:num_requests_running", []))
        vals.append(running)
    return round(sum(vals) / len(vals), 2) if vals else 0


async def scrape_loop():
    while True:
        all_nodes = NODES + EXT2_NODES + MULTI_NODES
        async with ClientSession() as session:
            await asyncio.gather(*[scrape_node(session, node) for node in all_nodes])
        logging.info(f"scrape cycle done — {len(NODES)} main + {len(EXT2_NODES)} ext2 + {len(MULTI_NODES)} multilingual nodes")
        concurrent_history.append({
            "ts":         int(time.time() * 1000),
            "bucketx":    cluster_avg_concurrent(NODES),
            "multi":      cluster_avg_concurrent(MULTI_NODES),
        })
        await asyncio.sleep(SCRAPE_INTERVAL_S)


# ── HTTP handlers ────────────────────────────────────────────────────────────

async def handle_proxy(request: web.Request) -> web.Response:
    """Proxy a single /metrics fetch (used by dashboard on-demand)."""
    host = request.query.get("host")
    if not host:
        return web.Response(status=400, text="Missing ?host=")
    if host.startswith("http://") or host.startswith("https://"):
        url = host
    else:
        if ":" not in host:
            host = f"{host}:{METRICS_PORT}"
        url = f"http://{host}{METRICS_PATH}"
    try:
        async with ClientSession() as session:
            async with session.get(url, timeout=ClientTimeout(total=5), ssl=SSL_CTX) as resp:
                text = await resp.text()
                return web.Response(text=text, content_type="text/plain",
                                    headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        return web.Response(status=502, text=str(e),
                            headers={"Access-Control-Allow-Origin": "*"})


async def handle_ext_summary(request: web.Request) -> web.Response:
    """Fetch metrics from nginx and return parsed summary."""
    try:
        async with ClientSession() as session:
            async with session.get(EXT_METRICS_URL, timeout=ClientTimeout(total=8), ssl=SSL_CTX) as resp:
                text = await resp.text()
        m = parse_prometheus(text)
        ws    = sum(e["value"] for e in m.get("tts_active_websockets", []))
        reqs  = sum(e["value"] for e in m.get("tts_requests_total", []))
        hits  = sum(e["value"] for e in m.get("tts_cache_hits_total", []))
        misses = sum(e["value"] for e in m.get("tts_cache_misses_total", []))
        e2e_ms = sum(e["value"] for e in m.get("tts_e2e_ms_total", []))
        named_reqs = sum(e["value"] for e in m.get("tts_requests_total", []) if e["labels"].get("voice", ""))
        avg_e2e = (e2e_ms / named_reqs) if named_reqs > 0 else 0
        cache_total = hits + misses
        return web.Response(
            text=json.dumps({
                "up": 1,
                "total": EXT_NODE_COUNT,
                "total_ws": int(ws),
                "avg_ws_per_node": round(ws / EXT_NODE_COUNT, 1),
                "total_reqs": int(reqs),
                "cache_hits": int(hits),
                "cache_misses": int(misses),
                "cache_rate": round(hits / cache_total * 100, 1) if cache_total else 0,
                "avg_e2e_ms": round(avg_e2e, 1),
            }),
            content_type="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )
    except Exception as e:
        return web.Response(
            text=json.dumps({"up": 0, "total": EXT_NODE_COUNT, "error": str(e)}),
            content_type="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )


async def handle_ext2_summary(request: web.Request) -> web.Response:
    """Return aggregated metrics for the ext2 (47.29.24.x) cluster from cache."""
    nodes = EXT2_NODES
    if not nodes:
        return web.Response(
            text=json.dumps({"nodes": [], "total": 0, "up": 0}),
            content_type="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    result = []
    for node in nodes:
        snaps = list(cache.get(node, []))
        if not snaps:
            result.append({"node": node, "up": False})
            continue
        m = snaps[-1]["metrics"]
        ws      = sum(e["value"] for e in m.get("tts_active_websockets", []))
        reqs    = sum(e["value"] for e in m.get("tts_requests_total", []))
        hits    = sum(e["value"] for e in m.get("tts_cache_hits_total", []))
        misses  = sum(e["value"] for e in m.get("tts_cache_misses_total", []))
        named_reqs = sum(e["value"] for e in m.get("tts_requests_total", []) if e["labels"].get("voice", ""))
        e2e_ms  = sum(e["value"] for e in m.get("tts_e2e_ms_total", []) if e["labels"].get("voice", ""))
        avg_e2e = (e2e_ms / named_reqs) if named_reqs > 0 else 0
        cache_total = hits + misses
        result.append({
            "node":       node,
            "up":         True,
            "ws":         int(ws),
            "reqs":       int(reqs),
            "cache_hits": int(hits),
            "cache_misses": int(misses),
            "cache_rate": round(hits / cache_total * 100, 1) if cache_total else 0,
            "avg_e2e_ms": round(avg_e2e, 1),
        })

    up_nodes       = [n for n in result if n.get("up")]
    active_nodes   = [n for n in up_nodes if n["ws"] > 0]
    total_ws       = sum(n["ws"] for n in up_nodes)
    total_reqs     = sum(n["reqs"] for n in up_nodes)
    total_hits     = sum(n["cache_hits"] for n in up_nodes)
    total_miss     = sum(n["cache_misses"] for n in up_nodes)
    cache_total    = total_hits + total_miss
    e2e_vals       = [n["avg_e2e_ms"] for n in up_nodes if n["avg_e2e_ms"] > 0]
    avg_e2e        = round(sum(e2e_vals) / len(e2e_vals), 1) if e2e_vals else 0
    max_e2e        = round(max(e2e_vals), 1) if e2e_vals else 0
    avg_ws_active  = round(total_ws / len(active_nodes), 1) if active_nodes else 0

    return web.Response(
        text=json.dumps({
            "nodes":             result,
            "total":             len(nodes),
            "up":                len(up_nodes),
            "total_ws":          total_ws,
            "avg_ws_per_active": avg_ws_active,
            "total_reqs":        total_reqs,
            "cache_hits":        total_hits,
            "cache_misses":      total_miss,
            "cache_rate":        round(total_hits / cache_total * 100, 1) if cache_total else 0,
            "avg_e2e_ms":        avg_e2e,
            "max_e2e_ms":        max_e2e,
        }),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def handle_multi_summary(request: web.Request) -> web.Response:
    """Return aggregated metrics for the multilingual cluster from cache."""
    nodes = MULTI_NODES
    if not nodes:
        return web.Response(
            text=json.dumps({"nodes": [], "total": 0, "up": 0}),
            content_type="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    result = []
    for node in nodes:
        snaps = list(cache.get(node, []))
        if not snaps:
            result.append({"node": node, "up": False})
            continue
        m = snaps[-1]["metrics"]
        ws      = sum(e["value"] for e in m.get("tts_active_websockets", []))
        reqs    = sum(e["value"] for e in m.get("tts_requests_total", []))
        hits    = sum(e["value"] for e in m.get("tts_cache_hits_total", []))
        misses  = sum(e["value"] for e in m.get("tts_cache_misses_total", []))
        named_reqs = sum(e["value"] for e in m.get("tts_requests_total", []) if e["labels"].get("voice", ""))
        e2e_ms  = sum(e["value"] for e in m.get("tts_e2e_ms_total", []) if e["labels"].get("voice", ""))
        avg_e2e = (e2e_ms / named_reqs) if named_reqs > 0 else 0
        cache_total = hits + misses
        result.append({
            "node":       node,
            "up":         True,
            "ws":         int(ws),
            "reqs":       int(reqs),
            "cache_hits": int(hits),
            "cache_misses": int(misses),
            "cache_rate": round(hits / cache_total * 100, 1) if cache_total else 0,
            "avg_e2e_ms": round(avg_e2e, 1),
        })

    up_nodes     = [n for n in result if n.get("up")]
    active_nodes = [n for n in up_nodes if n["ws"] > 0]
    total_ws     = sum(n["ws"] for n in up_nodes)
    total_reqs   = sum(n["reqs"] for n in up_nodes)
    total_hits   = sum(n["cache_hits"] for n in up_nodes)
    total_miss   = sum(n["cache_misses"] for n in up_nodes)
    cache_total  = total_hits + total_miss
    e2e_vals     = [n["avg_e2e_ms"] for n in up_nodes if n["avg_e2e_ms"] > 0]
    avg_e2e      = round(sum(e2e_vals) / len(e2e_vals), 1) if e2e_vals else 0
    max_e2e      = round(max(e2e_vals), 1) if e2e_vals else 0
    avg_ws_active = round(total_ws / len(active_nodes), 1) if active_nodes else 0

    return web.Response(
        text=json.dumps({
            "nodes":             result,
            "total":             len(nodes),
            "up":                len(up_nodes),
            "total_ws":          total_ws,
            "avg_ws_per_active": avg_ws_active,
            "total_reqs":        total_reqs,
            "cache_hits":        total_hits,
            "cache_misses":      total_miss,
            "cache_rate":        round(total_hits / cache_total * 100, 1) if cache_total else 0,
            "avg_e2e_ms":        avg_e2e,
            "max_e2e_ms":        max_e2e,
        }),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def handle_history(request: web.Request) -> web.Response:
    """
    Return cached snapshots for a node, optionally filtered by time.

    Query params:
      host=<ip>           required
      since=<unix_ms>     optional — only return snapshots after this timestamp
      last=<N>            optional — return only the last N snapshots
    """
    host = request.query.get("host")
    if not host:
        return web.Response(status=400, text="Missing ?host=")

    since   = int(request.query.get("since", 0))
    last_n  = int(request.query.get("last", 0))

    snapshots = list(cache.get(host, []))

    if since:
        snapshots = [s for s in snapshots if s["ts"] >= since]
    if last_n:
        snapshots = snapshots[-last_n:]

    return web.Response(
        text=json.dumps(snapshots),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def handle_activity(request: web.Request) -> web.Response:
    """
    For each node, check if tts_requests_total (named voices) changed
    in the last 5 minutes of cached snapshots.
    Returns: { ip: { active: bool, ws: int, idle_since_ms: int|null } }
    """
    window_ms = int(request.query.get("window_ms", 5 * 60 * 1000))
    now       = int(time.time() * 1000)
    cutoff    = now - window_ms
    result    = {}

    for node in NODES:
        snaps = [s for s in cache.get(node, []) if s["ts"] >= cutoff]
        if len(snaps) < 2:
            result[node] = {"active": None, "ws": 0, "idle_since_ms": None}
            continue

        def named_reqs(metrics):
            return sum(
                e["value"] for e in metrics.get("tts_requests_total", [])
                if e["labels"].get("voice", "") != ""
            )

        first_reqs = named_reqs(snaps[0]["metrics"])
        last_reqs  = named_reqs(snaps[-1]["metrics"])
        ws         = sum(e["value"] for e in snaps[-1]["metrics"].get("tts_active_websockets", []))
        active     = last_reqs > first_reqs

        idle_since = None
        if not active:
            for i in range(len(snaps) - 1, 0, -1):
                if named_reqs(snaps[i]["metrics"]) > named_reqs(snaps[i-1]["metrics"]):
                    idle_since = snaps[i]["ts"]
                    break
            if idle_since is None:
                idle_since = snaps[0]["ts"]

        result[node] = {
            "active":        active,
            "ws":            int(ws),
            "idle_since_ms": idle_since,
        }

    return web.Response(
        text=json.dumps(result),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def handle_daily(request: web.Request) -> web.Response:
    """Return today's cumulative deltas (current - start-of-day baseline) across all nodes."""
    all_nodes = NODES + EXT2_NODES + MULTI_NODES
    today = datetime.now(IST).strftime("%Y-%m-%d")

    def node_metric_sum(metrics, name):
        return sum(e["value"] for e in metrics.get(name, []))

    total_reqs = total_hits = total_misses = 0
    total_e2e_ms = total_named_reqs = 0
    nodes_with_data = 0

    def named_sum(metrics, name):
        return sum(e["value"] for e in metrics.get(name, []) if e["labels"].get("voice", ""))

    for node in all_nodes:
        snaps = list(cache.get(node, []))
        if not snaps:
            continue
        baseline = daily_baseline.get(node)
        if not baseline or baseline["date"] != today:
            continue

        cur = snaps[-1]["metrics"]
        base = baseline["metrics"]

        delta_reqs   = max(0, node_metric_sum(cur, "tts_requests_total")     - node_metric_sum(base, "tts_requests_total"))
        delta_hits   = max(0, node_metric_sum(cur, "tts_cache_hits_total")   - node_metric_sum(base, "tts_cache_hits_total"))
        delta_misses = max(0, node_metric_sum(cur, "tts_cache_misses_total") - node_metric_sum(base, "tts_cache_misses_total"))
        delta_e2e    = max(0, named_sum(cur, "tts_e2e_ms_total")             - named_sum(base, "tts_e2e_ms_total"))
        delta_named  = max(0, named_sum(cur, "tts_requests_total")           - named_sum(base, "tts_requests_total"))

        total_reqs        += delta_reqs
        total_hits        += delta_hits
        total_misses      += delta_misses
        total_e2e_ms      += delta_e2e
        total_named_reqs  += delta_named
        nodes_with_data   += 1

    cache_total = total_hits + total_misses
    avg_e2e_ms  = round(total_e2e_ms / total_named_reqs, 1) if total_named_reqs else 0
    return web.Response(
        text=json.dumps({
            "date":          today,
            "nodes":         nodes_with_data,
            "total_reqs":    int(total_reqs),
            "cache_hits":    int(total_hits),
            "cache_misses":  int(total_misses),
            "cache_rate":    round(total_hits / cache_total * 100, 1) if cache_total else 0,
            "avg_e2e_ms":    avg_e2e_ms,
        }),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def handle_history_bucketed(request: web.Request) -> web.Response:
    """
    Read persisted per-node history files and return bucketed averages.
    Query params:
      bucket_h=5   bucket size in hours (default 5)
    Returns: { node: [ {label, e2e_ms, decode_ms, cache_hits, reqs}, ... ] }
    """
    bucket_h  = float(request.query.get("bucket_h", 5))
    bucket_ms = int(bucket_h * 3600 * 1000)
    all_nodes = NODES + EXT2_NODES + MULTI_NODES
    result: dict[str, list] = {}

    for node in all_nodes:
        path = node_history_file(node)
        if not path.exists():
            continue
        buckets: dict[int, dict] = {}
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    key = r["ts"] // bucket_ms
                    if key not in buckets:
                        buckets[key] = {"e2e_sum": 0, "decode_sum": 0, "cache_hits_sum": 0, "reqs_sum": 0, "count": 0, "ts_start": r["ts"]}
                    b = buckets[key]
                    if r["reqs"] > 0:
                        b["e2e_sum"]    += r["e2e_ms"] * r["reqs"]
                        b["decode_sum"] += r["decode_ms"] * r["reqs"]
                        b["reqs_sum"]   += r["reqs"]
                    b["cache_hits_sum"] += r["cache_hits"]
                    b["count"]          += 1
        except Exception as e:
            logging.warning(f"history read {node}: {e}")
            continue

        points = []
        for key in sorted(buckets):
            b = buckets[key]
            ts = b["ts_start"]
            label = datetime.fromtimestamp(ts / 1000, tz=IST).strftime("%d %b %H:%M")
            points.append({
                "label":      label,
                "e2e_ms":     round(b["e2e_sum"] / b["reqs_sum"], 1) if b["reqs_sum"] else 0,
                "decode_ms":  round(b["decode_sum"] / b["reqs_sum"], 1) if b["reqs_sum"] else 0,
                "cache_hits": int(b["cache_hits_sum"] / b["count"]) if b["count"] else 0,
            })
        result[node] = points

    return web.Response(
        text=json.dumps(result),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def handle_concurrent_history(request: web.Request) -> web.Response:
    """Return time-series of avg concurrent requests per cluster."""
    since = int(request.query.get("since", 0))
    snaps = [s for s in concurrent_history if s["ts"] >= since] if since else list(concurrent_history)
    return web.Response(
        text=json.dumps(snaps),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def handle_hourly_e2e(request: web.Request) -> web.Response:
    """Return avg E2E latency over the last 1 hour using cached snapshot deltas."""
    all_nodes = NODES + EXT2_NODES + MULTI_NODES
    window_ms = int(request.query.get("window_ms", 3600 * 1000))
    cutoff = int(time.time() * 1000) - window_ms

    def named_sum(metrics, name):
        return sum(e["value"] for e in metrics.get(name, []) if e["labels"].get("voice", ""))

    total_e2e_ms = 0
    total_named_reqs = 0

    for node in all_nodes:
        snaps = [s for s in cache.get(node, []) if s["ts"] >= cutoff]
        if len(snaps) < 2:
            continue
        first, last = snaps[0]["metrics"], snaps[-1]["metrics"]
        delta_e2e   = max(0, named_sum(last, "tts_e2e_ms_total")   - named_sum(first, "tts_e2e_ms_total"))
        delta_reqs  = max(0, named_sum(last, "tts_requests_total") - named_sum(first, "tts_requests_total"))
        total_e2e_ms     += delta_e2e
        total_named_reqs += delta_reqs

    avg_e2e = round(total_e2e_ms / total_named_reqs, 1) if total_named_reqs else 0
    return web.Response(
        text=json.dumps({
            "avg_e2e_ms":   avg_e2e,
            "total_reqs":   int(total_named_reqs),
            "window_min":   window_ms // 60000,
        }),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def handle_cache_info(request: web.Request) -> web.Response:
    """Return how many snapshots are cached per node."""
    info = {node: len(cache[node]) for node in NODES}
    oldest = {}
    for node in NODES:
        if cache[node]:
            oldest[node] = cache[node][0]["ts"]
    return web.Response(
        text=json.dumps({"snapshots": info, "oldest_ts": oldest, "max": MAX_SNAPSHOTS}),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def handle_index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(Path(__file__).parent.parent / "html" / "dashboard.html")


# ── App startup ──────────────────────────────────────────────────────────────

async def on_startup(app):
    global NODES, EXT2_NODES
    import sys
    ips_file = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent / "data" / "ips.txt")
    try:
        current = NODES
        with open(ips_file) as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                if raw == "# cluster:ext2":
                    current = EXT2_NODES
                    continue
                if raw == "# cluster:multilingual":
                    current = MULTI_NODES
                    continue
                if raw.startswith("#"):
                    continue
                if ":" not in raw:
                    raw = f"{raw}:{METRICS_PORT}"
                current.append(raw)
        print(f"Loaded {len(NODES)} main + {len(EXT2_NODES)} ext2 + {len(MULTI_NODES)} multilingual nodes from {ips_file}")
    except FileNotFoundError:
        print("ips.txt not found — no background scraping")

    # Start background scraper
    asyncio.create_task(scrape_loop())
    print(f"Scraping every {SCRAPE_INTERVAL_S}s, keeping up to {MAX_SNAPSHOTS} snapshots per node (~{MAX_SNAPSHOTS * SCRAPE_INTERVAL_S // 3600}h)")



app = web.Application()
app.on_startup.append(on_startup)
app.router.add_get("/proxy",        handle_proxy)
app.router.add_get("/history",      handle_history)
app.router.add_get("/activity",     handle_activity)
app.router.add_get("/cache-info",   handle_cache_info)
app.router.add_get("/daily",             handle_daily)
app.router.add_get("/hourly-e2e",        handle_hourly_e2e)
app.router.add_get("/history-bucketed",      handle_history_bucketed)
app.router.add_get("/concurrent-history",    handle_concurrent_history)
app.router.add_get("/ext-summary",  handle_ext_summary)
app.router.add_get("/ext2-summary",  handle_ext2_summary)
app.router.add_get("/multi-summary", handle_multi_summary)
app.router.add_get("/",            handle_index)
app.router.add_static("/",        str(Path(__file__).parent.parent / "html"))

if __name__ == "__main__":
    print(f"Dashboard at http://localhost:{SERVE_PORT}")
    web.run_app(app, port=SERVE_PORT, print=None)
