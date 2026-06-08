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

# IPs loaded from ips.txt at startup (plain IPs, no port)
NODES: list[str] = []
NODE_PORTS: dict[str, int] = {}  # ip -> port override

# cache[ip] = deque of {"ts": unix_ms, "metrics": {name: [{labels, value}]}}
cache: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_SNAPSHOTS))

# ── Ext nginx config ─────────────────────────────────────────────────────────
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


# ── Background scraper ───────────────────────────────────────────────────────

async def scrape_node(session: ClientSession, ip: str):
    port = NODE_PORTS.get(ip, METRICS_PORT)
    url = f"http://{ip}:{port}{METRICS_PATH}"
    try:
        async with session.get(url, timeout=ClientTimeout(total=5)) as resp:
            text = await resp.text()
            snapshot = {
                "ts": int(time.time() * 1000),
                "metrics": parse_prometheus(text),
            }
            cache[ip].append(snapshot)
    except Exception as e:
        logging.warning(f"scrape {ip}: {e}")


async def scrape_loop():
    while True:
        async with ClientSession() as session:
            await asyncio.gather(*[scrape_node(session, ip) for ip in NODES])
        logging.info(f"scrape cycle done — {len(NODES)} nodes")
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
        ip = host.split(":")[0]
        port = NODE_PORTS.get(ip, METRICS_PORT)
        url = f"http://{ip}:{port}{METRICS_PATH}"
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

    for ip in NODES:
        snaps = [s for s in cache.get(ip, []) if s["ts"] >= cutoff]
        if len(snaps) < 2:
            # not enough data yet
            result[ip] = {"active": None, "ws": 0, "idle_since_ms": None}
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

        # Find when it last went idle — walk backwards to find last snapshot with new reqs
        idle_since = None
        if not active:
            for i in range(len(snaps) - 1, 0, -1):
                if named_reqs(snaps[i]["metrics"]) > named_reqs(snaps[i-1]["metrics"]):
                    idle_since = snaps[i]["ts"]
                    break
            if idle_since is None:
                idle_since = snaps[0]["ts"]  # idle for the whole window

        result[ip] = {
            "active":        active,
            "ws":            int(ws),
            "idle_since_ms": idle_since,
        }

    return web.Response(
        text=json.dumps(result),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def handle_cache_info(request: web.Request) -> web.Response:
    """Return how many snapshots are cached per node."""
    info = {ip: len(cache[ip]) for ip in NODES}
    oldest = {}
    for ip in NODES:
        if cache[ip]:
            oldest[ip] = cache[ip][0]["ts"]
    return web.Response(
        text=json.dumps({"snapshots": info, "oldest_ts": oldest, "max": MAX_SNAPSHOTS}),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


async def handle_index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(Path(__file__).parent.parent / "html" / "dashboard.html")


# ── App startup ──────────────────────────────────────────────────────────────

async def on_startup(app):
    # Load IPs
    global NODES, NODE_PORTS
    import sys
    ips_file = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent / "data" / "ips.txt")
    try:
        with open(ips_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    ip, port = line.rsplit(":", 1)
                    NODES.append(ip)
                    NODE_PORTS[ip] = int(port)
                else:
                    NODES.append(line)
        print(f"Loaded {len(NODES)} nodes from {ips_file}")
    except FileNotFoundError:
        print("ips.txt not found — no background scraping")

    # Start background scraper
    asyncio.create_task(scrape_loop())
    print(f"Scraping every {SCRAPE_INTERVAL_S}s, keeping up to {MAX_SNAPSHOTS} snapshots per node (~{MAX_SNAPSHOTS * SCRAPE_INTERVAL_S // 3600}h)")



app = web.Application()
app.on_startup.append(on_startup)
app.router.add_get("/proxy",       handle_proxy)
app.router.add_get("/history",     handle_history)
app.router.add_get("/activity",    handle_activity)
app.router.add_get("/cache-info",  handle_cache_info)
app.router.add_get("/ext-summary", handle_ext_summary)
app.router.add_get("/",           handle_index)
app.router.add_static("/",        str(Path(__file__).parent.parent / "html"))

if __name__ == "__main__":
    print(f"Dashboard at http://localhost:{SERVE_PORT}")
    web.run_app(app, port=SERVE_PORT, print=None)
