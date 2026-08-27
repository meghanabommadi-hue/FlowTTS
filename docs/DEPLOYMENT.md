# FlowTTS / OmniVoice — deployment on 101.53.141.123

Deployed 2026-08-26. Everything lives under `/root/omnivoice-svc`. Outside that
directory, exactly two things changed — both in nginx, both backed up first.

## What runs

```
/root/omnivoice-svc
├── flowtts/                 the service (this repo)
├── engines/
│   └── omnivoice-backbone/  TensorRT FP16 engine for the Qwen3 backbone (903 MB)
├── voices/                  24 voice-clone prompts (.npz)
├── cache/                   WAV cache, keyed by text + voice + generation params
├── omnivoice.env            all configuration, one file
├── start.sh / stop.sh       supervised launcher / precise stop
└── omnivoice.log
```

The Python environment is the **existing** `/home/jovyan/FlowTTS/omnivoice/.venv`,
which already had `omnivoice==0.2.1`, the NGC `torch 2.8.0a0+nv25.06` and
`tensorrt 10.11`. Seven small pure-Python packages were added to it with
`--no-deps` (`structlog`, `websockets`, `num2words`, `indic-num2words`,
`pydantic-settings`, `python-dotenv`, `docopt`); nothing already installed moved
version. The model is read from the shared HF cache — nothing was re-downloaded.

## Endpoints

Port 80 is the only externally reachable HTTP port, so these are what callers use:

```
http://101.53.141.123/tts/healthz                 liveness
http://101.53.141.123/tts/readyz                  readiness + voice list
http://101.53.141.123/tts/docs                    OpenAPI browser
http://101.53.141.123/tts/v1/tts                  synthesize
http://101.53.141.123/tts/v1/tts/stream           streaming synthesize (low TTFB)
http://101.53.141.123/tts/v1/audio/speech         OpenAI-compatible
http://101.53.141.123/tts/v1/voices               list / clone / delete voices
http://101.53.141.123/tts/v1/voices/preview       one-shot clone, nothing saved
http://101.53.141.123/tts/v1/languages            supported languages
http://101.53.141.123/tts/v1/normalize            preview the text preprocessor
http://101.53.141.123/tts/v1/stats                engine + latency counters
http://101.53.141.123/tts/metrics                 Prometheus
ws://101.53.141.123/tts/ws                        WebSocket (FlowTTS protocol)
```

`/dhvaani/` is kept as an alias onto the same upstream, so any integration still
calling the stopped service's path reaches a working TTS service instead of a 502.

On-box direct (bypasses nginx — use this for benchmarking):

```
http://127.0.0.1:9000     REST + WebSocket at /ws
http://127.0.0.1:9764     control API (healthz / readyz / stats / metrics / ports)
ws://127.0.0.1:9080,9081  raw-port WebSocket, for clients written against the old server
http://127.0.0.1:8090     the service's own nginx block (port not open externally)
```

## Operating it

```bash
cd /root/omnivoice-svc
nohup ./start.sh >/dev/null 2>&1 &   # supervised; restarts on non-zero exit
./stop.sh                            # stops ONLY this service
tail -f omnivoice.log
curl -s http://127.0.0.1:9764/stats | python3 -m json.tool
```

`start.sh` records its PID in `omnivoice.pid` and refuses to start a second
supervisor while one is alive. `stop.sh` kills the supervisor first — by PID —
then the service, and reports a non-zero exit if anything survives.

That ordering and that pidfile both exist because of a real failure here: the
original `stop.sh` matched the supervisor by command line, but the launcher runs
as `./start.sh`, so a pattern built from the absolute path never matched. Every
stop killed only the child, the loop restarted it ten seconds later, and each
subsequent start added another supervisor — **fifteen accumulated** before it
was noticed, all racing to restart the service.

Two self-match hazards are also guarded, both of which actually fired: the
service pattern is assembled from string fragments so the script's own command
line cannot contain it, and the fallback scan for pre-pidfile supervisors skips
this process and its ancestors and requires a candidate to *be* `bash …start.sh`
(exactly two arguments) rather than merely mention it. Without those, running
`./stop.sh` from inside `/root/omnivoice-svc` kills the shell that ran it.

Confirming it is really down (the supervisor retries after 10 s, so wait past
that before believing a clean result):

```bash
sleep 12
pgrep -af "flowtts[.]service"        # expect nothing
ss -ltn | grep -E ':(9000|9080|9081|9764) '   # expect nothing
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

Stopped, the service returns ~9.3 GiB: the card sits at ~15 GiB (the other
services and the training job) against ~24.3 GiB while it runs.

Change the latency profile in `start.sh` (`PROFILE=fast|balanced|quality`) or
override any single setting in `omnivoice.env`.

## What changed outside /root/omnivoice-svc

1. `/etc/nginx/conf.d/dhvaani.conf` → replaced by `omnivoice.conf`. That file
   was a server block on :8090 pointing at ports 9000/9080/9081 for the DhVaani
   service, which is now stopped; those ports are this service's. Leaving it
   would have served OmniVoice under the DhVaani name.
2. `/etc/nginx/sites-enabled/llm` → the two dead `/dhvaani/` location blocks
   replaced with `/tts/` blocks (plus `/dhvaani/` aliases). No existing
   directive belonging to Gemma (`/v1/`, `/health`) or transliteration
   (`/xlit/`) was touched.

Backups, and how to revert:

```bash
ls /root/omnivoice-svc/nginx-backup/
#   nginx-full-<ts>.tgz        the whole /etc/nginx tree
#   llm.orig.<ts>              just that file
#   dhvaani.conf.orig.<ts>

cp /root/omnivoice-svc/nginx-backup/llm.orig.<ts> /etc/nginx/sites-enabled/llm
cp /root/omnivoice-svc/nginx-backup/dhvaani.conf.orig.<ts> /etc/nginx/conf.d/dhvaani.conf
rm /etc/nginx/conf.d/omnivoice.conf
nginx -t && nginx -s reload
```

## What was stopped

`dhvaani-svc` only, via its own `/root/dhvaani-svc/stop.sh`. Its files are
untouched and it can be restarted — but note its ports (9000/9080/9081/9764) are
now this service's, so one of the two has to be re-pointed first.

Left running, untouched: the Gemma server, `transliteration.api` (:8082), the
two `transliteration.cli train` jobs, the older `flowtts.server` (:8080), the
OmniVoice demo `model.py` (:8081), Jupyter and nginx itself.

## Measured performance

NVIDIA L40S **shared with a multi-day `transliteration.cli train` job** holding
~15 GiB and bursting across the CPUs. A dedicated card would do materially better.

TensorRT FP16 backbone, streaming, Hindi voice-bot text mix, **cold WAV cache**,
`num_step=4`.

**Sustained offered load** — the measurement that sizes a deployment, because a
voice bot's open sockets are idle between turns. The load generator sends at a
fixed rate without waiting for responses, so a queue forms only if the server is
genuinely behind.

| requests/sec in | TTFB p50 | TTFB p90 | TTFB p99 | realtime | failures |
|---|---|---|---|---|---|
| 1 | 87 ms | 114 ms | 165 ms | 4.5x | 0 |
| 2 | 80 ms | 103 ms | 192 ms | 8.9x | 0 |
| 4 | 102 ms | 157 ms | 163 ms | 17.9x | 0 |
| 6 | 98 ms | 126 ms | 175 ms | 27.4x | 0 |
| 8 | 105 ms | 146 ms | 176 ms | 36.8x | 0 |
| 10 | 96 ms | 162 ms | 181 ms | 45.6x | 0 |
| 12 | 125 ms | 190 ms | 272 ms | 54.7x | 0 |
| 14 | 257 ms | 435 ms | 523 ms | 63.3x | 0 |

**p99 TTFB stays under 200 ms all the way to 10 requests/second.** The knee is
between 10 and 12.

**Fixed concurrency** — all N requests issued at once, 96 per level:

| concurrent | TTFB p50 | TTFB p99 | rps | realtime | failures |
|---|---|---|---|---|---|
| 1 | 101 ms | 187 ms | 9.7 | 44x | 0 |
| 4 | 286 ms | 479 ms | 14.4 | 66x | 0 |
| 8 | 574 ms | 874 ms | 13.9 | 63x | 0 |
| 16 | 899 ms | 1511 ms | 17.4 | 79x | 0 |
| 32 | 1798 ms | 3502 ms | 16.6 | 76x | 0 |
| 64 | 2817 ms | 4785 ms | 18.1 | 82x | 0 |

Zero failures at every level; the GPU saturates at ~18 rps (82x realtime).

### What batching is worth here, measured

`flowtts.test.bench_batching` on this box, `num_step=4`, uniform-length inputs:

| batch | total | per item | vs sequential |
|---|---|---|---|
| 1 | 61.2 ms | 61.2 ms | 1.00x |
| 2 | 97.6 ms | 48.8 ms | 1.25x |
| 4 | 196.5 ms | 49.1 ms | 1.25x |
| 8 | 426.4 ms | 53.3 ms | 1.15x |
| 16 | 765.4 ms | 47.8 ms | 1.28x |
| 24 | 1255.6 ms | 52.3 ms | 1.17x |

Batching is worth about 1.2x and the curve is **flat from batch 2 onward** —
unlike an LLM server, where it is the dominant throughput lever. The reason is
structural: OmniVoice's `_generate_iterative` runs a per-item Python loop inside
every denoise step (top-k, gumbel, masked_fill, copy_ per batch element) and
materializes float32 logits of `[2B, 8, S, 1025]`. Both scale with batch size
and neither is batched work.

Mixed lengths are worse than not batching at all, because `generate()` pads
every item to the longest:

| | |
|---|---|
| short alone | 41.1 ms |
| long alone | 83.2 ms |
| run separately | 124.3 ms |
| **batched together** | **156.7 ms** — 0.79x |
| 8 mixed, separately | 452.2 ms |
| **8 mixed, one batch** | **682.0 ms** — 0.66x |

So `max_batch` is deliberately 8 rather than 32 (past batch 4 there is nothing
left to gain and a batch of 24 blocks the GPU for 1.26 s), and
`length_bucket_ratio` is deliberately tight at 1.5. Both are set from this
measurement, not from intuition — rerun it after any model change.

### On the "TTFB < 200 ms at 100 concurrent" target

Not reachable on this hardware, and not by tuning. OmniVoice is
non-autoregressive: each of the `num_step` denoise passes re-runs the whole
28-layer backbone over the entire sequence, for the conditional and
unconditional halves both. 100 simultaneous first-chunks is tens of TFLOP of
work; the card sustains ~18 requests/second, so a 100-deep queue is still seconds deep
no matter how the queue is arranged.

What the numbers above do say:

* **100 concurrent voice-bot *sessions* are fine** if their combined turn rate
  stays at or under ~10 turns/second — i.e. each session speaking about once
  every 10 s. That is a normal IVR cadence, and p99 TTFB there is 181 ms.
* Sustained rates above ~10 rps need a second GPU. Throughput scales linearly
  with cards; the service is stateless apart from the voice registry, so two
  instances behind the existing nginx upstream is the direct path.
* The WAV cache serves repeated prompts in ~1 ms without touching the GPU, which
  for a call-centre script is a large fraction of real traffic.

### Backbone acceleration

| backbone | TTFB p50 @ conc 1 | peak rps | vs PyTorch |
|---|---|---|---|
| PyTorch | 146 ms | 6.95 | 1.00x |
| TensorRT FP16 | 73 ms | 10.04 | **2.0x TTFB, 1.44x throughput** |

(Measured back to back on the same box at `num_step=4`, before the
`min_chunk_seconds` change, so the two rows are comparable to each other rather
than to the tables above.)

The 1.39–1.44x throughput figure matches what upstream
(github.com/tlitech/omnivoice-trtllm) reports for FP16 TRT-LLM on an L4.

Rebuild the engine after a model change:

```bash
cd /root/omnivoice-svc && set -a; . omnivoice.env; set +a
/home/jovyan/FlowTTS/omnivoice/.venv/bin/python -m flowtts.trt.build_trt \
  --engine-dir engines/omnivoice-backbone --precision fp16 \
  --max-batch 64 --opt-seq 384 --max-seq 2048
./stop.sh; ./start.sh &
```

The service validates any engine against the real PyTorch module at startup
(cosine ≥ 0.99) and refuses to install one that does not match, falling back to
PyTorch instead. Current engine: **cosine 0.999998**.

## Verifying a deployment

```bash
cd /root/omnivoice-svc && export PYTHONPATH=/root/omnivoice-svc
V=/home/jovyan/FlowTTS/omnivoice/.venv/bin/python

$V -m flowtts.test.verify_deployment --url http://127.0.0.1:9000   # 38 checks
$V -m flowtts.test.bench --rate 1,2,4,6,8 --duration 15            # offered load
$V -m flowtts.test.bench --sweep 1,4,16,64,100 --requests 200      # concurrency
$V -m flowtts.test.diagnose_backbone                               # engine vs PyTorch
```

Last run: **38/38 passed**, including all 22 scheduled languages of India.

## Environment notes

This is an NVIDIA NGC container: PID 1 is not systemd, so there is no unit file —
`start.sh` is a supervised background process, matching how the other services on
this box run. A global `PIP_CONSTRAINT` pins torch to the container build, which
is why TensorRT-LLM is not installed here and the TensorRT path
(`flowtts.trt.build_trt`) is used instead; it produces an engine with the same
I/O contract from the TensorRT that ships with the container. See
`flowtts/trt/build_trt.py` for the full reasoning.
