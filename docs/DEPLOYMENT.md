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
./start.sh &                 # supervised; restarts on non-zero exit
./stop.sh                    # stops ONLY this service, never a broad pkill
tail -f omnivoice.log
curl -s http://127.0.0.1:9764/stats | python3 -m json.tool
```

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

TensorRT FP16 backbone, streaming, Hindi voice-bot text mix, **cold WAV cache**.

**Sustained offered load** — the measurement that sizes a deployment, because a
voice bot's open sockets are idle between turns. The load generator sends at a
fixed rate without waiting for responses, so a queue forms only if the server is
genuinely behind.

`num_step=4` (the `fast` profile):

| requests/sec in | TTFB p50 | TTFB p90 | TTFB p99 | realtime | failures |
|---|---|---|---|---|---|
| 1 | 85 ms | 181 ms | 209 ms | 4.8x | 0 |
| 2 | 124 ms | 186 ms | 209 ms | 9.6x | 0 |
| 4 | 120 ms | 180 ms | 273 ms | 19.3x | 0 |
| 6 | 140 ms | 207 ms | 304 ms | 29.5x | 0 |
| 8 | 251 ms | 431 ms | 519 ms | 39.1x | 0 |

`num_step=8` (the `balanced` profile, the default):

| requests/sec in | TTFB p50 | TTFB p90 | TTFB p99 | failures |
|---|---|---|---|---|
| 1 | 205 ms | 276 ms | 302 ms | 0 |
| 2 | 198 ms | 303 ms | 307 ms | 0 |
| 3 | 201 ms | 304 ms | 364 ms | 0 |
| 4 | 246 ms | 402 ms | 579 ms | 0 |

So: **median TTFB stays under 150 ms up to 6 requests/second at `num_step=4`**,
with the knee just past 7. At `num_step=8` the median sits around 200 ms — the
quality/latency trade is roughly 2x either way, and `num_step` is the only dial
that moves it (`guidance_scale` costs nothing on this model — see below).

These numbers are ~40-60 ms worse than an earlier revision, deliberately.
`min_chunk_seconds` was raised from 0.5 to 1.0 after measuring that OmniVoice
returns outright **silence** on targets below ~1 s of audio — reproducibly, in
Hindi and Santali alike, roughly two runs in three at 34 frames while the same
sentence at 125 frames was stable every time. A slightly later first byte is
worth not shipping silent audio to a live call.

**Fixed concurrency** — all N requests issued simultaneously, 200 per level:

| concurrent | TTFB p50 | TTFB p99 | rps | realtime | failures |
|---|---|---|---|---|---|
| 1 | 100 ms | 175 ms | 5.4 | 28x | 0 |
| 4 | 262 ms | 589 ms | 7.8 | 41x | 0 |
| 8 | 586 ms | 1183 ms | 8.2 | 43x | 0 |
| 16 | 1069 ms | 1928 ms | 8.6 | 45x | 0 |
| 32 | 2095 ms | 3918 ms | 8.6 | 45x | 0 |
| 64 | 5204 ms | 7494 ms | 8.7 | 46x | 0 |
| 100 | 7787 ms | 11890 ms | 8.6 | 45x | 0 |

Zero failures at every level; the GPU saturates at ~8.6 rps (45x realtime).

### On the "TTFB < 200 ms at 100 concurrent" target

Not reachable on this hardware, and not by tuning. OmniVoice is
non-autoregressive: each of the `num_step` denoise passes re-runs the whole
28-layer backbone over the entire sequence, for the conditional and
unconditional halves both. 100 simultaneous first-chunks is tens of TFLOP of
work; the card sustains ~8.6 requests/second, so a 100-deep queue is ~11 s deep
no matter how the queue is arranged.

What the numbers above do say:

* **100 concurrent voice-bot *sessions* are fine** if their combined turn rate
  stays at or under ~6 turns/second — i.e. each session speaking about once
  every 16 s. That is a normal IVR cadence, and median TTFB there is ~140 ms.
* Sustained rates above ~7 rps need a second GPU. Throughput scales linearly
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
