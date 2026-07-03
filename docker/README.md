# Running FlowTTS (Fish Audio S2 Pro) in Docker

Two services on one GPU host:

- **`fish-s2pro`** (GPU) — `sgl-omni serve` running `fishaudio/s2-pro` (all model +
  codec work). Built from [`docker/fish_s2pro.Dockerfile`](fish_s2pro.Dockerfile).
- **`flowtts-gateway`** (CPU) — the WebSocket + control-API proxy (protocol, voices,
  WAV cache, metrics). Built from [`docker/Dockerfile`](Dockerfile).

> ⚠ **License:** `fishaudio/s2-pro` is under the **Fish Audio Research License**
> (non-commercial). Commercial use needs a separate license (`business@fish.audio`).

## Host prerequisites (once)

- NVIDIA driver (`nvidia-smi`), an H200-class GPU recommended.
- **nvidia-container-toolkit**:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  Verify: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`
- Docker Compose v2 (`docker compose version`).

## Quick start

```bash
cd ~/FlowTTS            # the repo root (contains docker-compose.yml)

# 1) Start the GPU backend. First run downloads ~10GB weights (HF_TOKEN needed for
#    the gated repo) and captures CUDA graphs — the healthcheck has a 10-min grace.
export HF_TOKEN=hf_...
docker compose up -d fish-s2pro
docker compose ps                         # wait until fish-s2pro is "healthy"

# 2) Start the gateway.
docker compose up -d flowtts-gateway
docker compose logs -f flowtts-gateway
```

Gateway: `ws://<vm-ip>:8080/ws/<call_id>`; control API + Prometheus on `:8764`.

## Test it

```bash
# streaming benchmark from inside the gateway container
docker compose exec flowtts-gateway python -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 5 --streaming

# quick smoke test (Hindi/English)
docker compose exec flowtts-gateway bash run.sh --test --ports 1

# unit tests (no GPU)
docker compose exec flowtts-gateway python -m pytest \
  flowtts/test/test_text_chunker.py flowtts/test/test_voice_store.py flowtts/test/test_pcm.py -q

# hit the backend directly
curl -sf localhost:8000/v1/audio/speech -d '{"input":"hello","stream":false}' -o out.wav
```

## Voices

**Recommended — REST on the running gateway** (live, no restart, no GPU work):

```bash
curl -sf -X POST http://localhost:8764/voices \
  -F voice_id=niharika -F preferred_lang=bn \
  -F ref_text="<exact transcript of the clip>" \
  -F audio=@sample_files/niharika_bn.wav

curl -s http://localhost:8764/voices        # list loaded voices
```

Offline CLI alternative (no GPU):

```bash
docker compose run --rm flowtts-gateway clone --add priya \
  --ref-audio sample_files/simran.wav --ref-text "नमस्ते, मैं प्रिया बोल रही हूँ।" --lang hi
docker compose run --rm flowtts-gateway clone --list
```

Cloned voices land in `./voices/` on the host (bind-mounted into BOTH services at the
same path, so the backend can read the reference clips). The REST endpoint registers a
voice live; the CLI needs a gateway restart.

## Tuning

**Backend (`docker-compose.yml` env on `fish-s2pro`):**

| Env var | Meaning | Start |
|---|---|---|
| `TTS_BATCH_MAX_ITEMS` | server-side batch cap (throughput) | `32` → sweep 48/64 |
| `MEM_FRACTION` | `--mem-fraction-static` (KV-cache memory) | unset → try `0.85` |
| `FISH_MODEL` | weights: HF repo id or local/licensed path | `fishaudio/s2-pro` |
| `HF_TOKEN` | pull the gated weights | (required) |

**Gateway (env on `flowtts-gateway`):**

| Env var | Meaning | Default |
|---|---|---|
| `FLOWTTS_FISH__BACKEND_URL` | backend base URL | `http://fish-s2pro:8000` |
| `FLOWTTS_FISH__REFERENCE_MODE` | `local` (shared vol) / `base64` (inline) | `local` |
| `FLOWTTS_OUTPUT__SAMPLE_RATE` | resample from 24k (telephony) | unset = 24000 |
| `PORTS` / `BASE_PORT` / `CTRL_PORT` | gateway shape | `1` / `8080` / `8764` |

See [`docs/fish_s2pro_acceleration.md`](../docs/fish_s2pro_acceleration.md) for the full playbook.

## Notes

- **Backend image:** `docker/fish_s2pro.Dockerfile` bases on the **official
  `lmsysorg/sglang-omni:dev`** image (built with `uv`), which already has `sgl-omni` +
  torch/flash-attn/CUDA and the fishaudio deps with their conflicts (protobuf:
  `descript-audiotools` vs `s3prl`/`onnxruntime`) resolved. Do **not** `pip install -e .`
  sglang-omni from source on a bare CUDA image — that re-triggers `ResolutionImpossible`.
  Need CUDA 12? set `--build-arg BASE_IMAGE=lmsysorg/sglang-omni:dev-cu12` (or `-cu129`).
- **No custom build at all (optional):** you can skip our thin Dockerfile and run the
  official image directly — `docker pull lmsysorg/sglang-omni:dev` then `sgl-omni serve
  --model-path fishaudio/s2-pro --config examples/configs/s2pro_tts.yaml --port 8000`.
- **Local / licensed weights:** set `FISH_MODEL=/models/s2-pro` and mount your weights
  into the backend to skip the HF download and pin the version.
- **Multiple WS ports:** set `PORTS=8` on the gateway and add `8081:8081 … 8087:8087`
  to its `ports:` list.
- **Persistence:** weights in the `hf-cache` volume, WAV cache in `wav-cache`, voices in
  `./voices/` — none lost on rebuild.
- **Rebuild after code changes:** `docker compose build && docker compose up -d`.
- **`reference_mode`:** if the backend can't read the shared `voices/` volume, set
  `FLOWTTS_FISH__BACKEND_VOICES_DIR` or switch to `FLOWTTS_FISH__REFERENCE_MODE=base64`.
