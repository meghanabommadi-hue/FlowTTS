# Running FlowTTS (OmniVoice) in Docker

Keeps the VM clean — all Python/CUDA deps live in the container. Works on any
NVIDIA GPU with compute capability ≥ 8.0 (your **RTX 6000 Ada** = 8.9, H200 = 9.0).

## Host prerequisites (once)

- NVIDIA driver (check with `nvidia-smi`)
- **nvidia-container-toolkit** so Docker can see the GPU:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  Verify: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`
- Docker Compose v2 (`docker compose version`).

## Quick start

```bash
cd ~/FlowTTS            # the repo root (contains docker-compose.yml)

# 1) One-time setup: ensure model (local weights or HF) + build any voices with a ref_text.
#    (No ASR — ref_text is required in voices/manifest.json to clone a voice.)
docker compose run --rm omnivoice-tts setup

# 2) Serve (foreground; Ctrl-C to stop)
docker compose up
#    …or background:
docker compose up -d && docker compose logs -f omnivoice-tts
```

The server is now on `ws://<vm-ip>:8080/ws/<call_id>`; control API + Prometheus on `:8764`.

## Test it

```bash
# streaming benchmark from inside the container
docker compose exec omnivoice-tts python -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 5 --streaming

# quick smoke test (Hindi/English)
docker compose exec omnivoice-tts bash run.sh --test --ports 1

# unit tests (no GPU)
docker compose exec omnivoice-tts python -m pytest flowtts/test/test_text_chunker.py flowtts/test/test_voice_npz.py flowtts/test/test_pcm.py -q
```

## Voices

```bash
# add a voice (drop the wav in sample_files/ first — it's in the image + bind mount)
docker compose run --rm omnivoice-tts clone --add priya --ref-audio sample_files/priya.wav \
    --ref-text "नमस्ते, मैं प्रिया बोल रही हूँ।"

docker compose run --rm omnivoice-tts clone --list
```

Built `.npz` land in `./voices/` on the host (bind-mounted) and persist across
container rebuilds. Restart the server to pick up new voices.

## Tuning (edit `docker-compose.yml` env, or pass `-e`)

| Env var | Meaning | RTX 6000 Ada start |
|---|---|---|
| `FLOWTTS_OMNIVOICE__NUM_STEP` | diffusion steps (latency knob) | `16` (try 10–12) |
| `FLOWTTS_OMNIVOICE__MAX_BATCH` | dynamic batch size | `16` (48GB); raise on H200 |
| `FLOWTTS_OMNIVOICE__BATCH_TIMEOUT_MS` | batch window | `8` |
| `FLOWTTS_OMNIVOICE__COMPILE_MODEL` | torch.compile (+CUDA graphs) | `true` after it's stable |
| `FLOWTTS_OUTPUT__SAMPLE_RATE` | resample from 24k (e.g. telephony) | unset = 24000 |
| `PORTS` / `BASE_PORT` / `CTRL_PORT` | server shape | `1` / `8080` / `8764` |

See [`docs/omnivoice_acceleration.md`](../docs/omnivoice_acceleration.md) for the full speedup playbook.

## Notes

- **Local weights (skip the HF download):** place the OmniVoice snapshot at `./model_dir/base`
  on the host. It's bind-mounted to `/root/FlowTTS/model_dir` and used automatically when present
  (config `model_path`), so `setup`/serve won't touch HuggingFace. Override the path with
  `FLOWTTS_OMNIVOICE__MODEL_PATH`.
- **Multiple WS ports:** set `PORTS=8` and add `8081:8081 … 8087:8087` to the compose `ports:` list.
- **Persistence:** the model lives in the `hf-cache` volume, the WAV cache in `wav-cache`,
  voices in `./voices/` — none are lost on `docker compose build`/`up`.
- **OOM:** on repeated CUDA OOM the server exits and `restart: unless-stopped` brings it back;
  if it loops, lower `MAX_BATCH` / `NUM_STEP`.
- **Rebuild after code changes:** `docker compose build omnivoice-tts` (or `docker compose up --build`).
- **Custom torch/CUDA:** override build args, e.g.
  `docker build -f docker/Dockerfile --build-arg TORCH_VERSION=2.6.0 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t flowtts-omnivoice .`
