# FlowTTS Commands

## Launch server (Mira — default)
```bash
cd /home/ubuntu/FlowTTS
source .venv/bin/activate
bash run.sh --ctrl-port 8764
```

## Launch server (VoxCPM)
```bash
cd /home/ubuntu/FlowTTS
FLOWTTS_MODEL_TYPE=voxcpm \
.venv/bin/python -m flowtts.server --ports 1 --ctrl-port 8764
```

Defaults already point to the correct paths:
- Model: `/home/ubuntu/voxcpm/model`
- Ref audio: `/home/ubuntu/voxcpm/deployment/simran_3s.wav`
- Ref audio text: `"नमस्ते मैं बजाज फिनानके की तरफ से बात कर रही हूँ।"`

Override with env vars:
```bash
FLOWTTS_MODEL_TYPE=voxcpm \
FLOWTTS_VOXCPM__REF_AUDIO=/home/ubuntu/voxcpm/deployment/simran_3s.wav \
FLOWTTS_VOXCPM__REF_AUDIO_TEXT="नमस्ते मैं बजाज फिनानके की तरफ से बात कर रही हूँ।" \
.venv/bin/python -m flowtts.server --ports 1 --ctrl-port 8764
```

> **Note:** `REF_AUDIO_TEXT` is required when `REF_AUDIO` is set.
> If `REF_AUDIO_TEXT` is empty the server falls back to zero-shot mode automatically.
> Model weights: `/home/ubuntu/voxcpm/model/` (4.6 GB safetensors, real weights — not LFS pointers).

---

## Open N ports

```bash
python3 -m flowtts.test.open_ports --n 40
```

---

## Send N requests (1 per port)

```bash
# Full pipeline — Mira (LLM + decoder, returns WAV)
python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75

# Full pipeline — VoxCPM
python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75 --model-type voxcpm

# Streaming — audio chunks sent as they are generated (shows time-to-first-chunk)
python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75 --streaming

# Streaming + save each chunk as an individual WAV file
python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75 --streaming --save-chunks

# LLM only — no decoder, measure pure generation latency (Mira only)
python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75 --skip-decoder
```

- Auto-discovers all open ports from the server
- Assigns exactly 1 request per port (round-robin matches when requests == ports)
- `--model-type voxcpm` tells the test to launch/use a VoxCPM server and use English test sentences
- `--streaming` uses the streaming pipeline; summary shows `ttff(s)` (time to first audio chunk)
- `--save-chunks` (requires `--streaming`) saves each chunk WAV individually alongside the combined output
- Streaming chunk size, crossfade, and fade-out are configured in `flowtts/core/config.py → StreamingSettings`
  - Override via env: `FLOWTTS_STREAMING__CHUNK_TOKENS=50`, `FLOWTTS_STREAMING__CROSSFADE_SAMPLES=0`
- `--skip-decoder` sends `skip_decoder=true` per-request (Mira only, tokens-only, no WAV)

---

## Open M more ports

```bash
python3 -m flowtts.test.open_ports --n 50
```

Continues from the next port after the highest already open.

---

## Kill server (closes all ports)

```bash
kill $(ss -tlnp | grep :8764 | grep -oP 'pid=\K[0-9]+')
```

Or by PID directly:
```bash
kill -9 <pid>
```

---

## Check open ports

```bash
ss -tlnp | grep python3 | awk '{print $4}' | sort -t: -k2 -n
```

---

## Launch server without decoder (Mira — LLM only, faster)

```bash
bash run.sh --ctrl-port 8764 --skip-decoder
```

Returns `audio_tokens` only, no WAV. Use for LLM latency benchmarking.

---

## Test batch decode — Mira ncodec (no server needed)

```bash
cd /home/ubuntu/FlowTTS
python3 flowtts/test/test_concurrent_decode.py --n-requests 30 --rounds 3
```

- Codec initialised **once** (model load + ONNX session warm-up)
- Runs R rounds of N concurrent `decode_async()` — only round 1 pays cold-start cost
- Reports per-round: batch sizes dispatched, GPU call count, latency (p50/p95/p99), req/s

Options:
```bash
python3 flowtts/test/test_concurrent_decode.py \
    --n-requests 90 --rounds 5 --gpu-chunk 100 --onnx-workers 4
```

---

## Enable TensorRT decoder — Mira only (3-5x faster, first run ~60s compile)

```bash
# Via env var (one-off)
FLOWTTS_DECODER__USE_TRT=true bash run.sh --ports 1

# Or edit flowtts/core/config.py → DecoderSettings → use_trt: bool = True
```

- Engine cached to disk as `decoder_trt_b50.ep` next to model weights
- Subsequent starts load cache instantly (no recompile)

---

## VoxCPM tuning knobs

All override via env var (`FLOWTTS_VOXCPM__<FIELD>`):

| Env var | Default | Effect |
|---|---|---|
| `FLOWTTS_VOXCPM__MODEL_DIR` | `~/models/voxcpm2` | Path to model checkpoint |
| `FLOWTTS_VOXCPM__REF_AUDIO` | `~/models/voxcpm2/ref_audio.wav` | Reference audio for voice cloning |
| `FLOWTTS_VOXCPM__REF_AUDIO_TEXT` | `""` | Transcript of ref audio (required with ref audio) |
| `FLOWTTS_VOXCPM__INFERENCE_TIMESTEPS` | `6` | Diffusion ODE steps (fewer = faster, lower quality) |
| `FLOWTTS_VOXCPM__MAX_NUM_SEQS` | `64` | Max parallel decode sequences |
| `FLOWTTS_VOXCPM__GPU_MEMORY_UTILIZATION` | `0.80` | VRAM fraction for model + KV cache |
| `FLOWTTS_VOXCPM__CFG_VALUE` | `2.0` | Classifier-free guidance scale |
| `FLOWTTS_VOXCPM__TEMPERATURE` | `1.0` | Sampling temperature |
| `FLOWTTS_VOXCPM__MAX_GENERATE_LENGTH` | `2000` | Max latent patches to generate |

---

## Notes

- Server ctrl API runs on `127.0.0.1:8764`
- WS ports start at `8765` by default
- Model type is selected by `FLOWTTS_MODEL_TYPE=mira|voxcpm` (default: `mira`)
- All requests run fully parallel (sglang batches for Mira; VoxCPM2 scheduler batches internally)
- WAV output saved to `~/FlowTTS/test/pipeline_test_YYYYMMDD_HHMMSS/`
- VoxCPM outputs **48 kHz** audio; Mira outputs **16 kHz**
- Decoder config (Mira) lives in `DecoderSettings` (`max_batch`, `gpu_chunk_size`, `onnx_workers`, `use_trt`)
