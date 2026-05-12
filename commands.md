# FlowTTS Commands

## Launch server
```bash
cd FlowTTS/
source llmc/bin/activate
source .venv/bin/activate

```
```bash
cd /root/FlowTTS
bash run.sh --ctrl-port 8764
```

## Open N ports

```bash
python3 -m flowtts.test.open_ports --n 40
```

## Send N requests (1 per port)

```bash
# Full pipeline (LLM + decoder, returns WAV)
python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75

# Streaming — audio chunks sent as they are generated (shows time-to-first-chunk)
python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75 --streaming

# Streaming + save each chunk as an individual WAV file
python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75 --streaming --save-chunks

# LLM only — no decoder, measure pure generation latency
python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75 --skip-decoder
```

- Auto-discovers all open ports from the server
- Assigns exactly 1 request per port (round-robin matches when requests == ports)
- `--streaming` uses the streaming pipeline: LLM tokens → decoder → WAV in chunks; summary shows `ttff(s)` (time to first audio chunk) per request
- `--save-chunks` (requires `--streaming`) saves each chunk WAV individually alongside the combined output
- Streaming chunk size, crossfade, and fade-out are configured in `flowtts/core/config.py → StreamingSettings` (or via env vars)
- `--skip-decoder` sends `skip_decoder=true` per-request to the running server (no WAV decode, tokens only)

---

## Open M more ports

```bash
python3 -m flowtts.test.open_ports --n 50
```

- Continues from the next port after the highest already open

---

## Send M+N requests (1 per port)

```bash
cd /root/FlowTTS && python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 90
```

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

## Launch server without decoder (LLM only, faster)

```bash
cd /root/FlowTTS && bash run.sh --ctrl-port 8764 --skip-decoder
```

Returns `audio_tokens` only, no WAV. Use for LLM latency benchmarking.

---

## Test batch decode (no server needed)

Must set LD_LIBRARY_PATH so onnxruntime uses GPU (libcudnn.so.9):
```bash
cd /root/FlowTTS
export LD_LIBRARY_PATH=/root/CleanTTSData/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
python3 flowtts/test/test_concurrent_decode.py --n-requests 30 --rounds 3
```

- Codec initialised **once** (model load + ONNX session warm-up)
- Runs R rounds of N concurrent `decode_async()` — only round 1 pays cold-start cost
- Reports per-round: batch sizes dispatched, GPU call count, latency (p50/p95/p99), req/s
- Without `libcudnn.so.9` on LD_LIBRARY_PATH, ONNX falls back to CPU (~15x slower!)
- `run.sh` sets this automatically when launching the server

Options:
```bash
# 90 requests, 5 rounds, larger GPU chunk, more ONNX workers
export LD_LIBRARY_PATH=/root/CleanTTSData/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
/root/CleanTTSData/.venv/bin/python3 flowtts/test/test_concurrent_decode.py \
    --n-requests 90 --rounds 5 --gpu-chunk 100 --onnx-workers 4
```

---

## Enable TensorRT decoder (3-5x faster, first run ~60s compile)

Set in config or via env var:
```bash
# Via env var (one-off)
FLOWTTS_DECODER__USE_TRT=true cd /root/FlowTTS && bash run.sh --ports 1

# Or edit flowtts/core/config.py → DecoderSettings → use_trt: bool = True
```

- Engine cached to disk as `decoder_trt_b50.ep` next to model weights
- Subsequent starts load cache instantly (no recompile)
- Requires: `torch-tensorrt 2.9.0` (already installed)

---

## Notes

- Server ctrl API runs on `127.0.0.1:8764`
- WS ports start at `8080` by default
- All requests run fully parallel (sglang batches LLM, decoder batches via TTSCodec queue)
- WAV output saved to `/root/FlowTTS/test/pipeline_test_YYYYMMDD_HHMMSS/`
- `--skip-decoder` skips ONNX/GPU decode — returns tokens only, no audio_base64
- Decoder config lives in `DecoderSettings` (`max_batch`, `gpu_chunk_size`, `onnx_workers`, `use_trt`)
- Streaming config lives in `StreamingSettings` (`chunk_tokens`, `crossfade_samples`, `fade_out_samples`)
  - Override via env: `FLOWTTS_STREAMING__CHUNK_TOKENS=50`, `FLOWTTS_STREAMING__CROSSFADE_SAMPLES=0`, `FLOWTTS_STREAMING__FADE_OUT_SAMPLES=480`
