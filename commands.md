# FlowTTS (OmniVoice) Commands

## Docker (recommended on a dev VM — keeps the box clean)

```bash
docker compose run --rm omnivoice-tts setup     # one-time: model download + build voices
docker compose up -d                  # serve on :8080 (WS) + :8764 (ctrl/metrics)
docker compose logs -f omnivoice-tts
```

Needs the NVIDIA driver + nvidia-container-toolkit on the host. Full guide + tuning:
[docker/README.md](docker/README.md). Everything below runs the same **inside** the
container (`docker compose exec omnivoice-tts …`) or on a bare-metal venv install.

---

## One-time setup (bare metal / venv)

```bash
# 1. Install PyTorch CUDA build matching your box, then deps + model + voices:
bash flowtts/setup/setup.sh
```

This installs `requirements.txt`, downloads `k2-fsa/OmniVoice`, and builds voice
npz artifacts from `sample_files/` (+ `voices/manifest.json`).

## Launch server

```bash
cd ~/FlowTTS
source .venv/bin/activate
bash run.sh --ctrl-port 8764 --ports 1
```

- `--ports N` opens N WebSocket ports from `--port` (default 8080).
- The model loads once and self-warms; the control API is on `--ctrl-port`.

### Engine tuning (speed / throughput)

```bash
bash run.sh --num-step 12 --max-batch 48 --batch-timeout-ms 8   # faster / bigger batches
bash run.sh --compile                                           # torch.compile (+CUDA graphs); slow first run
```

- `--num-step` — diffusion steps; the dominant latency knob (16 default; try 8–12).
- `--max-batch` / `--batch-timeout-ms` — dynamic in-flight batch size / window.
- All also settable via env: `FLOWTTS_OMNIVOICE__NUM_STEP=8`, `FLOWTTS_OMNIVOICE__MAX_BATCH=64`, etc.
- Output rate: `FLOWTTS_OUTPUT__SAMPLE_RATE=16000` (or 8000) to resample from native 24 kHz.

## Voices (clone by alias)

### Easiest — REST endpoint on the running server (no restart, no extra GPU load)

`POST /voices` on the control API (`:8764`). Inputs: `audio` file, `voice_id`,
`ref_text`, optional `preferred_lang`. Returns clone status; voice is usable immediately.

```bash
# multipart upload
curl -sf -X POST http://localhost:8764/voices \
  -F voice_id=niharika \
  -F preferred_lang=hi \
  -F ref_text="एकटा कथा बोली कोलकाता का जो मिष्टी दही है ना …" \
  -F audio=@voices/voice_preview_niharika_bengali.mp3
# → {"status":"ok","voice_id":"niharika","tokens":[8,NNN],"sample_rate":24000, ...}

curl -s http://localhost:8764/voices          # list loaded voices
```
Then synthesize with `"voice_id":"niharika"` right away — no restart needed.

### Offline CLI (alternative)

```bash
python -m flowtts.voices.clone --build-all --manifest voices/manifest.json   # build all
python -m flowtts.voices.clone --add priya --ref-audio sample_files/priya.wav \
    --ref-text "नमस्ते, मैं प्रिया बोल रही हूँ।"                              # one voice (ref_text required)
python -m flowtts.voices.clone --list                                        # list installed
```

Select a voice per request with `voice_id`.

## Smoke test

```bash
bash run.sh --test --ports 1                 # against a running server
```

## Send requests / benchmark

```bash
# streaming (measures time-to-first-chunk)
python -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 75 --streaming

# throughput sweep across ports
python -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 200 --concurrency 16 --streaming

# a specific voice
python -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 20 --voice priya
```

## Unit tests (no GPU required)

```bash
python -m pytest flowtts/test/test_text_chunker.py flowtts/test/test_voice_npz.py flowtts/test/test_pcm.py -q
```

## WebSocket contract (in / out)

**Client → Server**
```json
{ "type": "synthesize", "call_id": "c1", "text_id": "t1",
  "text": "...", "voice_id": "priya", "speed": 1.0, "language": "hi", "streaming": true }
```
Also `{ "type": "cancel", "text_id": "t1" }`.

**Server → Client** (streaming): repeated binary frames of
`audio_chunk` JSON header (`{type,call_id,text_id,chunk_index,sample_rate,encoding,tokens,is_final,cache_hit}`)
**+ raw int16 PCM bytes appended in the same frame**, then a final `audio_done` JSON,
plus `error` / `cancelled` as applicable.

## Kill server

```bash
kill $(ss -tlnp | grep :8764 | grep -oP 'pid=\K[0-9]+')
```

## Notes

- `add_voice.py` is deprecated → use `python -m flowtts.voices.clone`.
- WAV cache (`~/FlowTTS/cached_data/<sha256(text)>.wav`) still bypasses the model on hit
  — cache files must be at the configured output sample rate.
- 200 RPS on one H200 is a tuning target: combine large `--max-batch`, low `--num-step`,
  short first chunks, and the WAV cache; verify empirically with the throughput sweep.
