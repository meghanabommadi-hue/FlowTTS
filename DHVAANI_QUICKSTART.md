# DhVaani quickstart

Zero-shot TTS for 27 Indian languages on FlowTTS. Full guide: [docs/DHVAANI.md](docs/DHVAANI.md).

## 1. Install

```bash
cd ~/FlowTTS
source .venv/bin/activate
pip install -r requirements.txt
pip install "git+https://github.com/Ajaj-Ali/text_preprocessor_for_TTS.git"
```

## 2. Fetch the model (gated)

Accept the terms at <https://huggingface.co/ARTPARK-IISc/DhVaani-0.5> while
signed in, then:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
python -m flowtts.dhvaani.setup.fetch_model
```

Downloads the 491 MB checkpoint plus the Vocos vocoder.

## 3. Start the server

```bash
./run_dhvaani.sh --ports 4 --http-port 8000 --ctrl-port 8764 --profile balanced
```

| endpoint | what |
|---|---|
| `ws://host:8080` (…8083) | production WebSocket protocol (unchanged from FlowTTS) |
| `http://host:8000/docs` | OpenAPI browser |
| `http://host:8000/v1/audio/speech` | OpenAI-compatible speech |
| `http://host:8764/ready` | readiness |
| `http://host:8000/metrics` | Prometheus |

## 4. Create a voice

The transcript is **required** — DhVaani derives its speaking rate from
`prompt_frames / prompt_tokens`, so a wrong-length transcript makes the voice
speak at the wrong speed.

```bash
curl -X POST http://localhost:8000/v1/voices \
  -F file=@sample_files/simran.wav \
  -F voice_id=simran \
  -F transcript="नमस्ते, मैं वाणी बोल रही हूं" \
  -F language=hi

curl -X POST http://localhost:8000/v1/voices/simran/preview -o preview.wav
```

## 5. Synthesize

```bash
# WAV
curl -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"नमस्ते, आपका स्वागत है।","voice":"simran"}' -o out.wav

# streaming PCM, played as it arrives
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"आपकी ईएमआई बकाया है।","voice":"simran","response_format":"pcm","stream":true}' \
  | ffplay -f s16le -ar 24000 -ac 1 -
```

OpenAI SDK works unchanged:

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
c.audio.speech.create(model="dhvaani-0.5", voice="simran",
                      input="வணக்கம்!").stream_to_file("out.wav")
```

## 6. Verify and measure

```bash
# end-to-end health check -- run this first after deploying
python -m flowtts.dhvaani.test.smoke --voice-id simran --out /tmp/smoke

# unit tests (CPU, no model needed)
pytest flowtts/dhvaani/test/ -q

# where does the time go
python -m flowtts.dhvaani.test.bench latency --voice simran

# how far from the GPU roofline
python -m flowtts.dhvaani.test.bench step

# sustained load against the running server
python -m flowtts.dhvaani.test.loadtest ws --url ws://localhost:8080 \
    --voice simran --rps 100 --duration 60
```

## 7. Going faster

`num_step` is linear in cost and CFG is exactly 2x, so:

```bash
./run_dhvaani.sh --ports 4 --profile fast     # num_step=4, CFG off
```

For the lowest time-to-first-byte, build TensorRT engines once:

```bash
pip install tensorrt-cu12
python -m flowtts.dhvaani.setup.build_trt --max-batch 128
python -m flowtts.dhvaani.setup.build_trt --validate
./run_dhvaani.sh --ports 4 --profile fast --backend trt
```

**Capacity, honestly:** on a single L40S, ~200 RPS of 3-second utterances needs
the `fast` profile (4 steps, CFG off, 2–3 s prompt). `balanced` gives roughly
55–115 RPS, `quality` under 30. Check your own numbers:

```bash
python -m flowtts.dhvaani.test.bench capacity --target-rps 200 --utterance-s 3
```

See [docs/DHVAANI.md §8](docs/DHVAANI.md) for the full cost model.

## 8. Triton Inference Server (optional)

Only needed when the GPU is shared with other models:

```bash
python -m flowtts.dhvaani.setup.build_triton_repo --profile balanced
cat triton/README.md
```
