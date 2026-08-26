# FlowTTS / OmniVoice — API reference

Every command below was run against the live deployment and its output verified.

**Base URL**

```bash
BASE=http://101.53.141.123/tts       # public, through nginx
BASE=http://127.0.0.1:9000           # on-box, bypasses nginx (benchmarking)
```

Interactive docs: `$BASE/docs` · OpenAPI JSON: `$BASE/openapi.json`

Auth is off by default. If `FLOWTTS_API_KEYS` is set, add `-H "Authorization: Bearer <key>"`
or `-H "X-API-Key: <key>"` to every request.

---

## 1. Generate speech

### The simplest call

```bash
curl -X POST $BASE/v1/tts \
  -H 'content-type: application/json' \
  -d '{"text": "नमस्ते, आपका बकाया ₹2,500 है, कृपया आज ही भुगतान करें।"}' \
  --output speech.wav
```

Language is auto-detected from the script; the default voice is used. Response
body is the audio; metadata comes back in headers:

```
content-type: audio/wav
x-sample-rate: 24000
x-audio-format: wav
x-duration-seconds: 5.07
x-total-ms: 105
x-cache-hit: 0
x-language: hi
```

### With a voice, language and format

```bash
curl -X POST $BASE/v1/tts \
  -H 'content-type: application/json' \
  -d '{
    "text": "आपकी EMI ₹3,750 की due date निकल चुकी है।",
    "voice_id": "anika",
    "language": "hi",
    "format": "mp3",
    "sample_rate": 24000,
    "speed": 1.0
  }' --output speech.mp3
```

`format`: `wav` (default) · `pcm` (raw int16, no header) · `mp3` · `opus`
`sample_rate`: `24000` native · `16000` · `8000` (telephony)

### JSON response instead of raw audio

Add `Accept: application/json` to get base64 plus full metadata:

```bash
curl -X POST $BASE/v1/tts \
  -H 'content-type: application/json' -H 'accept: application/json' \
  -d '{"text": "Hello there.", "voice_id": "anika"}' | jq
```

```json
{
  "audio_base64": "UklGRi...",
  "metadata": {
    "sample_rate": 24000, "format": "wav", "duration_seconds": 0.43,
    "chunks": 1, "language": "en", "voice_id": "anika",
    "normalized_text": "Hello there.",
    "ttfb_ms": 213, "total_ms": 213, "real_time_factor": 0.495,
    "cache_hit": false
  }
}
```

---

## 2. Stream speech (low latency)

Same body, different path. First bytes arrive as soon as chunk 0 leaves the GPU
while later chunks are still generating — measured TTFB ~80–190 ms.

```bash
curl -N -X POST $BASE/v1/tts/stream \
  -H 'content-type: application/json' \
  -d '{
    "text": "नमस्ते, मैं बजाज finance से वाणी बोल रही हूं। आपका बकाया ₹2,500 है।",
    "voice_id": "anika", "language": "hi"
  }' --output stream.wav
```

`-N` disables curl's own buffering. The response is a WAV with a streaming
header (size fields `0xFFFFFFFF`) followed by PCM as it is produced.

### Raw PCM for a telephony pipeline

```bash
curl -N -X POST $BASE/v1/tts/stream \
  -H 'content-type: application/json' \
  -d '{"text": "आपका भुगतान सफल रहा।", "language": "hi",
       "format": "pcm", "sample_rate": 8000}' --output stream.pcm
```

No header at all — 8 kHz mono signed 16-bit little-endian, ready to push into a
call leg. Play it back with:

```bash
ffplay -f s16le -ar 8000 -ac 1 stream.pcm
```

### Measuring TTFB yourself

```bash
curl -N -o /dev/null -X POST $BASE/v1/tts/stream \
  -H 'content-type: application/json' \
  -d '{"text":"...","voice_id":"anika","language":"hi","format":"pcm"}' \
  -w 'ttfb=%{time_starttransfer}s total=%{time_total}s\n'
```

---

## 3. Clone a voice

Cloning is instant (~60 ms) and the voice is usable immediately — no restart.

**`reference_text` is required.** This server runs no ASR; a wrong transcript
degrades every future synthesis with that voice. 5–20 seconds of clean speech is
the sweet spot (longer references are trimmed to 20 s, because prompt tokens sit
in front of every generated chunk and cost latency on every request).

### From a file (multipart)

```bash
curl -X POST $BASE/v1/voices \
  -F voice_id=priya \
  -F reference_text="नमस्ते, मैं प्रिया बोल रही हूं, बजाज फाइनेंस से।" \
  -F language=hi \
  -F audio=@priya.wav
```

```json
{
  "status": "ok",
  "voice_id": "priya",
  "tokens": [8, 85],
  "reference_frames": 85,
  "reference_seconds": 3.4,
  "ref_rms": 0.0657,
  "ref_text": "नमस्ते, मैं प्रिया बोल रही हूं, बजाज फाइनेंस से।.",
  "language": "hi",
  "sample_rate": 24000,
  "npz": "/root/omnivoice-svc/voices/priya.npz"
}
```

Add `-F overwrite=true` to replace an existing voice (without it, a duplicate
`voice_id` returns **409**).

`voice_id` must match `[A-Za-z0-9_-]{1,64}`.

### From base64 (JSON)

```bash
python3 - <<'PY' > clone.json
import base64, json
print(json.dumps({
    "voice_id": "priya",
    "reference_text": "नमस्ते, मैं प्रिया बोल रही हूं, बजाज फाइनेंस से।",
    "language": "hi",
    "overwrite": True,
    "audio_base64": base64.b64encode(open("priya.wav", "rb").read()).decode(),
}))
PY

curl -X POST $BASE/v1/voices -H 'content-type: application/json' -d @clone.json
```

### Use the cloned voice

```bash
curl -X POST $BASE/v1/tts -H 'content-type: application/json' \
  -d '{"text": "अब मैं क्लोन की गई आवाज़ में बोल रही हूं।",
       "voice_id": "priya", "language": "hi"}' --output cloned.wav
```

### List voices

```bash
curl -s $BASE/v1/voices | jq
```

```json
{
  "voices": [
    {"voice_id": "anika", "language": "hi", "reference_frames": 215,
     "ref_text": "सर इस प्रोसेस में...", "sample_rate": 24000, "is_default": true}
  ],
  "default_voice": "anika"
}
```

### Delete a voice

```bash
curl -X DELETE $BASE/v1/voices/priya
# {"status":"ok","voice_id":"priya"}
```

Removes it from the registry and from disk. Unknown id returns **404**.

---

## 4. Clone without saving

### Preview — hear a voice before you keep it

Clone and speak in one call. Nothing is written to disk.

```bash
curl -X POST $BASE/v1/voices/preview \
  -F audio=@candidate.wav \
  -F reference_text="नमस्ते, मैं प्रिया बोल रही हूं।" \
  -F text="यह preview है, कोई voice save नहीं हुई।" \
  -F language=hi \
  -F num_step=8 \
  --output preview.wav
```

Form fields: `audio`, `reference_text`, `text`, and optionally `language`,
`speed`, `format`, `sample_rate`, `num_step`, `guidance_scale`.

### Inline reference audio — per-request cloning

Attach a reference to any `/v1/tts` or `/v1/tts/stream` call. Useful when the
voice comes from the request rather than from a registry.

```bash
python3 - <<'PY' > inline.json
import base64, json
print(json.dumps({
    "text": "This is an inline one-shot clone.",
    "language": "en",
    "reference_text": "नमस्ते, मैं प्रिया बोल रही हूं, बजाज फाइनेंस से।",
    "reference_audio": base64.b64encode(open("priya.wav", "rb").read()).decode(),
}))
PY

curl -X POST $BASE/v1/tts -H 'content-type: application/json' \
  -d @inline.json --output inline.wav
```

Note the reference is re-encoded on every request (~60 ms). For a voice you use
repeatedly, register it once with `POST /v1/voices` instead.

---

## 5. The three synthesis modes

| mode | how | example |
|---|---|---|
| **voice clone** | `voice_id`, or `reference_audio` + `reference_text` | a registered or ad-hoc speaker |
| **voice design** | `instruct` | `"Female, Elderly, British Accent"` |
| **auto voice** | neither | model picks a voice |

```bash
# voice design — no reference audio at all
curl -X POST $BASE/v1/tts -H 'content-type: application/json' \
  -d '{"text": "The world is full of amazing wonders.",
       "instruct": "Female, Elderly, British Accent", "language": "en"}' \
  --output design.wav

# auto voice
curl -X POST $BASE/v1/tts -H 'content-type: application/json' \
  -d '{"text": "The world is full of amazing wonders.", "language": "en"}' \
  --output auto.wav
```

### Inline control tags

OmniVoice's bracket syntax passes through the text preprocessor untouched:

```bash
# non-verbal
curl -X POST $BASE/v1/tts -H 'content-type: application/json' \
  -d '{"text": "[laughter] You really got me.", "voice_id": "anika"}' -o t1.wav

# ARPAbet pronunciation control
curl -X POST $BASE/v1/tts -H 'content-type: application/json' \
  -d '{"text": "He plays the [B EY1 S] guitar while catching a [B AE1 S] fish."}' -o t2.wav

# mixed emotion tokens
curl -X POST $BASE/v1/tts -H 'content-type: application/json' \
  -d '{"text": "[dissatisfaction-hnn] यह ठीक नहीं है। [surprise-oh] अच्छा?"}' -o t3.wav
```

---

## 6. OpenAI-compatible endpoint

Drop-in for `POST /v1/audio/speech`. Works with the OpenAI SDKs unchanged.

```bash
curl -X POST $BASE/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{
    "model": "omnivoice",
    "input": "Testing OpenAI compatibility.",
    "voice": "anika",
    "response_format": "mp3",
    "speed": 1.0
  }' --output openai.mp3
```

`stream: true` streams the response. It also accepts this server's own fields
(`language`, `instruct`, `generation`, `normalizer`, `sample_rate`).

```python
from openai import OpenAI
client = OpenAI(base_url="http://101.53.141.123/tts/v1", api_key="unused")
client.audio.speech.create(model="omnivoice", voice="anika",
                           input="नमस्ते").write_to_file("out.wav")
```

---

## 7. Tuning per request

Every OmniVoice generation parameter is overridable on any endpoint. Anything
omitted uses the server default.

```bash
curl -X POST $BASE/v1/tts -H 'content-type: application/json' -d '{
  "text": "आपका बकाया ₹2,500 है।",
  "voice_id": "anika",
  "generation": {
    "num_step": 4,
    "guidance_scale": 2.0,
    "t_shift": 0.1,
    "layer_penalty_factor": 5.0,
    "position_temperature": 5.0,
    "class_temperature": 0.0,
    "denoise": true,
    "preprocess_prompt": true,
    "postprocess_output": true,
    "pad_duration": 0.1,
    "fade_duration": 0.1,
    "audio_chunk_duration": 15.0,
    "audio_chunk_threshold": 30.0
  }
}' --output tuned.wav
```

| parameter | effect | default |
|---|---|---|
| `num_step` | **the latency dial.** Cost is linear in it. 4 fast, 8 balanced, 32 upstream quality | `16` |
| `guidance_scale` | CFG strength. **Not a speed knob here** — see below | `2.0` |
| `t_shift` | timestep shift; smaller emphasises low-SNR steps | `0.1` |
| `layer_penalty_factor` | pushes earlier codebooks to unmask first | `5.0` |
| `position_temperature` | randomness in which positions unmask. `0` = deterministic | `5.0` |
| `class_temperature` | randomness in token values. `0` = greedy | `0.0` |
| `denoise` | prepend `<\|denoise\|>` — cleaner on noisy references | `true` |
| `preprocess_prompt` | trim + punctuate the reference clip | `true` |
| `postprocess_output` | OmniVoice's silence removal | `true` |
| `pad_duration` / `fade_duration` | edge padding and fades (seconds) | `0.1` |
| `audio_chunk_duration` / `_threshold` | OmniVoice's own long-form chunking | `15` / `30` |

Top-level: `speed` (`>1` faster), `duration` (force exact seconds, overrides speed).

> **`guidance_scale` is not a throughput lever on this model.** OmniVoice builds
> the conditional + unconditional batch and runs the backbone over all of it on
> every step regardless of the value — measured 186 ms at `0.0` vs 190 ms at
> `2.0`. Setting it to 0 saves nothing and makes short chunks come back silent.
> Leave it at 2.0 and use `num_step` to trade latency for quality.

Latency at concurrency 1, 2.4 s of Hindi, TensorRT backbone:
`num_step` 4 → 86 ms · 8 → 190 ms · 16 → 342 ms · 32 → 695 ms.

---

## 8. Text preprocessing

### Preview what the model will actually say

Cheaper than a GPU call when debugging a mispronunciation.

```bash
curl -X POST $BASE/v1/normalize -H 'content-type: application/json' \
  -d '{"text": "आपका OTP 4821 है, ₹2,500 15/04/2026 तक जमा करें। mail: ravi@abc.co.in",
       "language": "hi"}' | jq
```

```json
{
  "original":   "आपका OTP 4821 है, ₹2,500 15/04/2026 तक जमा करें। mail: ravi@abc.co.in",
  "normalized": "आपका O T P चार आठ दो एक है, दो हज़ार पाँच सौ रुपये पन्द्रह अप्रैल दो हज़ार छब्बीस तक जमा करें। mail: ravi at abc dot co dot in",
  "language": "hi",
  "resolved_language": "hi",
  "omnivoice_language": "hi",
  "detected_language": "hi",
  "chunks": [
    {"index": 0, "text": "...", "estimated_seconds": 2.25},
    {"index": 1, "text": "...", "estimated_seconds": 2.58}
  ],
  "estimated_seconds": 10.41
}
```

Note what happened: the OTP is read digit by digit, the amount as a cardinal, the
date localized, the acronym spelled out, the email spoken — and the Latin run
inside the Hindi sentence handled in its own language.

### Turning stages off

```bash
curl -X POST $BASE/v1/tts -H 'content-type: application/json' -d '{
  "text": "Read 2026 exactly as written",
  "normalizer": {"numbers": false, "datetime": false}
}' --output raw.wav

# disable normalization entirely
curl -X POST $BASE/v1/tts -H 'content-type: application/json' \
  -d '{"text": "...", "normalize": false}' --output raw.wav
```

Toggles: `enabled`, `numbers`, `datetime`, `urls_emails`, `phone_numbers`,
`otp_digit_splitting`, `abbreviations`, `symbols`, `contractions`, `code_mixed`,
`lowercase`, `min_digit_run`, `latin_language`.

### Supported languages

```bash
curl -s $BASE/v1/languages | jq '.count, .languages[:3]'
```

30 languages have normalization tables, including all 22 scheduled languages of
India. OmniVoice itself speaks 600+ — anything not listed still synthesizes, it
just gets lighter text preprocessing.

| | codes |
|---|---|
| Scheduled languages | `hi` `bn` `mr` `te` `ta` `gu` `ur` `kn` `or` `ml` `pa` `as` `mai` `sat` `ks` `ne` `sd` `kok` `doi` `mni` `brx` `sa` |
| Dialects | `bho` `hne` `mag` `awa` `raj` `tcy` |
| English | `en` `en-IN` |

Names and alternate codes resolve too: `"hindi"`, `"odia"`, `"bangla"`,
`"panjabi"`, `"ory"`, `"npi"`. `/v1/normalize` reports all three views —
what you sent (`language`), the normalizer's canonical code
(`resolved_language`), and what OmniVoice is actually given
(`omnivoice_language`, which is ISO 639-3 for several Indic languages, so
`or` → `ory` and `ne` → `npi`):

```bash
curl -s -X POST $BASE/v1/normalize -H 'content-type: application/json' \
  -d '{"text":"₹1,200","language":"odia"}' \
  | jq '{language, resolved_language, omnivoice_language, normalized}'
# {"language":"odia","resolved_language":"or","omnivoice_language":"ory",
#  "normalized":"ଏକ ହଜାର ଦୁଇ ଶହେ ଟଙ୍କା"}
```

---

## 9. WebSocket

`ws://101.53.141.123/tts/ws/<call_id>` (public) or `ws://127.0.0.1:9080/ws/<call_id>`
(on-box raw port, for existing clients).

**Send:**

```json
{"type": "synthesize", "text_id": "utt-1",
 "text": "नमस्ते, आपका बकाया ₹2,500 है।",
 "voice_id": "anika", "language": "hi", "speed": 1.0,
 "sample_rate": 24000, "generation": {"num_step": 4}}
```

**Receive:** repeated **binary** frames, each a JSON header immediately followed
by raw little-endian int16 PCM:

```
{"type":"audio_chunk","call_id":"...","text_id":"utt-1","chunk_index":0,
 "sample_rate":24000,"encoding":"pcm_int16","wav_bytes":81120,"tokens":40560,
 "is_final":false,"cache_hit":false}<PCM bytes…>
```

then one final **text** frame:

```json
{"type":"audio_done","chunks":3,"total_wav_bytes":242880,"sample_rate":24000,
 "llm_ttft_ms":77,"rtf":0.023,"cancelled":false}
```

Split each binary frame at the first `}`; everything after it is audio.
Concatenate the PCM across frames for the full utterance.

Other messages: `{"type":"cancel","text_id":"utt-1"}` stops an utterance in
flight · `{"type":"ping"}` → `{"type":"pong"}` · errors arrive as
`{"type":"error","text_id":...,"error":"..."}` without closing the socket.

```python
import asyncio, json, websockets

async def speak(text, voice="anika", lang="hi"):
    url = "ws://101.53.141.123/tts/ws/call-1"
    async with websockets.connect(url, max_size=64*1024*1024) as ws:
        await ws.send(json.dumps({"type": "synthesize", "text_id": "u1",
                                  "text": text, "voice_id": voice,
                                  "language": lang}))
        pcm = b""
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                pcm += msg[msg.index(b"}") + 1:]      # play this immediately
            else:
                print(json.loads(msg)); return pcm

asyncio.run(speak("नमस्ते, यह websocket के ज़रिए है।"))
```

---

## 10. Health and observability

```bash
curl $BASE/healthz     # {"status":"ok","ready":true}          — liveness
curl $BASE/readyz      # ready + voice list + available formats — readiness
curl $BASE/v1/stats    # engine counters, batch sizes, TTFB percentiles
curl $BASE/metrics     # Prometheus
```

```bash
curl -s $BASE/v1/stats | jq '{counters, ttfb_ms, backbone: .engine.backbone.backend}'
```

```json
{
  "counters": {"requests": 507, "streamed": 475, "errors": 0,
               "cache_hits": 29, "cache_misses": 1, "rejected": 0},
  "ttfb_ms": {"count": 475, "p50": 180, "p90": 331.3, "p99": 514.4},
  "backbone": "tensorrt"
}
```

---

## 11. Errors

| status | meaning |
|---|---|
| **400** | bad request — e.g. `reference_audio` without `reference_text`, invalid base64 |
| **401** | missing/invalid API key (only when `FLOWTTS_API_KEYS` is set) |
| **404** | unknown `voice_id` on delete |
| **409** | voice already exists — pass `overwrite=true` |
| **413** | reference audio over 64 MB |
| **422** | schema violation — missing field, or a parameter out of range |
| **500** | synthesis produced no audible output (retried once internally first) |
| **503** | model still loading, restarting, or recovering from GPU OOM |

All errors are JSON: `{"detail": "..."}`.

```bash
$ curl -X POST $BASE/v1/voices -H 'content-type: application/json' \
    -d '{"voice_id":"anika","reference_text":"hi","audio_base64":"UklGRg=="}'
{"detail":"voice 'anika' exists; pass overwrite=true to replace"}

$ curl -X DELETE $BASE/v1/voices/nope
{"detail":"voice 'nope' not found"}
```

Streaming responses are special: chunk 0 is generated *before* the response
headers are sent, so a failed synthesis is a real status code rather than a
`200` with an empty body. A failure after streaming has begun can only end the
stream — check `x-duration-seconds` on the non-streaming path if you need
certainty.

---

## 12. Recipes

**Batch-generate a prompt library** (populates the WAV cache, so production
calls hit it in ~1 ms):

```bash
while IFS= read -r line; do
  curl -s -X POST $BASE/v1/tts -H 'content-type: application/json' \
    -d "$(jq -nc --arg t "$line" '{text:$t,voice_id:"anika",language:"hi",generation:{num_step:32}}')" \
    -o "prompts/$(echo -n "$line" | shasum -a 256 | cut -c1-16).wav"
done < prompts.txt
```

**Clone every wav in a directory** (filename becomes the voice id, `.txt`
sidecar the transcript):

```bash
for f in refs/*.wav; do
  id=$(basename "$f" .wav)
  curl -s -X POST $BASE/v1/voices \
    -F voice_id="$id" -F language=hi -F overwrite=true \
    -F reference_text="$(cat "refs/$id.txt")" \
    -F audio=@"$f" | jq -c '{voice_id, reference_seconds}'
done
```

**Health-gate a deploy:**

```bash
until curl -sf $BASE/readyz >/dev/null; do sleep 2; done && echo ready
```
