# Voices — Fish S2 Pro reference-clip artifacts

Each voice here is a **reference clip + transcript** addressed by alias:

- `<alias>.wav`  — the reference audio (mono, written by the cloner)
- `<alias>.json` — the manifest: `{alias, ref_text, language, audio_file}`

The gateway loads all `*.json` at startup and resolves the WebSocket `voice_id`
to `references=[{audio_path, text}]`, which it sends to the Fish S2 Pro sglang
backend. The backend encodes the clip into VQ codes once and caches the KV via
RadixAttention, so repeated same-voice requests hit the prefix cache (~86–90%).

> No codec tokens are precomputed here anymore (the old OmniVoice `.npz` format is
> gone). `ref_text` is still **mandatory** and must be the exact transcript of the
> clip, in the clip's language/script.

## Build voices (no GPU needed)

```bash
# Build every voice defined in the manifest (each entry needs ref_audio + ref_text):
python -m flowtts.voices.clone --build-all --manifest voices/manifest.json

# A single voice — ref_text is REQUIRED (no ASR/auto-transcribe):
python -m flowtts.voices.clone --add priya --ref-audio sample_files/simran.wav \
    --ref-text "नमस्ते, मैं प्रिया बोल रही हूँ।" --lang hi

# List installed voices
python -m flowtts.voices.clone --list
```

Or clone live over REST (no restart) — see the top-level README / DEV_GUIDE:
`curl -i -X POST localhost:8764/voices -F voice_id=priya -F preferred_lang=hi -F ref_text="…" -F audio=@clip.wav`.

## Use a voice

```json
{ "type": "synthesize", "call_id": "c1", "text_id": "t1",
  "text": "नमस्ते, मैं प्रिया बोल रही हूँ बजाज फाइनेंस से।", "voice_id": "priya" }
```

The default alias (used when `voice_id` is omitted) is `settings.voices.default_voice`
(`priya`). If no matching voice exists, the gateway falls back to the backend's
`default` voice.

`.wav`/`.json` artifacts are git-ignored (derived); rebuild them on each box. The
`voices/` dir is a **shared volume** the sglang backend also mounts (read-only) so
local reference paths resolve there.
