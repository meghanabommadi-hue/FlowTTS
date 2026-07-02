# Voices — OmniVoice voice-clone artifacts

Each `<alias>.npz` here is a precomputed `VoiceClonePrompt` (the Higgs-codec token
grid of a reference clip + its transcript + loudness). The server loads them all
at startup and addresses them by alias via the WebSocket `voice_id` field.

## Build voices

```bash
# From every clip in sample_files/ (stem = alias), applying voices/manifest.json overrides:
python -m flowtts.voices.clone --build-all --manifest voices/manifest.json

# A single voice (ref_text optional — auto-transcribed with Whisper if omitted):
python -m flowtts.voices.clone --add priya --ref-audio sample_files/priya.wav \
    --ref-text "नमस्ते, मैं प्रिया बोल रही हूँ।"

# List installed voices
python -m flowtts.voices.clone --list
```

## Use a voice

```json
{ "type": "synthesize", "call_id": "c1", "text_id": "t1",
  "text": "नमस्ते, मैं प्रिया बोल रही हूँ बजाज फाइनेंस से।", "voice_id": "priya" }
```

The default alias (used when `voice_id` is omitted) is `settings.voices.default_voice`
(`priya`). If no matching npz exists, the server falls back to OmniVoice's auto voice.

`.npz` files are git-ignored (they're derived artifacts); rebuild them on each box.
