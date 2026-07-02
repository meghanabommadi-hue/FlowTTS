#!/usr/bin/env bash
# Container entrypoint for FlowTTS (OmniVoice).
#
# Modes (first arg):
#   serve  (default) — ensure model + voices, then start the WS server
#   setup            — download model + build all voices, then exit
#   clone  ...       — run the voice-clone CLI (passes remaining args through)
#   bash             — drop into a shell
#   <anything else>  — exec verbatim
set -euo pipefail
cd /root/FlowTTS

# An empty HF_TOKEN (the "" default Compose injects) makes huggingface_hub send an
# illegal "Authorization: Bearer " header. Unset it so public repos use anonymous
# access; a real token (gated repos) is left intact.
if [ -z "${HF_TOKEN:-}" ]; then unset HF_TOKEN || true; fi

ensure_model() {
    echo "[entrypoint] ensuring OmniVoice weights are cached (HF cache volume)..."
    python -m flowtts.setup.download_models
}

ensure_voices() {
    if ! ls voices/*.npz >/dev/null 2>&1; then
        echo "[entrypoint] no voice npz found — building from sample_files/ (one-time; may download Whisper for auto-transcription)..."
        python -m flowtts.voices.clone --build-all --manifest voices/manifest.json \
            || python -m flowtts.voices.clone --build-all \
            || echo "[entrypoint] voice build failed — server will fall back to OmniVoice auto-voice."
    else
        echo "[entrypoint] voices present: $(ls voices/*.npz 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
    fi
}

MODE="${1:-serve}"
case "$MODE" in
    serve)
        ensure_model
        ensure_voices
        echo "[entrypoint] starting server: PORTS=${PORTS:-1} BASE_PORT=${BASE_PORT:-8080} CTRL_PORT=${CTRL_PORT:-8764}"
        exec python -m flowtts.server \
            --ports "${PORTS:-1}" --base-port "${BASE_PORT:-8080}" \
            ${CTRL_PORT:+--ctrl-port "$CTRL_PORT"} \
            ${SAVE_AUDIO:+--save-audio "$SAVE_AUDIO"}
        ;;
    setup)
        ensure_model
        python -m flowtts.voices.clone --build-all --manifest voices/manifest.json
        python -m flowtts.voices.clone --list
        ;;
    clone)
        shift
        exec python -m flowtts.voices.clone "$@"
        ;;
    bash|sh)
        exec /bin/bash
        ;;
    *)
        exec "$@"
        ;;
esac
