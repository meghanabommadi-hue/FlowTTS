#!/usr/bin/env bash
# Container entrypoint for the FlowTTS gateway (CPU-only proxy to the Fish S2 Pro backend).
#
# Modes (first arg):
#   serve  (default) — ensure voices exist, then start the WS server
#   setup            — build all voices from the manifest, then exit
#   clone  ...       — run the voice-clone CLI (passes remaining args through)
#   bash             — drop into a shell
#   <anything else>  — exec verbatim
#
# The TTS MODEL is NOT downloaded here — it lives in the separate fish-s2pro
# (sglang-omni) service. This gateway only needs voice references + config.
set -euo pipefail
cd /root/FlowTTS

ensure_voices() {
    if ! ls voices/*.json >/dev/null 2>&1; then
        echo "[entrypoint] no voice manifests found — building from voices/manifest.json (needs ref_text; no GPU)..."
        python -m flowtts.voices.clone --build-all --manifest voices/manifest.json \
            || echo "[entrypoint] voice build skipped/failed — server falls back to the backend 'default' voice."
    else
        echo "[entrypoint] voices present: $(ls voices/*.json 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
    fi
}

MODE="${1:-serve}"
case "$MODE" in
    serve)
        ensure_voices
        echo "[entrypoint] backend=${FLOWTTS_FISH__BACKEND_URL:-http://fish-s2pro:8000}"
        echo "[entrypoint] starting gateway: PORTS=${PORTS:-1} BASE_PORT=${BASE_PORT:-8080} CTRL_PORT=${CTRL_PORT:-8764}"
        exec python -m flowtts.server \
            --ports "${PORTS:-1}" --base-port "${BASE_PORT:-8080}" \
            ${CTRL_PORT:+--ctrl-port "$CTRL_PORT"} \
            ${SAVE_AUDIO:+--save-audio "$SAVE_AUDIO"}
        ;;
    setup)
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
