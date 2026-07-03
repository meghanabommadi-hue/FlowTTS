# Fish Audio S2 Pro inference backend — sglang-omni server.
#
# Serves `fishaudio/s2-pro` (Dual-AR + EVA-GAN/RVQ codec) over the OpenAI-compatible
# `POST /v1/audio/speech` endpoint with continuous batching, paged KV cache, CUDA-graph
# replay and RadixAttention prefix caching. Target: single NVIDIA H200 (Hopper, CC 9.0).
#
# We base on the OFFICIAL sglang-omni image (built with uv), which already ships the
# `sgl-omni` CLI + torch/flash-attn/CUDA + the fishaudio_s2_pro model deps with their
# dependency conflicts (e.g. protobuf: descript-audiotools vs s3prl/onnxruntime) already
# resolved. Building sglang-omni from source on a bare CUDA image re-triggers those
# conflicts — don't. If you need CUDA 12, use a `-cu12`/`-cu129` tag instead of :dev.
#
# ⚠ LICENSE: fishaudio/s2-pro weights are under the **Fish Audio Research License**
#   (non-commercial). Commercial use requires a separate license from Fish Audio
#   (business@fish.audio). You are responsible for obtaining it.
#
# Build:  docker build -f docker/fish_s2pro.Dockerfile -t fish-s2pro:latest .
# (usually via docker-compose.yml at the repo root)

ARG BASE_IMAGE=lmsysorg/sglang-omni:dev
FROM ${BASE_IMAGE}

# Small additions only — the heavy stack is already in the base image.
#   curl      → container healthcheck + fetching the S2 Pro pipeline config
#   soundfile → lets the backend decode base64 (data-URI) reference audio
# Use uv (present in the base image, per sglang-omni's install docs); fall back to pip.
RUN (command -v curl >/dev/null 2>&1) || (apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*)
RUN uv pip install --system --no-cache soundfile || pip install --no-cache-dir soundfile

# The S2 Pro pipeline config lives in the sglang-omni repo. Fetch just that one file so
# we don't depend on the image's internal layout. Pin S2PRO_CONFIG_REF to the tag/commit
# that matches your base image for reproducibility.
ARG S2PRO_CONFIG_REF=main
ARG S2PRO_CONFIG_URL=https://raw.githubusercontent.com/sgl-project/sglang-omni/${S2PRO_CONFIG_REF}/examples/configs/s2pro_tts.yaml
RUN mkdir -p /opt/fish && curl -fsSL "${S2PRO_CONFIG_URL}" -o /opt/fish/s2pro_tts.yaml \
    && echo "[build] fetched s2pro_tts.yaml:" && head -n 40 /opt/fish/s2pro_tts.yaml

ENV HF_HOME=/root/.cache/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 \
    PYTHONUNBUFFERED=1

# Serving knobs (override in docker-compose.yml):
#   FISH_MODEL             HF repo id or local path to the weights
#   FISH_CONFIG            sglang-omni pipeline config for S2 Pro
#   TTS_BATCH_MAX_ITEMS    server-side batch cap (throughput)
#   MEM_FRACTION           static GPU memory fraction for the KV cache (e.g. 0.85)
#   PORT                   HTTP port
ENV FISH_MODEL=fishaudio/s2-pro \
    FISH_CONFIG=/opt/fish/s2pro_tts.yaml \
    TTS_BATCH_MAX_ITEMS=32 \
    MEM_FRACTION= \
    PORT=8000

EXPOSE 8000

# Auto-downloads the weights into HF_HOME (mounted volume) on first run. `sgl-omni serve`
# reads local reference audio paths directly; the huggingface.co domains are allowlisted
# so http(s) reference URLs also work.
CMD sgl-omni serve \
      --model-path "${FISH_MODEL}" \
      --config "${FISH_CONFIG}" \
      --tts-batch-max-items "${TTS_BATCH_MAX_ITEMS}" \
      ${MEM_FRACTION:+--mem-fraction-static "${MEM_FRACTION}"} \
      --allowed-media-domain huggingface.co \
      --allowed-media-domain cas-bridge.xethub.hf.co \
      --host 0.0.0.0 --port "${PORT}"
