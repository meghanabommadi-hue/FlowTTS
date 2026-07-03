# Fish Audio S2 Pro inference backend — sglang-omni server.
#
# Serves `fishaudio/s2-pro` (Dual-AR + EVA-GAN/RVQ codec) over the OpenAI-compatible
# `POST /v1/audio/speech` endpoint with continuous batching, paged KV cache, CUDA-graph
# replay and RadixAttention prefix caching. Target: single NVIDIA H200 (Hopper, CC 9.0).
#
# ⚠ LICENSE: fishaudio/s2-pro weights are under the **Fish Audio Research License**
#   (non-commercial). Commercial use requires a separate license from Fish Audio
#   (business@fish.audio). You are responsible for obtaining it.
#
# Build:  docker build -f docker/fish_s2pro.Dockerfile -t fish-s2pro:latest .
# (usually via docker-compose.yml at the repo root)

ARG BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential git curl ca-certificates \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# sglang-omni provides the `sgl-omni` CLI and the fishaudio_s2_pro model plugin.
# Pin SGLANG_OMNI_REF to a known-good tag/commit before production (main tracks HEAD).
# Cloning the repo (not just pip-installing the wheel) gives us examples/configs/.
ARG SGLANG_OMNI_REPO=https://github.com/sgl-project/sglang-omni.git
ARG SGLANG_OMNI_REF=main
RUN git clone --depth 1 --branch ${SGLANG_OMNI_REF} ${SGLANG_OMNI_REPO} /opt/sglang-omni
WORKDIR /opt/sglang-omni
# Installs sglang-omni + its sglang/torch CUDA deps. soundfile decodes base64
# (data-URI) reference audio when the gateway uses reference_mode=base64.
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir soundfile

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
    FISH_CONFIG=/opt/sglang-omni/examples/configs/s2pro_tts.yaml \
    TTS_BATCH_MAX_ITEMS=32 \
    MEM_FRACTION= \
    PORT=8000

EXPOSE 8000

# Auto-downloads the weights into HF_HOME (mounted volume) on first run.
# `sgl-omni serve` reads local reference audio paths directly; the huggingface.co
# domains are allowlisted so http(s) reference URLs also work.
CMD sgl-omni serve \
      --model-path "${FISH_MODEL}" \
      --config "${FISH_CONFIG}" \
      --tts-batch-max-items "${TTS_BATCH_MAX_ITEMS}" \
      ${MEM_FRACTION:+--mem-fraction-static "${MEM_FRACTION}"} \
      --allowed-media-domain huggingface.co \
      --allowed-media-domain cas-bridge.xethub.hf.co \
      --host 0.0.0.0 --port "${PORT}"
