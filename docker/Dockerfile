# FlowTTS (OmniVoice) — GPU container.
#
# Works on any NVIDIA GPU with compute capability >= 8.0 (Ada RTX 6000 = 8.9, Hopper H200 = 9.0).
# The repo lives at /root/FlowTTS so the code's `Path.home()/"FlowTTS"` defaults
# (sample_files/, voices/, cached_data/) resolve correctly inside the container.
#
# Build:  docker build -f docker/Dockerfile -t flowtts-omnivoice .
# (or just use docker-compose.yml at the repo root)

ARG BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential git curl ca-certificates \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv on PATH → clean `python`/`pip`, no PEP-668 hassle.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install the CUDA-matched PyTorch FIRST so requirements.txt can't pull CPU wheels.
# cu128 wheels run on Ada (sm_89) and Hopper (sm_90). Override via build args if needed.
ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir \
        torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} --index-url ${TORCH_INDEX}

# Python deps (torch already satisfied → not re-downloaded).
COPY requirements.txt /root/FlowTTS/requirements.txt
RUN pip install --no-cache-dir -r /root/FlowTTS/requirements.txt

# App code.
ENV HOME=/root
WORKDIR /root/FlowTTS
COPY . /root/FlowTTS
RUN chmod +x /root/FlowTTS/docker/entrypoint.sh

ENV PYTHONPATH=/root/FlowTTS \
    PYTHONUNBUFFERED=1 \
    HF_XET_HIGH_PERFORMANCE=1

# WS base port + control API port (override with PORTS/BASE_PORT/CTRL_PORT env).
EXPOSE 8080 8764

ENTRYPOINT ["/root/FlowTTS/docker/entrypoint.sh"]
CMD ["serve"]
