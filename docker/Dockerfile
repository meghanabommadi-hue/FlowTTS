# FlowTTS gateway — CPU-only WebSocket/control server.
#
# Proxies to the Fish Audio S2 Pro sglang backend (docker/fish_s2pro.Dockerfile).
# No torch, no CUDA — the gateway just builds requests and forwards streamed PCM.
# The repo lives at /root/FlowTTS so the code's `Path.home()/"FlowTTS"` defaults
# (sample_files/, voices/, cached_data/) resolve correctly inside the container.
#
# Build:  docker build -f docker/Dockerfile -t flowtts-gateway .
# (usually via docker-compose.yml at the repo root)

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg → pydub decodes mp3/m4a/etc reference clips; libsndfile1 → soundfile WAV I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Python deps (CPU-only set).
COPY requirements.txt /root/FlowTTS/requirements.txt
RUN pip install --no-cache-dir -r /root/FlowTTS/requirements.txt

# App code.
ENV HOME=/root
WORKDIR /root/FlowTTS
COPY . /root/FlowTTS
RUN chmod +x /root/FlowTTS/docker/entrypoint.sh

ENV PYTHONPATH=/root/FlowTTS \
    PYTHONUNBUFFERED=1

# WS base port + control API port (override with PORTS/BASE_PORT/CTRL_PORT env).
EXPOSE 8080 8764

ENTRYPOINT ["/root/FlowTTS/docker/entrypoint.sh"]
CMD ["serve"]
