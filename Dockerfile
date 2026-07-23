# VoxLens — Docker image
#
# Two usage modes:
#   1. NIM backend (Docker-friendly, no GPU passthrough needed):
#      docker build -t voxlens .
#      docker run -p 8000:8000 -e MODEL_BACKEND=nim -e NVIDIA_API_KEY=... voxlens
#
#   2. Local backend (requires GPU passthrough + LiteRT-LM support):
#      Not recommended in Docker due to webcam/mic passthrough complexity.
#      See README for details.
#
# Base image: CUDA 12.x + cuDNN runtime.
# TTS uses kokoro-onnx (CPU-only), no GPU needed for audio.

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

LABEL description="VoxLens — Real-time voice + vision AI"
LABEL maintainer="Shashank R"

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy dependency files first for layer caching
COPY src/pyproject.toml src/uv.lock ./

# Install Python dependencies
RUN uv sync --python 3.12 --no-dev

# Copy application code
COPY src/ ./

# Default: NIM backend (no GPU needed)
ENV MODEL_BACKEND=nim
ENV NVIDIA_API_KEY=""
ENV NIM_MODEL="nvidia/nemotron-nano-12b-v2-vl"
ENV PORT=8000

# For local backend (needs GPU passthrough, see README):
# ENV MODEL_BACKEND=local

EXPOSE 8000

CMD ["uv", "run", "--python", "3.12", "python", "server.py"]
