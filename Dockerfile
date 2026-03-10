# Chatterbox Turbo Worker for RunPod Serverless
# Text-to-Speech with voice cloning and paralinguistic tags
#
# Based on: https://github.com/geronimi73/runpod_chatterbox

# Use RunPod's pre-built PyTorch image (has CUDA, Python 3.11, PyTorch ready)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Chatterbox TTS without deps to avoid PyTorch version conflicts
RUN pip install --no-cache-dir --no-deps chatterbox-tts

# Install chatterbox dependencies (from reference: github.com/geronimi73/runpod_chatterbox)
# These are installed without version pins to work with the base image's PyTorch
RUN pip install --no-cache-dir \
    conformer \
    s3tokenizer \
    librosa \
    resemble-perth \
    huggingface_hub \
    safetensors \
    transformers \
    einops \
    soundfile \
    scipy \
    omegaconf \
    pyloudnorm \
    torchaudio \
    diffusers

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# Copy handler
COPY handler.py /app/handler.py

# Pre-download model during build
# The model is public but chatterbox defaults to token=True, so we download directly
RUN python -c "\
from huggingface_hub import snapshot_download; \
print('Downloading Chatterbox Turbo model...'); \
snapshot_download(repo_id='ResembleAI/chatterbox-turbo', token=False, allow_patterns=['*.safetensors', '*.json', '*.txt', '*.pt', '*.model']); \
print('Model downloaded successfully')"

# Start handler
CMD ["python", "-u", "/app/handler.py"]
