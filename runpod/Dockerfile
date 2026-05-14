# RunPod Serverless worker for transcribe-anything (--device insane).
# Build context = repo root, e.g.
#   docker build -f runpod/Dockerfile -t <registry>/transcribe-anything-runpod:latest .
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends build-essential curl ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r runpod/requirements.txt

# Note: we intentionally do NOT pre-initialize the insane backend here.
# transcribe-anything-init-insane requires a working CUDA stack to complete,
# which is unavailable during the build (no GPU on the builder). The env is
# initialized at handler import time on the worker instead — see handler.py.
# Trade-off: first request on a cold worker pays ~30-60 sec env-install cost.

CMD ["python", "-u", "runpod/handler.py"]
