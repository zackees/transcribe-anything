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

# Pre-initialize the insane backend so cold starts don't pay the install cost.
# GPU is unavailable during build; the init script skips shared-library checks.
RUN transcribe-anything-init-insane || true

CMD ["python", "-u", "runpod/handler.py"]
