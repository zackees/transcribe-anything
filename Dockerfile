# GPU-accelerated transcribe-anything Docker image
# Prerequisites on HOST:
#   - nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#   - Run with: docker run --gpus all transcribe-anything <args>
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends build-essential curl dos2unix && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy source code and prepare entrypoint
COPY . .
RUN chmod +x entrypoint.sh && dos2unix entrypoint.sh

# Install transcribe-anything in editable mode
RUN pip install -e .

# Pre-initialize the insane whisper environment (downloads dependencies)
# NOTE: GPU is not available during build, so shared library checks are skipped
RUN transcribe-anything-init-insane

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
