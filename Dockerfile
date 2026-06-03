# GPU-accelerated transcribe-anything Docker image
# Prerequisites on HOST:
#   - nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#   - Run with: docker run --gpus all transcribe-anything <args>
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# In Docker, both --device insane and --device insane-flash use one prebuilt
# FlashAttention-capable backend env to avoid duplicating the full torch stack.
ENV TRANSCRIBE_ANYTHING_SHARED_INSANE_BACKEND=flash

WORKDIR /app

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends build-essential ca-certificates curl dos2unix python3 python3-pip python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only the files needed to install and prebuild backend environments.
COPY pyproject.toml README.md ./
COPY src ./src
COPY check_linux_shared_libraries.py ./

# Install transcribe-anything in editable mode
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -e .

# Prebuild both CUDA insane backends by default using one shared flash-capable
# env. Use PREBUILD_BACKENDS=none for the smallest image, or one of:
# insane, insane-flash, both, all.
# NOTE: GPU is not available during build, so runtime CUDA checks are skipped.
ARG PREBUILD_BACKENDS=both
RUN set -eux; \
    prebuild_shared_insane() { UV_CACHE_DIR=/tmp/uv-cache transcribe-anything-init-insane-flash; }; \
    case "$PREBUILD_BACKENDS" in \
        none|"") ;; \
        insane|insane-flash|insane,insane-flash|insane-flash,insane|both|all) prebuild_shared_insane ;; \
        *) echo "Unsupported PREBUILD_BACKENDS=$PREBUILD_BACKENDS"; exit 1 ;; \
    esac; \
    rm -rf /root/.cache/pip /root/.cache/uv /tmp/uv-cache /tmp/*

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh && dos2unix entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
