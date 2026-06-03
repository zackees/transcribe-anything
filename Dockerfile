# GPU-accelerated transcribe-anything Docker image
# Prerequisites on HOST:
#   - nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#   - Run with: docker run --gpus all transcribe-anything <args>
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

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
RUN pip install --no-cache-dir -e .

# Keep the default image lean. Backend environments are built on first use unless
# PREBUILD_BACKENDS is set to one of: insane, insane-flash, both, all.
# NOTE: GPU is not available during build, so runtime CUDA checks are skipped.
ARG PREBUILD_BACKENDS=none
RUN set -eux; \
    prebuild_insane() { UV_CACHE_DIR=/tmp/uv-cache transcribe-anything-init-insane; }; \
    prebuild_insane_flash() { UV_CACHE_DIR=/tmp/uv-cache transcribe-anything-init-insane-flash; }; \
    case "$PREBUILD_BACKENDS" in \
        none|"") ;; \
        insane) prebuild_insane ;; \
        insane-flash) prebuild_insane_flash ;; \
        insane,insane-flash|insane-flash,insane|both|all) prebuild_insane; prebuild_insane_flash ;; \
        *) echo "Unsupported PREBUILD_BACKENDS=$PREBUILD_BACKENDS"; exit 1 ;; \
    esac; \
    rm -rf /root/.cache/pip /root/.cache/uv /tmp/uv-cache /tmp/*

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
