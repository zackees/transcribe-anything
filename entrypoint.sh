#!/bin/bash

# GPU-accelerated transcribe-anything launcher
# Configures CUDA library paths and validates shared libraries

# Configure CUDA library paths for the NVIDIA base image and prebuilt backend
# environments. The torch CUDA wheels install most runtime libraries under the
# backend venv's site-packages/nvidia/*/lib directories.
prepend_ld_path() {
    if [ -d "$1" ]; then
        export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH}"
    fi
}

prepend_ld_path /usr/local/cuda/lib64
for site_packages in /app/src/transcribe_anything/venv/*/.venv/lib/python*/site-packages; do
    if [ -d "$site_packages" ]; then
        prepend_ld_path "$site_packages/torch/lib"
        for lib_dir in "$site_packages"/nvidia/*/lib "$site_packages"/nvidia/*/lib64 "$site_packages"/cusparselt/lib; do
            prepend_ld_path "$lib_dir"
        done
    fi
done

# Validate CUDA shared libraries (only if GPU access is present)
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    if python3 check_linux_shared_libraries.py; then
        echo "OK: CUDA shared libraries validated"
    else
        echo "WARNING: CUDA library check failed. GPU acceleration may not work."
        echo "Ensure container was started with: docker run --gpus all ..."
    fi
else
    echo "NOTE: No NVIDIA GPU detected by nvidia-smi. Running in CPU-only mode."
fi

# Exit early if only checking libraries
if [[ "$1" == "--only-check-shared-libs" ]]; then
    echo "OK: --only-check-shared-libs"
    exit 0
fi

# Launch transcribe-anything with GPU acceleration
transcribe-anything "$@"
