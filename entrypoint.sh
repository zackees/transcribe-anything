#!/bin/bash

# GPU-accelerated transcribe-anything launcher
# Configures CUDA library paths and validates shared libraries

# Configure CUDA library paths for conda-installed NVIDIA packages
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/cusparselt/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_cupti/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cusparse/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cufft/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/curand/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib/:${LD_LIBRARY_PATH}

# Validate CUDA shared libraries (only if GPU device is present)
if [ -e /dev/nvidia0 ]; then
    if python3 check_linux_shared_libraries.py; then
        echo "OK: CUDA shared libraries validated"
    else
        echo "WARNING: CUDA library check failed. GPU acceleration may not work."
        echo "Ensure container was started with: docker run --gpus all ..."
    fi
else
    echo "NOTE: No GPU device detected (/dev/nvidia0). Running in CPU-only mode."
fi

# Exit early if only checking libraries
if [[ "$1" == "--only-check-shared-libs" ]]; then
    echo "OK: --only-check-shared-libs"
    exit 0
fi

# Launch transcribe-anything with GPU acceleration
transcribe-anything "$@"
