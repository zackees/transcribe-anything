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

# Validate CUDA shared libraries
if python3 check_linux_shared_libraries.py
then
    echo "✓ check_linux_shared_libraries.py"
else
    echo "✗ check_linux_shared_libraries.py"
    exit 1
fi

# Exit early if only checking libraries
if [[ "$1" == "--only-check-shared-libs" ]]; then
    echo "✓ --only-check-shared-libs"
    exit 0
fi

# Launch transcribe-anything with GPU acceleration
transcribe-anything "$@"
