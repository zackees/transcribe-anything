########################################################
# This is a community contributed Dockerfile!
# If you find any issues with it, please please open an issue on the GitHub repository:
#   https://github.com/zackees/transcribe-anything/issues/new


FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# BASICALLY update and upgrade the image dependencies
RUN apt-get update && apt-get -y upgrade && apt-get clean

# Check NVIDIA drivers with nvidia-smi
RUN nvidia-smi

# Check CUDA drivers with nvcc
RUN nvcc --version

# Add /usr/lib/x86_64-linux-gnu to LD_LIBRARY_PATH to fix the libcudart.so error
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install cuDNN 9.x for CUDA 12 (in this precompiled image, I think version 8 is installed) to fix the cuDNN9 library not found error
RUN apt-get update && apt-get -y install cudnn9-cuda-12 && apt-get clean

# Now install cuSPARSELt, which is required to fix the error importing the libcusparselt.so library
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install libcusparselt0 libcusparselt-dev && apt-get clean

# Check Python version
RUN python3 --version

# Now install transcribe-anything using pip
RUN pip install transcribe-anything

# The container will not execute any command on startup;
# you can enter it with "docker run -it <image_name> --gpus all /bin/bash" and manually run command like this :
# transcribe-anything https://www.youtube.com/watch?v=dQw4w9WgXcQ --device insane
CMD ["bash"]