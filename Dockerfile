########################################################
# This is a community contributed Dockerfile!
# If you find any issues with it, please please open an issue on the GitHub repository:
# https://github.com/zackees/transcribe-anything/issues/new

# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set the default shell for RUN commands to bash
SHELL ["/bin/bash", "-c"]

# Set the LD_LIBRARY_PATH environment variable to include the system library directory
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Set the working directory in the container to /workspace
WORKDIR /workspace

# Update the package lists and install wget
RUN apt-get update && apt-get install -y wget

# Download the CUDA keyring package from NVIDIA and install it to enable the CUDA repository
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb

# Update the package lists again (now including the new repository) and apply all available upgrades
RUN apt-get -y update && apt-get -y upgrade

# Install the required packages:
# - build-essential, dkms: tools for compiling software and managing kernel modules
# - libnccl2, libnccl-dev: libraries for GPU communication
# - cudnn9-cuda-12, cuda-toolkit-12-6: libraries and toolkit for CUDA development
# - libcusparselt0, libcusparselt-dev: libraries for sparse operations on GPU
# - python3.11, python3-pip, python3-venv: Python interpreter, package manager, and virtual environment tools
RUN apt-get -y install build-essential dkms libnccl2 libnccl-dev cudnn9-cuda-12 cuda-toolkit-12-6 libcusparselt0 libcusparselt-dev python3.11 python3-pip git python3-venv

# Create a Python virtual environment in the .venv directory
RUN python3 -m venv .venv

# Activate the virtual environment (note: activation only affects the current RUN layer and does not persist in subsequent layers)
RUN source /workspace/.venv/bin/activate

# Upgrade pip, setuptools, and wheel, and install the transcribe-anything package within the virtual environment
RUN pip install --upgrade pip setuptools wheel transcribe-anything

# Set the default command to run when the container starts (launches a bash shell)
CMD ["bash"]
