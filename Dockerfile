# GPU-accelerated transcribe-anything Docker image
# Uses PyTorch CUDA 12.6 runtime for optimal GPU performance
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update -y
RUN apt-get install -y build-essential curl dos2unix

# Configure NVIDIA Docker runtime for GPU access
RUN distribution=$(. /etc/os-release;echo  $ID$VERSION_ID)  && \
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -  && \
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
 
RUN apt-get update -y
RUN apt-get install -y nvidia-container-toolkit
RUN mkdir -p /etc/docker
RUN nvidia-ctk runtime configure --runtime=docker

# Copy source code and prepare entrypoint
COPY . .
RUN chmod +x entrypoint.sh && dos2unix entrypoint.sh

# Install transcribe-anything in editable mode
RUN pip install -e .

# Pre-initialize GPU dependencies to reduce startup time
RUN /bin/bash /app/entrypoint.sh --only-check-shared-libs && transcribe-anything-init-insane

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
