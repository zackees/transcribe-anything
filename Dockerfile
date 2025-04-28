# pick the CUDA version your host GPU driver supports
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update -y

# install any extra system deps
RUN apt-get install -y build-essential
RUN apt-get install -y curl dos2unix


RUN distribution=$(. /etc/os-release;echo  $ID$VERSION_ID)  && \
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -  && \
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
 
RUN apt-get update -y
RUN apt-get install -y nvidia-container-toolkit

RUN mkdir -p /etc/docker

RUN nvidia-ctk runtime configure --runtime=docker

RUN pip install uv
RUN uv venv
# RUN uv pip install transcribe-anything
# # Force install ffmpeg
# RUN uv run static_ffmpeg -version

# RUN pip install transcribe-anything


# transcribe-anything-init-insane

# Install the transcriber.
ENV VERSION=3.0.9
RUN uv pip install transcribe-anything>=${VERSION} \
    || uv pip install transcribe-anything>=${VERSION} \
    || uv pip install transcribe-anything>=${VERSION}

RUN uv run static_ffmpeg -version

COPY ./check_linux_shared_libraries.py check_linux_shared_libraries.py
COPY ./entrypoint.sh /app/entrypoint.sh
RUN chmod +x entrypoint.sh && dos2unix entrypoint.sh


COPY . .

RUN chmod +x entrypoint.sh && dos2unix entrypoint.sh

RUN uv pip install -e .

RUN /bin/bash /app/entrypoint.sh --only-check-shared-libs && uv run -m transcribe_anything.cli_init_insane



ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["--help"]


