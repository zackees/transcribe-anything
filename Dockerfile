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

COPY . .
RUN chmod +x entrypoint.sh && dos2unix entrypoint.sh
RUN pip install -e .

# Now install the lengthy list of heavy dependencies so startup is reasonable.
RUN /bin/bash /app/entrypoint.sh --only-check-shared-libs && transcribe-anything-init-insane

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]


