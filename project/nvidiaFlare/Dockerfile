FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

COPY ./MedNIST /app/MedNIST

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt . 
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN python -m pip install --no-cache-dir nvflare==2.7.2

ENV HOME=/tmp
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache
RUN mkdir -p /tmp/torch_cache && chmod -R 777 /tmp/torch_cache

WORKDIR /workspace/

