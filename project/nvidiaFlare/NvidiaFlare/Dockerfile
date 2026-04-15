FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt . 
RUN python -m pip install -r requirements.txt

ARG NVF_BRANCH=main
RUN python -m pip install -U setuptools
RUN python -m pip install nvflare~=2.7.0rc

ENV HOME=/tmp
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache
RUN mkdir -p /tmp/torch_cache && chmod -R 777 /tmp/torch_cache

WORKDIR /workspace/
RUN git clone https://github.com/NVIDIA/NVFlare.git --branch ${NVF_BRANCH} --single-branch NVFlare

