FROM nvidia/cuda:13.0.0-devel-ubuntu22.04


RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY pyproject.toml .
RUN python -m pip install --no-cache-dir flwr==1.28.0
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
    && python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]