FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85

ARG DEBIAN_FRONTEND=noninteractive
ARG STOCKFISH_REVISION=cb3d4ee9b47d0c5aae855b12379378ea1439675c

ENV TRAINING_CONTAINER_BASE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        ninja-build \
        python3.10 \
        python3.10-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir uv==0.11.14

WORKDIR /workspace
COPY py/requirements-training.lock /workspace/py/requirements-training.lock
RUN uv pip install --system --require-hashes --torch-backend cu128 \
    --requirements /workspace/py/requirements-training.lock

RUN git clone --filter=blob:none https://github.com/official-stockfish/Stockfish.git /opt/stockfish \
    && git -C /opt/stockfish checkout "${STOCKFISH_REVISION}" \
    && make --directory /opt/stockfish/src --jobs profile-build ARCH=x86-64-avx2 \
    && install /opt/stockfish/src/stockfish /usr/local/bin/stockfish-18

COPY . /workspace
RUN cmake -S /workspace/cpp -B /workspace/cpp/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE=/usr/bin/python3.10 \
    && cmake --build /workspace/cpp/build --parallel

WORKDIR /workspace/py
