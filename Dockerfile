# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY marl_uav ./marl_uav
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt && \
    python -m pip install matplotlib tensorboard PyFlyt && \
    python -m pip install -e .

COPY docker/train_in_container.sh /usr/local/bin/train_in_container.sh
RUN chmod +x /usr/local/bin/train_in_container.sh

RUN mkdir -p /workspace/results /workspace/checkpoints /workspace/save_result

ENTRYPOINT ["/usr/local/bin/train_in_container.sh"]
CMD ["train"]
