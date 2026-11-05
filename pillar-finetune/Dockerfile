FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    NODE_TLS_REJECT_UNAUTHORIZED=0 \
    TZ=America/New_York

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get update -qq \
    && apt-get install -qqy --no-install-recommends \
        apt-utils \
        man-db \
        tzdata \
        jq \
        vim \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libgl1-mesa-glx \
        libgdcm-tools \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

WORKDIR /workspace
