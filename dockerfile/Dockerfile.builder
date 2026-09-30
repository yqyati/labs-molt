# CUDA 13 builder image for the molt source extensions.
# Keep these layers aligned with dockerfile/Dockerfile through the Dion/tooling
# layer. TransformerEngine, flash-attn, mamba and DeepEP are intentionally built
# later inside a high-memory rlaunch environment.
ARG BASE_IMAGE=registry.h.pjlab.org.cn/ailab-ai4sgi-ailab_ai4sgi_pool/yqy:cuda13.0.2-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# Passed only at build time; do not bake the lab proxy into the image.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    RDMA_CORE_HOME=/opt/rdma-core/build \
    CUDA_HOME=/usr/local/cuda

ARG SITE=/usr/local/lib/python3.12/dist-packages

ENV NCCL_HOME=${SITE}/nvidia/nccl \
    NVTE_NCCL_HOME=${SITE}/nvidia/nccl \
    CUDNN_HOME=${SITE}/nvidia/cudnn \
    CUDNN_PATH=${SITE}/nvidia/cudnn

# Match molt's system toolchain layer exactly.
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && echo Etc/UTC > /etc/timezone && \
    echo 'tzdata tzdata/Areas select Etc' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/Etc select UTC' | debconf-set-selections && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python-is-python3 \
        build-essential gdb git gosu libomp-dev sudo \
        libopenmpi-dev libnvidia-ml-dev ca-certificates \
        pkg-config cython3 cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Match molt's cu130 torch trio exactly.
RUN python -m pip install --ignore-installed pip setuptools wheel && \
    python -m pip install \
        torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu130

# Match molt's constraints and source-build dependencies. No CUDA extensions are
# compiled in this image; they are compiled from the mounted source in rlaunch.
RUN T=$(python -c "import torch; print(torch.__version__)") && \
    TV=$(python -c "import torchvision; print(torchvision.__version__)") && \
    TA=$(python -c "import torchaudio; print(torchaudio.__version__)") && \
    printf 'torch==%s\ntorchvision==%s\ntorchaudio==%s\n' "$T" "$TV" "$TA" > /opt/torch-pin.txt && \
    python -m pip install -c /opt/torch-pin.txt \
        pybind11 setuptools_scm ninja cmake pydantic onnxscript onnx importlib_metadata && \
    (python -m pip uninstall -y xgboost pynvml opencv-python-headless || true) && \
    python -m pip install -c /opt/torch-pin.txt \
        "dion @ git+https://github.com/microsoft/dion.git"

WORKDIR /workspace
