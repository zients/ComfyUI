ARG PYTHON_IMAGE=python:3.13-slim-bookworm

FROM ${PYTHON_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_PACKAGES="torch torchvision torchaudio"
ARG UID=1000
ARG GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${GID}" comfy \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash comfy

WORKDIR /opt/ComfyUI

COPY requirements.txt /tmp/comfyui-requirements.txt
# hadolint ignore=DL3013
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" ${TORCH_PACKAGES} \
    && python -m pip install -r /tmp/comfyui-requirements.txt \
    && rm /tmp/comfyui-requirements.txt

COPY --chown=comfy:comfy . /opt/ComfyUI

RUN mkdir -p /data /mnt/comfyui/models /mnt/comfyui/custom_nodes \
    && chown -R comfy:comfy /data /mnt/comfyui /opt/ComfyUI \
    && chmod +x /opt/ComfyUI/scripts/container-entrypoint.sh

USER comfy

EXPOSE 8188

ENTRYPOINT ["/opt/ComfyUI/scripts/container-entrypoint.sh"]
