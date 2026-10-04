# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
FROM python:3.12-slim

# ffmpeg: audio extraction from any container format, and ffprobe for the
# duration and the stream list.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HOME=/opt/models

WORKDIR /src
COPY pyproject.toml ./

# The dependencies and the model weights are installed before the source is
# copied, so editing app/ costs one small layer instead of re-downloading the
# weights. The stub package exists only so that the project's dependencies can
# be resolved from pyproject before app/ is there.
#
# faster-whisper runs on CTranslate2, whose CUDA runtime comes from the
# nvidia-cublas-cu12 and nvidia-cudnn-cu12 wheels in the "cuda" extra; no
# torch, no system CUDA. ctranslate2 does not declare those wheels, so the
# extra is what actually puts them in the image, and LD_LIBRARY_PATH below is
# what lets CTranslate2 dlopen them.
#
# The dev extra is installed so the image can run its own test suite:
#   podman run --rm --network none -v .:/src:ro -w /src tachygraf:latest pytest
RUN mkdir -p app && touch app/__init__.py \
    && uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python ".[dev,cuda]"

# The wheels put their shared objects under site-packages/nvidia/*/lib, which
# is on no default search path. CTranslate2 dlopens them by bare soname, so
# without this the GPU path dies at the first inference and the stage falls
# back to the CPU for a reason that is a packaging bug, not a busy card.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib

# Bake the ASR weights into the image. The container has no network at run
# time, so a model that is not here is a model we cannot use.
#
# The target directory is explicit and world-readable for the reason glosator
# learned the hard way: with --userns=keep-id $HOME is not the one that existed
# at build time, and weights baked under it sit behind a permission error.
RUN python -c "\
from faster_whisper.utils import download_model; \
download_model('large-v3', cache_dir='/opt/models/whisper'); \
download_model('large-v3-turbo', cache_dir='/opt/models/whisper')" \
    && chmod -R a+rX /opt/models

# From here on nothing may reach out: a missing model must fail loudly rather
# than quietly download itself on the first run.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TACHYGRAF_WHISPER_DIR=/opt/models/whisper

COPY app ./app
RUN uv pip install --python /opt/venv/bin/python --no-deps .

EXPOSE 7860
CMD ["python", "-m", "app.ui"]
