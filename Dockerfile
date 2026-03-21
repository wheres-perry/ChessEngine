# ======================================================================
# STAGE 1: Base
# Shared runtime foundation for all stages.
# ======================================================================
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# ======================================================================
# STAGE 2: Builder
# Heavy native build toolchain used by both development and production.
# ======================================================================
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# ======================================================================
# STAGE 3: Development
# Full-featured image for local dev, CI workflows, and nox sessions.
# ======================================================================
FROM builder AS development

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    ccache \
    clang-format \
    clang-tidy \
    curl \
    dos2unix \
    stockfish \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN if ! getent group ${USER_GID} >/dev/null 2>&1; then \
    groupadd --gid ${USER_GID} ${USERNAME}; \
    fi \
    && if ! id -u ${USERNAME} >/dev/null 2>&1; then \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}; \
    fi \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && chown -R ${USERNAME}:${USER_GID} /opt/venv \
    && chown ${USERNAME}:${USER_GID} /app

COPY --chown=${USERNAME}:${USER_GID} . .

USER ${USERNAME}
RUN echo "auto_activate_base: false" > /home/${USERNAME}/.condarc
RUN uv sync --frozen --group dev

CMD ["nox", "-t", "safe"]

# ======================================================================
# STAGE 4: Production
# Lean deployable image with runtime dependencies and compiled extension.
# ======================================================================
FROM base AS production

COPY pyproject.toml uv.lock CMakeLists.txt ./
COPY src/ ./src/
COPY scripts/download_syzygy.py ./scripts/download_syzygy.py

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    && uv sync --frozen \
    && apt-get purge -y --auto-remove \
    build-essential \
    cmake \
    ninja-build \
    git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV SYZYGY_PATH=/app/data/syzygy
RUN mkdir -p "${SYZYGY_PATH}" && python scripts/download_syzygy.py --path "${SYZYGY_PATH}"

ENTRYPOINT ["python", "-m", "engine"]
