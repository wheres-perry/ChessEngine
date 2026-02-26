# Stage 1: Base - Setup python and UV
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Install system runtime deps (libraries needed for running, not building)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Stage 2: Builder - Compilers and Build Tools
FROM base AS builder
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install build dependencies (compilers, git, clang-format for linting)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    clang-format \
    clang-tidy \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
# Create the virtual environment
RUN uv sync --frozen --no-install-project --no-dev

# Stage 3: Development - The "Fat" Image for CI/Testing
FROM builder AS development
# Install ALL dependencies (including dev)
RUN uv sync --frozen --no-install-project

COPY . .
# Install the project itself (compiles C++ extensions)
RUN uv sync --frozen

# Default to running the safe suite
CMD ["nox", "-t", "safe"]

# Stage 4: Production - The "Lean" Image for Deployment
FROM base AS production

COPY --from=builder /bin/uv /bin/uv
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY setup.py ./
COPY scripts/download_syzygy.py ./scripts/download_syzygy.py

# Install PROD only, no dev tools.
# This compiles the C++ extension for production flags.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && uv sync --frozen --no-dev \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Download Syzygy tablebases (baked into image).
# Set SYZYGY_PATH so the engine config picks it up at runtime.
ENV SYZYGY_PATH=/app/data/syzygy
RUN python3 scripts/download_syzygy.py --path "$SYZYGY_PATH"

ENV PATH="/app/.venv/bin:$PATH"

# Entry point
ENTRYPOINT ["python", "-m", "engine.main"]
