# Build stage
FROM python:3.11-slim@sha256:9e1912aab0a30bbd9488eb79063f68f42a68ab0946cbe98fecf197fe5b085506 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Configure Poetry
RUN poetry config virtualenvs.create false

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --without=dev --no-root --no-cache

# Clean up Poetry and cache
RUN pip uninstall -y poetry && \
    rm -rf ~/.cache/pip && \
    rm -rf ~/.cache/pypoetry

# Runtime stage
FROM python:3.11-slim@sha256:9e1912aab0a30bbd9488eb79063f68f42a68ab0946cbe98fecf197fe5b085506

# Copy only the installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Clean up unnecessary files
RUN find /usr/local -name "*.pyc" -delete && \
    find /usr/local -name "__pycache__" -delete && \
    find /usr/local -name "*.pyo" -delete

WORKDIR /app
COPY . .

CMD ["python", "-c", "import torch; print('Chess Engine Ready')"]