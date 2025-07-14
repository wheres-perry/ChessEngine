# syntax=docker/dockerfile:1

# Stage 1: Dependencies Export
FROM python:3.11-slim AS deps-export

RUN pip install --no-cache-dir poetry poetry-plugin-export

COPY pyproject.toml poetry.lock ./

RUN poetry export --format=requirements.txt --output=requirements.txt --without-hashes --with=dev

# Stage 2: System Dependencies
FROM python:3.11-slim AS system-deps

RUN apt-get update && \
    apt-get install -y --no-install-recommends stockfish && \
    rm -rf /var/lib/apt/lists/*

# Stage 3: Python Dependencies (with pip cache mount)
FROM system-deps AS python-deps

COPY --from=deps-export requirements.txt .

# Use cache mount for pip to speed up dependency installs
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Stage 4: Test Runtime (OPTIMIZED LAYER ORDER)
FROM python-deps AS test-runtime

WORKDIR /app

# Copy files in order of change frequency (least to most frequently changed)
# 1. Configuration files (rarely change)
COPY pyproject.toml poetry.lock ./

# 2. Data
COPY data/raw/example_fens/ ./data/raw/example_fens/

# 3. Root Python files (change occasionally)
COPY *.py ./

# 4. Source code (MOST frequently changed - copy last)
COPY src/ ./src/

# 5. Specific shell scripts only
COPY run_tests.sh ./
RUN chmod +x run_tests.sh

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "pytest", "tests/", "-v"]