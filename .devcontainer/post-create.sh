#!/bin/bash
# =============================================================================
# Dev Container Post-Create Setup Script
# =============================================================================
#
# PURPOSE:
#   This script runs AFTER the dev container is created and the workspace is
#   mounted. It prepares your project-specific development environment.
#
# GOALS:
#   1. Clean stale cache/build artifacts from the mounted workspace
#   2. Install Python dependencies from your project's pyproject.toml
#   3. Compile C++ extensions (pybind11) into Python modules
#
# WHY IN A SEPARATE FILE/ WHY NOT IN THE DOCKERFILE?
#     Bash commands in JSON are hard to read, Dockerfile runs at IMAGE BUILD time, before workspace
#     exists, project files are MOUNTED at runtime, not in the image.
# =============================================================================

set -e

echo "🧹 Cleaning cache and build artifacts..."
find /workspace -type d \( \
    -name '__pycache__' -o \
    -name '.pytest_cache' -o \
    -name '.ruff_cache' -o \
    -name '.mypy_cache' -o \
    -name 'build' -o \
    -name '.ipynb_checkpoints' -o \
    -name '*.egg-info' \
\) -exec rm -rf {} + 2>/dev/null || true

echo "📦 Installing Python packages..."
sudo uv pip install --system -e .[dev]

echo "🔨 Compiling C++ extensions..."
python compile.py --no-tests

echo "✅ Dev container setup complete!"
