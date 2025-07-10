# syntax=docker/dockerfile:1

# Using a PyTorch image with Python 3.11 and CUDA 11.8 on Ubuntu
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-py3.11-runtime

WORKDIR /app

ENV PYTHONPATH=/app

COPY . .

# Install dependencies, tools, and project packages in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends stockfish && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --with=dev --no-cache --no-root

# Command to confirm the chess engine is ready
CMD ["poetry", "run", "python", "-c", "import chess; import chess.engine; print('Chess Engine Ready with Stockfish')"]