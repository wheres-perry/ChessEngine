# syntax=docker/dockerfile:1
FROM python:3.13.2-alpine3.21

WORKDIR /app

ENV PYTHONPATH=/app

COPY . .

RUN echo "https://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories && \
    apk update && \
    apk add --no-cache stockfish && \
    pip install --no-cache-dir pip==25.1.1 && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --with=dev --no-cache --no-root

CMD ["poetry", "run", "python", "-c", "import chess; import chess.engine; print('Chess Engine Ready with Stockfish')"]