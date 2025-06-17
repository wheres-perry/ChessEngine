FROM python:3.13.2-alpine3.21@sha256:323a717dc4a010fee21e3f1aac738ee10bb485de4e7593ce242b36ee48d6b352

WORKDIR /app

COPY . .


RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --with=dev --no-cache

CMD ["poetry", "run", "python", "-c", "import chess; print('Chess Engine Ready')"]