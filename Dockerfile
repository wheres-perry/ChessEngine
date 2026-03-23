FROM python:3.11-slim

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# System dependencies — add what the project needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgomp1 \
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
    && chmod 0440 /etc/sudoers.d/${USERNAME}

WORKDIR /app

# Ensure correct ownership of the /app directory itself beforehand
RUN mkdir -p /app && chown -R ${USERNAME}:${USER_GID} /app

# Create a virtual environment and update PATH
ENV PATH="/opt/venv/bin:$PATH"
RUN python -m venv /opt/venv && \
    chown -R ${USERNAME}:${USER_GID} /opt/venv

# Copy codebase into container
COPY --chown=${USERNAME}:${USER_GID} . /app

USER ${USERNAME}

# Install Python dev dependencies into the venv
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

CMD ["nox", "-t", "safe"]
