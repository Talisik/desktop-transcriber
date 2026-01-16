FROM python:3.11-slim

WORKDIR /app

# install system dependencies for audio/video processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# copy dependency files
COPY pyproject.toml uv.lock ./

# install dependencies
RUN uv sync --frozen

# copy application code
COPY transcriber/ ./transcriber/
COPY main.py ./

# set python path
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# default command
CMD ["python", "main.py"]

