# One image, many roles. Every service in this stack runs the same code with a different
# entrypoint, so a single build is cached once and reused by all of them.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# curl is needed by the container healthchecks; nothing else is installed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer, cached until pyproject/uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# Application layer.
COPY agent/ ./agent/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY servers/ ./servers/
COPY services/ ./services/
COPY data/ ./data/
COPY scripts_index_documents.py ./
RUN uv sync --frozen --no-dev

# Run as a non-root user. Configuration arrives at runtime through the environment;
# no secret is ever baked into this image.
RUN useradd --create-home --uid 10001 analyst \
    && mkdir -p /app/data/chroma \
    && chown -R analyst:analyst /app
USER analyst

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
