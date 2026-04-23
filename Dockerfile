# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed because chromadb pulls a few wheels that compile
# C extensions on first install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the package metadata first so dependency resolution caches across
# code edits; the full source comes in afterward.
COPY pyproject.toml README.md ./
COPY api ./api
COPY harness ./harness
COPY grounding ./grounding
COPY memory ./memory
COPY tools ./tools
COPY providers ./providers
COPY data ./data
COPY evals ./evals

RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 1001 app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
