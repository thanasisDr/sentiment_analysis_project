# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the sentiment-analysis serving image.
#
# Build context is the REPOSITORY ROOT (not sentimentanalysis/), because the
# dependency lock lives at the root:
#
#     docker build -t sentiment-analysis:local .
#
# Stage 1 ("builder") resolves the locked dependencies into a self-contained
# virtualenv. Stage 2 ("runtime") copies only that venv plus the application
# source, so build tools, the uv binary, caches, and the rest of the repo never
# reach the final image. The result is a small, non-root, reproducible image.

############################
# Stage 1 — builder
############################
FROM python:3.12-slim AS builder

# Pinned uv binary, pulled from its official image (no pip bootstrap needed).
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install ONLY the locked dependencies first, in their own layer. This layer is
# cached and only re-runs when pyproject.toml or uv.lock change — application
# code edits (the common case) reuse it.
#   --frozen             : install exactly what uv.lock pins; never re-resolve
#   --no-install-project : the app is run as source, not installed as a package
#   --no-dev             : exclude the dev group (pytest/black/...); the
#                          non-default `training` group is excluded automatically
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

############################
# Stage 2 — runtime
############################
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/appuser \
    # Put the venv's interpreter/entrypoints first on PATH.
    PATH="/app/.venv/bin:$PATH"

# Non-root user with a fixed numeric UID/GID (satisfies Kubernetes
# runAsNonRoot and makes file ownership predictable).
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# The venv is relocatable as-is because both stages share the same base image,
# so /app/.venv references the identical interpreter path in each.
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv

# Application source only. mlruns/, mlflow.db, data/, .git, caches, etc. are
# kept out of the build context by .dockerignore — the model is loaded at
# runtime from the MLflow registry named by MLFLOW_TRACKING_URI, not baked in.
# sentiment_analysis_lr.py must be present: the registered model is an instance
# of its custom estimator class and is unpickled at startup.
COPY --chown=10001:10001 sentimentanalysis/ /app/sentimentanalysis/

# Run from the package dir so `app:app` resolves and the custom estimator module
# is importable on unpickle.
WORKDIR /app/sentimentanalysis
USER 10001

EXPOSE 8000

# Container-level readiness probe. Kubernetes uses its own probes against the
# same endpoint, but this gives Docker/Compose a signal today: /health/ready
# returns 200 only once the model has loaded (503 otherwise). Uses the stdlib so
# the slim image needs no curl. start-period covers model load on boot.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health/ready', timeout=2).status == 200 else 1)"]

# gunicorn manages a pool of uvicorn workers for process-level concurrency on top
# of the per-process threadpool. Worker count, bind address, timeouts and log
# level are all env-driven (see gunicorn_conf.py); WEB_CONCURRENCY sets workers.
CMD ["gunicorn", "app:app", "-c", "gunicorn_conf.py"]
