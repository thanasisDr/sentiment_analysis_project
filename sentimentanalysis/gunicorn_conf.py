"""
gunicorn configuration for the sentiment-analysis serving image.

Run with:

    gunicorn app:app -c gunicorn_conf.py

Everything is driven by environment variables so the same image runs unchanged
across environments. gunicorn forks ``workers`` processes (process-level
concurrency); each runs a uvicorn ASGI worker, which keeps the per-process
threadpool fix for the sync ``/predict`` handler. Two layers of concurrency:
processes across cores, threads within each process.
"""

import multiprocessing
import os

from logging_config import build_logging_config

# --- Socket -----------------------------------------------------------------
_host = os.getenv("HOST", "0.0.0.0")
_port = os.getenv("PORT", "8000")
bind = f"{_host}:{_port}"

# --- Workers ----------------------------------------------------------------
# Default to the common (2 * CPU) + 1 heuristic, overridable via WEB_CONCURRENCY
# (gunicorn's standard knob). Each worker loads its own copy of the model on
# startup, so size this against available memory, not just CPU.
workers = int(os.getenv("WEB_CONCURRENCY", (multiprocessing.cpu_count() * 2) + 1))
worker_class = "worker.AppUvicornWorker"

# --- Timeouts / graceful shutdown -------------------------------------------
# On SIGTERM gunicorn stops accepting new connections and gives in-flight
# requests up to `graceful_timeout` seconds to drain before workers are killed;
# the app's lifespan shutdown then runs to release the model. `timeout` is the
# worker liveness watchdog (a worker silent this long is recycled).
graceful_timeout = int(os.getenv("GRACEFUL_TIMEOUT", "30"))
timeout = int(os.getenv("TIMEOUT", "30"))
keepalive = int(os.getenv("KEEPALIVE", "5"))

# --- Logging ----------------------------------------------------------------
# Apply the shared JSON config to gunicorn's master and worker loggers. The
# custom worker (worker.py) extends this to uvicorn's loggers as well.
logconfig_dict = build_logging_config()
