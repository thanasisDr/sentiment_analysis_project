"""
Structured (JSON) logging configuration shared by the app and the gunicorn
master/worker processes.

One JSON object is emitted per line to stdout, which is what container log
collectors (Docker, Kubernetes/Fluent Bit, Azure Monitor) expect: the platform
captures stdout and the structure is preserved for indexing and querying. The
same configuration governs application logs, gunicorn's own logs, and (via the
custom worker in ``worker.py``) uvicorn's access/error logs, so every line on
stdout has a consistent shape.
"""

import datetime as dt
import json
import logging
import logging.config  # not pulled in by `import logging`; needed for dictConfig
import os

# Attribute names that already live on a bare LogRecord. Anything *not* in this
# set was attached by the caller via ``logger.info(..., extra={...})`` and is
# promoted to a top-level field in the JSON output.
_RESERVED_ATTRS = set(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": dt.datetime.fromtimestamp(
                record.created, dt.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        # Promote caller-supplied `extra=` fields to top-level keys.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


def build_logging_config(level: str | None = None) -> dict:
    """
    Build a ``logging.config.dictConfig`` dictionary that sends JSON to stdout.

    The level defaults to the ``LOG_LEVEL`` env var (then ``INFO``). The
    formatter is referenced by class object rather than dotted path so the config
    works regardless of whether this module is imported as ``logging_config``
    (serving, WORKDIR = the package dir) or ``sentimentanalysis.logging_config``
    (tests, from the repo root).
    """
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    server_logger = {"handlers": ["default"], "level": level, "propagate": False}

    return {
        "version": 1,
        # Loggers created at import time (e.g. the app's module logger) must keep
        # working, so do not disable them when this config is applied.
        "disable_existing_loggers": False,
        "formatters": {"json": {"()": JsonFormatter}},
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"handlers": ["default"], "level": level},
        # Give the server loggers the JSON handler directly (and stop them
        # propagating to root) so they are not double-logged.
        "loggers": {
            "gunicorn.error": server_logger,
            "gunicorn.access": server_logger,
            "uvicorn": server_logger,
            "uvicorn.error": server_logger,
            "uvicorn.access": server_logger,
        },
    }


def configure_logging(level: str | None = None) -> None:
    """Apply the JSON logging configuration to the current process."""
    logging.config.dictConfig(build_logging_config(level))
