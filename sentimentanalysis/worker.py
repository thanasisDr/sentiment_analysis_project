"""
Custom gunicorn worker.

gunicorn applies our JSON logging config (see ``gunicorn_conf.py``'s
``logconfig_dict``) at master startup. By default the uvicorn worker would then
re-apply uvicorn's own logging config inside each worker, overriding the JSON
formatter on the ``uvicorn.*`` loggers. Passing ``log_config=None`` tells
uvicorn to leave logging alone: the ``uvicorn.access`` / ``uvicorn.error``
loggers fall through to the JSON handler from the shared config, so access logs
are structured JSON too.
"""

from uvicorn_worker import UvicornWorker


class AppUvicornWorker(UvicornWorker):
    CONFIG_KWARGS = {"log_config": None}
