"""
Tests for the liveness/readiness health endpoints.

The model is mocked so these run without an MLflow registry: the readiness
contract (ready only after the model loads) is what's under test, not the model
itself.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import sentimentanalysis.app as app_module


def test_live_always_ok_even_before_startup():
    # Without entering the TestClient context manager the lifespan does not run,
    # so the model is never loaded — liveness must still report 200.
    app_module.service_state["ready"] = False
    client = TestClient(app_module.app)

    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_ready_is_503_before_model_loads():
    app_module.service_state["ready"] = False
    client = TestClient(app_module.app)  # no `with`: lifespan/startup not run

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "not_ready"}


def test_ready_flips_to_200_after_startup_loads_model(monkeypatch):
    monkeypatch.setattr(app_module, "load_model", lambda: MagicMock())

    # Entering the context manager runs the lifespan startup, which loads the
    # (mocked) model and flips readiness to True.
    with TestClient(app_module.app) as client:
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ready"}

    # Lifespan shutdown ran on context exit: readiness is back to False.
    assert app_module.service_state["ready"] is False
