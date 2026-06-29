"""
This module contains the code for the sentiment analysis FastAPI app.

The app can be started by running `uvicorn app:app --reload`.
"""

import logging
import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# Configure structured JSON logging for the whole process. Importable both as a
# top-level module (serving: WORKDIR is the package dir) and as a package module
# (tests: from the repo root), so try both.
try:
    from logging_config import configure_logging
except ModuleNotFoundError:  # pragma: no cover - import-path shim
    from sentimentanalysis.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Model resolution is configured entirely through the environment so the same
# image can be promoted across environments without code changes. The model is
# loaded from the MLflow Model Registry via "models:/<name>@<alias>", which
# decouples serving from any specific training run id.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "sentiment_analysis_clf")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# Holds the loaded model for the lifetime of the process. Populated on startup.
ml_models: dict = {}

# Readiness flag for the /health/ready probe. Flips to True only once the model
# is loaded, and back to False on shutdown, so orchestrators route traffic to a
# replica only when it can actually serve predictions.
service_state: dict = {"ready": False}


def load_model():
    """
    Load the serving model from the MLflow Model Registry.

    Returns
    -------
    The loaded scikit-learn model.

    Raises
    ------
    Exception
        If the model cannot be resolved or loaded, so the app fails fast on
        startup instead of accepting traffic it cannot serve.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"Loading model from {MODEL_URI} (tracking: {MLFLOW_TRACKING_URI})")
    model = mlflow.sklearn.load_model(MODEL_URI)
    logger.info("Model loaded successfully.")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model once on startup and keep it in memory for every request.

    Loading at startup (rather than import time) lets the app fail fast with a
    clear error, keeps prediction latency low, and plays well with the ASGI
    lifecycle and test clients.

    Readiness flips True only after the model is in memory and False again on
    shutdown. On SIGTERM the ASGI server stops accepting new connections and
    drains in-flight requests before this teardown runs (graceful shutdown).
    """
    ml_models["sentiment"] = load_model()
    service_state["ready"] = True
    yield
    service_state["ready"] = False
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

Instrumentator().instrument(app).expose(app)


class PredictionInput(BaseModel):
    """
    Prediction input model.

    This model is used to parse the input data from the request body.
    """

    text: str


@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message.

    Returns:
        dict: A dictionary containing a welcome message.
    """
    return {"message": "This is a sentiment analysis app for book reviews"}


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    """
    Liveness probe: is the process up and the event loop responsive?

    Always returns 200 while the app can answer. It deliberately does *not*
    check the model — a liveness failure tells an orchestrator to restart the
    pod, and a missing model is not fixed by a restart. Kubernetes maps this to
    `livenessProbe`.
    """
    return {"status": "alive"}


@app.get("/health/ready")
async def health_ready():
    """
    Readiness probe: can this replica serve predictions right now?

    Returns 200 only after the model has loaded, otherwise 503 so the
    orchestrator keeps the replica out of the load-balancer rotation (and holds
    a rolling deploy) until it is ready. Kubernetes maps this to `readinessProbe`.
    """
    if service_state["ready"] and ml_models.get("sentiment") is not None:
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "not_ready"})


@app.post("/predict")
async def predict(input_data: PredictionInput) -> dict[str, str]:
    """
    Predicts the sentiment of the given text.

    Args:
        input_data (PredictionInput): The input data to be analyzed.

    Returns:
        dict[str, str]: A dictionary with the sentiment of the input text.
    """
    if input_data is None:
        raise HTTPException(status_code=400, detail="Input data is None")

    if input_data.text is None or input_data.text.strip() == "":
        raise HTTPException(status_code=400, detail="Input text is empty or null")

    # Create a DataFrame with the input text
    df = pd.DataFrame({"text": [input_data.text]})

    # Check if the model has been loaded
    model = ml_models.get("sentiment")
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not initialized")

    # Make a prediction using the model
    try:
        prediction = model.predict(df)
    except Exception as e:
        logger.error(f"Error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error occurred during prediction")

    # Return a dictionary with the sentiment
    return {"sentiment": prediction[0]}


if __name__ == "__main__":
    """
    Main entry point.

    If the module is ran directly, it starts the FastAPI app using Uvicorn.
    """
    uvicorn.run("app:app", reload=True)
