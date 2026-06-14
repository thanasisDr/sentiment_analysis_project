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
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

LOGGING_MSG_FORMAT = (
    "%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level="INFO",
    format=LOGGING_MSG_FORMAT,
    datefmt=LOGGING_DATE_FORMAT,
)
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
    """
    ml_models["sentiment"] = load_model()
    yield
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
