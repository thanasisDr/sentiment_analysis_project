"""
This module contains the code for the sentiment analysis FastAPI app.

The app can be started by running `uvicorn app:app --reload`.
"""

import logging

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

app = FastAPI()

try:
    model_path = (
        "./mlruns/596370538489019037/fbcb4f82232146a3ac1ba6322ce6ac20/artifacts/model"
    )
    if model_path:
        model = mlflow.sklearn.load_model(model_path)
        logger.info("Model loaded successfully.")
    else:
        logger.error("Model path is not set.")
        raise HTTPException(status_code=500, detail="Model path is not set")
except mlflow.exceptions.MlflowException as e:
    logger.error(f"MlflowException: {e}")
    raise HTTPException(
        status_code=500, detail="Failed to load model due to Mlflow error"
    )
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise HTTPException(
        status_code=500, detail="Failed to load model due to unexpected error"
    )

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

    # Check if the model is not None
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
