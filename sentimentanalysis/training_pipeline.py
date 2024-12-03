import json
import logging
import os
from typing import Any

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sentiment_analysis_lr import SentimentAnalysisClf
from sklearn.model_selection import train_test_split

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


def get_data(configuration: dict) -> pd.DataFrame:
    """
    Reads a CSV file based on the given configuration and returns it as a pandas DataFrame.

    Parameters
    ----------
    configuration : dict
        A dictionary containing the configuration for the data loading step.
        It must contain the key "data_path" pointing to the location of the CSV file.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data.
    """
    return pd.read_csv(configuration["data_path"])


def train_model(
    model: Any,
    configuration: dict,
    data: pd.DataFrame,
) -> None:
    """
    Train a given model using the given configuration and data.

    Parameters
    ----------
    model : Any
        The model to be trained.
    configuration : dict
        A dictionary containing the configuration for the training step.
        It must contain the key "experiment_name" pointing to the name
        of the MLflow experiment to be created.
    data : pd.DataFrame
        The data to be used for training.

    Returns
    -------
    None

    Notes
    -----
    This function will create a new MLflow experiment if it does not exist
    and log all the experiment details. It will then split the data into
    training and test sets, train the model, evaluate the model on the
    test set, log the metrics, and save the model in the MLflow experiment.
    """
    client = MlflowClient()

    # Get parsed experiment name
    experiment_name = configuration["experiment_name"]

    # Create MLflow experiment
    try:
        # Try to create a new experiment
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = client.get_experiment_by_name(experiment_name)
    except:
        # If experiment already exists, get the existing experiment
        experiment = client.get_experiment_by_name(experiment_name)

    mlflow.set_experiment(experiment_name)

    # Log experiment details
    logger.info(f"Name: {experiment_name}")
    logger.info(f"Experiment_id: {experiment.experiment_id}")
    logger.info(f"Artifact Location: {experiment.artifact_location}")
    logger.info(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    logger.info(f"Tracking uri: {mlflow.get_tracking_uri()}")

    # Setup and wrap AutoML training with MLflow
    with mlflow.start_run():

        # Split data into training and test sets
        df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)

        # Train the model
        model.fit(data=df_train)

        # Evaluate out-of-sample on test set
        pred_test_df = model.predict(data=df_test, append_prediction=True)
        test_metrics = model.evaluate(data=pred_test_df, metric_names="accuracy_score")

        # Log the metrics
        mlflow.log_metrics(test_metrics)

        # Save the model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Get the model URI
        model_uri = mlflow.get_artifact_uri("model")
        logger.info(f"The model saved in {model_uri}")


def main():
    """
    Main function to train a sentiment analysis model using MLflow.

    This function will load the configuration, read the data, create the model, train the model,
    evaluate the model, log the metrics to MLflow, and save the model to MLflow.
    """

    # Load the configuration
    load_dotenv()
    env = os.getenv("ENVIRONMENT")
    config_path = os.getenv("CONFIG_PATH")
    with open(config_path, "r") as fin:
        configuration = json.load(fin)
    configuration = configuration[env]

    # Read the data
    data = get_data(configuration)

    # Create the model
    model = SentimentAnalysisClf(
        feature_cols="text", target_col="sentiment", prediction_col="prediction"
    )

    # Train the model
    train_model(model, configuration, data)


if __name__ == "__main__":
    main()
