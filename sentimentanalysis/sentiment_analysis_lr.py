import logging
from typing import Any, List, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class SentimentAnalysisClf:
    def __init__(
        self, target_col: str, feature_cols: Union[str, List[str]], prediction_col: str
    ) -> None:
        """
        Initializes the SentimentAnalysisClf instance.

        Parameters
        ----------
        target_col : str
            The column name of the target variable.
        feature_cols : Union[str, List[str]]
            The column name(s) of the feature(s).
        prediction_col : str
            The column name of the prediction.

        Returns
        -------
        None
        """
        # Set the column names
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.prediction_col = prediction_col

        # Initialize the model
        self._model = None

        # Initialize the model pipeline
        self._init_model()

    def _init_model(self) -> None:
        """
        Initialize the machine learning model pipeline.

        This function sets up a scikit-learn Pipeline consisting of a
        CountVectorizer and a LogisticRegression model. The CountVectorizer
        transforms the text data into a matrix of token counts, and the
        LogisticRegression is used for classification.

        The CountVectorizer is set up with a minimum document frequency of 100
        and will ignore words that appear less than 100 times. The
        LogisticRegression is set up with default parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logger.info("Initializing the machine learning model pipeline")
        # CountVectorizer to convert text data into a matrix of token counts
        count_vect: CountVectorizer = CountVectorizer(
            min_df=100, stop_words="english"
        )  # ignore words that appear less than 100 times
        # Logistic Regression model for sentiment classification
        lr: LogisticRegression = LogisticRegression()
        # Set up the scikit-learn Pipeline
        self._model: Pipeline = Pipeline(steps=[("count_vect", count_vect), ("lr", lr)])

    @property
    def model(self):
        """
        Get the machine learning model pipeline.

        Returns:
            Pipeline: The scikit-learn Pipeline object.
        """
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def fit(
        self,
        data: pd.DataFrame,
        hyperparameters: dict = None,
        **kwargs,
    ) -> None:
        """
        Train the machine learning model pipeline.

        This function trains the scikit-learn Pipeline by fitting it to the given
        data. The data is expected to have the feature columns and the target
        column set.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be used for training the model.
        hyperparameters : dict, optional
            A dictionary of hyperparameters to be used for training the model.
            The default is None.

        Returns
        -------
        None
        """

        logger.info(f"[fit|in] Configured model: {self._model}")

        # Train the model
        self._model.fit(
            X=data[self.feature_cols],
            y=data[self.target_col],
        )

        logger.info("[fit|out] Model fitted.")

    def predict(
        self,
        data: pd.DataFrame,  # Input DataFrame containing the text data.
        append_prediction: bool = False,  # Whether to append the prediction result to the input DataFrame.
    ) -> Union[pd.DataFrame, np.ndarray]:  # Predicted sentiment of the given text.
        """
        Predict the sentiment of the given text.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be used for prediction.
        append_prediction : bool, optional
            Whether to append the prediction result to the input DataFrame.
            The default is False.

        Returns
        -------
        Union[pd.DataFrame, np.ndarray]
            The predicted sentiment of the given text. If append_prediction is
            True, then the prediction result is appended to the input DataFrame
            and returned as a pandas DataFrame. Otherwise, the prediction result
            is returned as a numpy array.
        """
        logger.info(f"[predict|in]")

        # Select the feature columns from the input DataFrame
        model_input: pd.DataFrame = data[self.feature_cols].copy()  # type: ignore

        # Predict the sentiment of the text using the machine learning model
        model_output: np.ndarray = self.model.predict(model_input)  # type: ignore

        # If append_prediction is True, append the prediction result to the input DataFrame
        if append_prediction:
            logger.info("[predict|out] Appending prediction result to DataFrame")
            data: pd.DataFrame = data.assign(**{self.prediction_col: model_output})  # type: ignore
            return data
        else:
            logger.info("[predict|out] Returning prediction result as numpy array")
            return model_output

    def evaluate(
        self,
        data: pd.DataFrame,  # DataFrame containing the text and target sentiment.
        metric_names: List[
            str
        ] = None,  # List of metric names to be used for evaluation.
        **kwargs: Any,  # Additional keyword arguments to be passed to the evaluation metrics.
    ) -> dict[str, float]:  # Dictionary containing the evaluation result.
        """
        Evaluate the performance of the model on the given data.

        This function evaluates the performance of the model using the given
        metric names. The result is returned as a dictionary.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be used for evaluation. It must contain the text column
            and the target sentiment column.
        metric_names : List[str], optional
            A list of metric names to be used for evaluation. The default is None.
        **kwargs : Any
            Additional keyword arguments to be passed to the evaluation metrics.

        Returns
        -------
        dict[str, float]
            A dictionary containing the evaluation result. The keys are the metric
            names and the values are the corresponding evaluation scores.
        """
        logger.info(f"[evaluate|in] ({metric_names})")

        # Initialize an empty dictionary to store the result
        result = {}

        # Check if accuracy_score is in the metric names
        if "accuracy_score" in metric_names:
            # Calculate the accuracy score
            accuracy = accuracy_score(data[self.prediction_col], data[self.target_col])
            # Store the accuracy score in the result dictionary
            result["accuracy_score"] = accuracy

        # Log the result
        logger.info(f"[evaluate|out] ({result})")
        # Return the result dictionary
        return result
