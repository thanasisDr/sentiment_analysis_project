import numpy as np
import pandas as pd
import pytest

from sentimentanalysis.sentiment_analysis_lr import SentimentAnalysisClf


@pytest.fixture
def model_impl():

    model = SentimentAnalysisClf(
        target_col="sentiment",
        feature_cols="text",
        prediction_col="prediction",
    )
    return model


@pytest.fixture()
def data():
    # mock training data that contains all feature columns and the target column
    return pd.DataFrame(
        {
            "text": [
                "I love this product. It is amazing",
                "Nothing special",
                "I hate this product",
            ]
            * 1000,
            "sentiment": ["positive", "neutral", "negative"] * 1000,
        }
    )


def test_sentiment_analysis_clf_fit_predict_no_append(model_impl, data):

    model_impl.fit(data)
    predictions = model_impl.predict(data)

    # Assert: Check that the output is np array and has the correct shape
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (3000,)


def test_sentiment_analysis_clf_fit_predict_with_append(model_impl, data):
    model_impl.fit(data)
    predictions = model_impl.predict(data, append_prediction=True)

    # Assert: Check that the output is a DataFrame and has the columns and shape
    assert isinstance(predictions, pd.DataFrame)
    assert "prediction" in predictions.columns
    assert predictions.shape == (3000, 3)


def test_sentiment_analysis_clf_evaluate(model_impl, data):

    model_impl.fit(
        data,
    )
    df = model_impl.predict(data, append_prediction=True)
    metrics = model_impl.evaluate(df, metric_names=["accuracy_score"])

    # Assert: Check that the output is a dictionary and has the correct keys
    assert isinstance(metrics, dict)
    assert "accuracy_score" in metrics
