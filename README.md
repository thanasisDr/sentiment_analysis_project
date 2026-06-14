# Sentiment Analysis Project

## Overview

This project is a sentiment analysis project using the Amazon Book Reviews dataset. It showcases a range of skills in developing and deploying machine learning models.

## Components

- **Model Development**: Develop machine learning models using scikit-learn.
- **API Development**: Deploy models with a FastAPI backend.
- **Monitoring**: Implement monitoring using prometheus_fastapi_instrumentator.
- **Containerisation**: Containerize a FastAPI application using Docker.

## Setup Instructions

1. **Clone the Repository**: 
   ```bash
   git clone <repository-url>
   cd sentiment_analysis_project
   ```

2. **Install Dependencies**:
   - Create and activate a virtual environment.
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```
3. **Train the model locally**:
   - Create a .env file with the following variables:
      ```
      ENVIRONMENT=local
      CONFIG_PATH=PATH_TO_CONFIG_FILE
      # Optional. The MLflow Model Registry requires a database-backed store;
      # this defaults to a local SQLite file if unset.
      MLFLOW_TRACKING_URI=sqlite:///mlflow.db
      ```
   - Update the path to the data files in the assets/config.json file
   - Run the data preparation  and training pipelines
     ```bash
     cd sentiment_analysis_project/sentimentanalysis
     python data_preparation_pipeline.py
     python training_pipeline.py
     ```
   - Training registers the model in the MLflow Model Registry under the
     `registered_model_name` from the config and promotes the new version to the
     `champion` alias.

4. **Run the Application**:

   The app loads the model from the registry via `models:/<name>@<alias>`, so it
   is decoupled from any specific training run. It is configured through the
   environment (defaults shown):

   ```
   MLFLOW_TRACKING_URI=sqlite:///mlflow.db
   MODEL_NAME=sentiment_analysis_clf
   MODEL_ALIAS=champion
   ```

   Run it (from the same directory used for training, so the SQLite store
   resolves to the same file):

     ```bash
     cd sentiment_analysis_project/sentimentanalysis
     python app.py
     ```

## Demo

With the application running (see step 4), try it out from another terminal.
The server listens on `http://127.0.0.1:8000` by default.

```bash
# Health check
curl -s http://127.0.0.1:8000/

# Predictions
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Beautifully written and deeply moving."}'

curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Boring, poorly written, and a total waste of money."}'

curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "It was okay, nothing special but not bad either."}'

# Error handling: blank text -> 400, missing field -> 422
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "   "}'

curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{}'

# Monitoring: Prometheus metrics (request counts and latency histograms)
curl -s http://127.0.0.1:8000/metrics | grep http_request
```

Interactive API docs are also available at `http://127.0.0.1:8000/docs`.

## Testing

Run tests using pytest:
```bash
pip install -r requirements_dev.txt
cd sentiment_analysis_project
pytest
```

## Development

- Code formatting is enforced using Black.
- Imports are sorted with isort.


