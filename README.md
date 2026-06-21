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

   Dependencies are managed with [uv](https://docs.astral.sh/uv/). A single
   `pyproject.toml` is the source of truth and `uv.lock` pins every package
   (and transitive dependency) to an exact, reproducible version. The pinned
   Python interpreter is recorded in `.python-version`.

   ```bash
   # Creates a .venv at the pinned Python, installs runtime + dev deps from the lock
   uv sync

   # Add the training/data-prep deps too (needed to run the pipelines below)
   uv sync --group training
   ```

   Prefix commands with `uv run` to execute them inside the synced environment
   (e.g. `uv run python app.py`), or activate `.venv` directly.
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

## Running in Docker

The service ships as a multi-stage, non-root image. **Build from the repository
root** (the build context needs `pyproject.toml` / `uv.lock`):

```bash
docker build -t sentiment-analysis:local .
```

The build installs the exact locked dependencies (`uv sync --frozen`) into a
virtualenv in a builder stage and copies only that venv plus the application
source into the final image — `mlruns/`, `mlflow.db`, `data/`, `.git`, and caches
are excluded via `.dockerignore`.

The image does **not** bake in a model; it loads one at runtime from the MLflow
registry you point it at, so supply the registry config as environment:

```bash
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=<your-tracking-uri> \
  -e MODEL_NAME=sentiment_analysis_clf \
  -e MODEL_ALIAS=champion \
  sentiment-analysis:local
```

(A self-contained local stack with a tracking server comes in a later phase; see
`docs/enterprise-infra-plan.md`.)

## Testing

Run tests using pytest (dev dependencies are installed by `uv sync`):
```bash
cd sentiment_analysis_project
uv run pytest
```

## Development

- Code formatting is enforced using Black.
- Imports are sorted with isort.


