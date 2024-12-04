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
      ENVIRONMENT=local
      CONFIG_PATH=PATH_TO_CONFIG_FILE
   - Update the path to the data files in the assets/config.json file
   - Run the data preparation  and training pipelines
     ```bash
     cd sentiment_analysis_project/sentimentanalysis
     python data_preparation_pipeline.py
     python training_pipeline.py
     ```

3. **Run the Application**:

     ```bash
     cd sentiment_analysis_project/sentimentanalysis
     python app.py
     ```

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


