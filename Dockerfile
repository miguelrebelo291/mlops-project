FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_COMPILE=1

COPY requirements-serving.txt .
RUN pip install --no-cache-dir --no-compile -r requirements-serving.txt

COPY src ./src
COPY conf ./conf

COPY data/04_feature/feature_transformers.pkl ./data/04_feature/feature_transformers.pkl
COPY data/06_models/mlflow_model ./data/06_models/mlflow_model
COPY data/08_reporting/model_training_metadata.json ./data/08_reporting/model_training_metadata.json

EXPOSE 8000

CMD ["uvicorn", "mortgage_default.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]