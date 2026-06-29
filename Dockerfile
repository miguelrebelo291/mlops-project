FROM python:3.11-slim

WORKDIR /app

# Needed for packages that build native extensions, e.g. twofish
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY conf ./conf

COPY data/04_feature/feature_transformers.pkl ./data/04_feature/feature_transformers.pkl
COPY data/06_models/mlflow_model ./data/06_models/mlflow_model
COPY data/08_reporting/model_training_metadata.json ./data/08_reporting/model_training_metadata.json

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "mortgage_default.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]