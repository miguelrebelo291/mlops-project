# Running the Pipeline with Sample Data

This repository includes a small delivery sample dataset (~1,000 loans per year)
so the full pipeline can be demonstrated without needing the complete Freddie Mac
dataset files (which are several GB in size).

---

## Requirements

- Python 3.11
- [uv](https://github.com/astral-sh/uv) for package management
- Docker (for model serving)

---

## Setup

```bash
# Clone the repository
git clone https://github.com/miguelrebelo291/mlops-project
cd mortgage-default

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows (Git Bash: source .venv/Scripts/activate)
```

---

## Running the Pipelines

All pipelines use the `--env delivery` flag, which points to the sample data
in `data/01_raw/delivery/` instead of the full dataset.

### Step 1 — Training pipeline (data ingestion → cleaning → model training)

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
kedro run --env delivery --pipeline training
```

### Step 2 — Model evaluation (temporal test set 2003+)

```bash
kedro run --env delivery --pipeline model_evaluate
```

### Step 3 — SHAP explainability

```bash
kedro run --env delivery --pipeline model_explainability
```

### Step 4 — Inference / prediction (2007 loans, no ground truth)

```bash
kedro run --env delivery --pipeline model_predict
```

### Step 5 — Data drift detection (Evidently, year by year)

```bash
kedro run --env delivery --pipeline data_drift
```

### Run everything sequentially

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
kedro run --env delivery --pipeline training
kedro run --env delivery --pipeline model_evaluate
kedro run --env delivery --pipeline model_explainability
kedro run --env delivery --pipeline model_predict
kedro run --env delivery --pipeline data_drift
```

---

## Model Serving (Docker)

Build and run the FastAPI serving container:

```bash
docker build -t mortgage-default-api .
docker run -p 8000:8000 -d mortgage-default-api
```

Open the interactive API docs at: http://localhost:8000/docs

Test the API with the provided notebook:

```bash
jupyter notebook notebooks/test_serving_api.ipynb
```

---

## MLflow UI

To inspect experiment runs, metrics, and model artifacts:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open: http://localhost:5000

---

## Notes

- The sample data contains ~1,000 loans per year for origination files and
  ~5,000 rows per year for performance files — sufficient to run the full
  pipeline end to end, but results will differ from the full dataset.
- The NSD (Non-Standard Dataset) sample is not included in the delivery
  sample — the pipeline will run with Standard Dataset only.
- Model performance metrics on the sample data will be lower than reported
  in the full pipeline run, due to the reduced training set size.
