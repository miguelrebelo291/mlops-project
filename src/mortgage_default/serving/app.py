from typing import Any

from fastapi import FastAPI, HTTPException

from mortgage_default.serving.model_service import ModelNotReadyError, ModelService
from mortgage_default.serving.schemas import (
    BatchFeaturePayload,
    BatchPredictionResponse,
    FeaturePayload,
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
)


app = FastAPI(
    title="Mortgage Default Model Serving API",
    version="0.1.0",
    description="Serves the selected mortgage default model trained by the Kedro/MLflow pipeline.",
)

model_service = ModelService()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Mortgage Default Model Serving API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    status = model_service.status()

    return {
        "status": "ok" if status["is_ready"] else "not_ready",
        "model_ready": status["is_ready"],
        "model_uri": status["model_uri"],
        "metadata_path": status["metadata_path"],
        "selected_model_family": status["selected_model_family"],
        "threshold": status["threshold"],
        "load_error": status["load_error"],
    }


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> dict[str, Any]:
    if not model_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not ready. Load error: {model_service.load_error}",
        )

    assert model_service.metadata is not None

    return {
        "model_uri": model_service.model_uri,
        "metadata_path": str(model_service.metadata_path),
        "selected_model_family": model_service.metadata.get("selected_model_family"),
        "threshold": model_service.metadata.get("threshold"),
        "target_column": model_service.metadata.get("target_column"),
        "id_column": model_service.metadata.get("id_column"),
        "n_features": len(model_service.metadata.get("feature_columns", [])),
        "scoring": model_service.metadata.get("scoring"),
        "selection_metric": model_service.metadata.get("selection_metric"),
    }


@app.post("/predict", response_model=BatchPredictionResponse)
def predict(request: BatchFeaturePayload) -> dict[str, Any]:
    try:
        predictions = model_service.predict_many(
            rows=request.rows,
            threshold=request.threshold,
        )

        return {
            "n_rows": len(predictions),
            "predictions": predictions,
        }

    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected prediction error: {exc}",
        ) from exc


@app.post("/predict-one", response_model=PredictionResponse)
def predict_one(request: FeaturePayload) -> dict[str, Any]:
    try:
        return model_service.predict_one(
            features=request.features,
            threshold=request.threshold,
        )

    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected prediction error: {exc}",
        ) from exc


@app.post("/reload-model", response_model=HealthResponse)
def reload_model() -> dict[str, Any]:
    global model_service

    model_service = ModelService()
    status = model_service.status()

    return {
        "status": "ok" if status["is_ready"] else "not_ready",
        "model_ready": status["is_ready"],
        "model_uri": status["model_uri"],
        "metadata_path": status["metadata_path"],
        "selected_model_family": status["selected_model_family"],
        "threshold": status["threshold"],
        "load_error": status["load_error"],
    }