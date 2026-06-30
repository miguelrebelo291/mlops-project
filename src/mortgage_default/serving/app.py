from typing import Any

from fastapi import FastAPI, HTTPException

from mortgage_default.serving.model_service import ModelService
from mortgage_default.serving.schemas import BatchLoanRequest, LoanRequest

app = FastAPI(
    title="Mortgage Default API",
    version="0.1.0",
    description=(
        "API for serving the mortgage default model. "
        "Requests should contain raw Freddie Mac origination-style fields."
    ),
)

service = ModelService()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Mortgage Default API",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    status = service.status()
    status["status"] = "ok" if service.ready else "not_ready"
    return status


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    try:
        return service.metadata_info()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict-one")
def predict_one(request: LoanRequest) -> dict[str, Any]:
    try:
        return service.predict_one(
            features=request.features,
            threshold=request.threshold,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: BatchLoanRequest) -> dict[str, Any]:
    try:
        predictions = service.predict_many(
            rows=request.rows,
            threshold=request.threshold,
        )
        return {
            "n_rows": len(predictions),
            "predictions": predictions,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reload-model")
def reload_model() -> dict[str, Any]:
    service.load()
    status = service.status()
    status["status"] = "ok" if service.ready else "not_ready"
    return status