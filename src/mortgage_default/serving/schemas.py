from typing import Any

from pydantic import BaseModel, Field


class FeaturePayload(BaseModel):
    """One prediction request using already-engineered model features."""

    features: dict[str, Any] = Field(
        ...,
        description="Dictionary mapping feature names to values for one loan.",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional threshold override. Defaults to training threshold.",
    )

    class Config:
        extra = "forbid"


class BatchFeaturePayload(BaseModel):
    """Batch prediction request using already-engineered model features."""

    rows: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of feature dictionaries to score.",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional threshold override. Defaults to training threshold.",
    )

    class Config:
        extra = "forbid"


class PredictionResponse(BaseModel):
    loan_sequence_number: str | None = None
    prediction: int
    probability_default: float | None = None
    threshold: float | None = None
    selected_model_family: str | None = None
    model_uri: str
    model_status: str

    class Config:
        extra = "forbid"


class BatchPredictionResponse(BaseModel):
    n_rows: int
    predictions: list[PredictionResponse]

    class Config:
        extra = "forbid"


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_uri: str
    metadata_path: str
    selected_model_family: str | None = None
    threshold: float | None = None
    load_error: str | None = None

    class Config:
        extra = "forbid"


class MetadataResponse(BaseModel):
    model_uri: str
    metadata_path: str
    selected_model_family: str | None = None
    threshold: float | None = None
    target_column: str | None = None
    id_column: str | None = None
    n_features: int
    scoring: str | None = None
    selection_metric: str | None = None

    class Config:
        extra = "forbid"