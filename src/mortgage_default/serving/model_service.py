import json
import pickle
from pathlib import Path
from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd

from mortgage_default.pipelines.data_cleaning.nodes import clean_origination_data
from mortgage_default.pipelines.model_train.nodes import (  # CHANGED: was feature_engineering.nodes
    apply_feature_transformers,
    drop_unused_columns,
    engineer_date_features,
)

MODEL_URI = "data/06_models/mlflow_model"
METADATA_PATH = Path("data/08_reporting/model_training_metadata.json")
TRANSFORMERS_PATH = Path("data/04_feature/feature_transformers.pkl")


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.metadata: dict[str, Any] | None = None
        self.transformers: dict[str, Any] | None = None
        self.error: str | None = None
        self.load()

    def load(self) -> None:
        try:
            with METADATA_PATH.open("r", encoding="utf-8") as file:
                self.metadata = json.load(file)

            with TRANSFORMERS_PATH.open("rb") as file:
                self.transformers = pickle.load(file)

            self.model = mlflow.sklearn.load_model(MODEL_URI)
            self.error = None

        except Exception as exc:
            self.model = None
            self.metadata = None
            self.transformers = None
            self.error = str(exc)

    @property
    def ready(self) -> bool:
        return (
            self.model is not None
            and self.metadata is not None
            and self.transformers is not None
        )

    def status(self) -> dict[str, Any]:
        return {
            "model_ready": self.ready,
            "model_uri": MODEL_URI,
            "metadata_path": str(METADATA_PATH),
            "feature_transformers_path": str(TRANSFORMERS_PATH),
            "error": self.error,
        }

    def metadata_info(self) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError(f"Model is not ready: {self.error}")

        assert self.metadata is not None

        return {
            "model_uri": MODEL_URI,
            "selected_model_family": self.metadata.get("selected_model_family"),
            "threshold": self.metadata.get("threshold"),
            "target_column": self.metadata.get("target_column"),
            "id_column": self.metadata.get("id_column"),
            "n_features": len(self.metadata.get("feature_columns", [])),
        }

    def predict_one(
        self,
        features: dict[str, Any],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        return self.predict_many([features], threshold=threshold)[0]

    def predict_many(
        self,
        rows: list[dict[str, Any]],
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if not self.ready:
            raise RuntimeError(f"Model is not ready: {self.error}")

        assert self.model is not None
        assert self.metadata is not None
        assert self.transformers is not None

        raw_df = pd.DataFrame(rows)

        id_col = self.metadata.get("id_column", "loan_sequence_number")
        target_col = self.metadata.get("target_column", "default")
        feature_columns = self.metadata["feature_columns"]

        ids = (
            raw_df[id_col].astype(str).tolist()
            if id_col in raw_df.columns
            else [None] * len(raw_df)
        )

        cleaned_df, _ = clean_origination_data(raw_df)
        cleaned_df = drop_unused_columns(cleaned_df)
        engineered_df = engineer_date_features(cleaned_df)
        feature_df = apply_feature_transformers(engineered_df, self.transformers)

        X = feature_df.drop(
            columns=[id_col, target_col],
            errors="ignore",
        )
        X = X.reindex(columns=feature_columns, fill_value=0)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            probabilities = self.model.predict(X)

        threshold_value = (
            float(threshold)
            if threshold is not None
            else float(self.metadata.get("threshold", 0.5))
        )

        predictions = (probabilities >= threshold_value).astype(int)

        results = []
        for loan_id, probability, prediction in zip(ids, probabilities, predictions):
            results.append(
                {
                    "loan_sequence_number": loan_id,
                    "prediction": int(prediction),
                    "probability_default": float(probability),
                    "threshold": threshold_value,
                    "model_uri": MODEL_URI,
                }
            )

        return results
