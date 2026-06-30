import json
import os
import pickle
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd

from mortgage_default.pipelines.data_cleaning.nodes import clean_origination_data
from mortgage_default.pipelines.feature_engineering.nodes import (
    apply_feature_transformers,
    drop_unused_columns,
    engineer_date_features,
)


MODEL_URI = os.getenv("MODEL_URI", "data/06_models/mlflow_model")
METADATA_PATH = Path(
    os.getenv("MODEL_METADATA_PATH", "data/08_reporting/model_training_metadata.json")
)
TRANSFORMERS_PATH = Path(
    os.getenv("FEATURE_TRANSFORMERS_PATH", "data/04_feature/feature_transformers.pkl")
)


class ModelService:
    def __init__(self) -> None:
        self.model: Any | None = None
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

            try:
                self.model = mlflow.sklearn.load_model(MODEL_URI)
            except Exception:
                self.model = mlflow.pyfunc.load_model(MODEL_URI)

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

    def _require_ready(self) -> None:
        if not self.ready:
            raise RuntimeError(f"Model is not ready: {self.error}")

    def _get_threshold(self, threshold: float | None = None) -> float:
        self._require_ready()
        assert self.metadata is not None

        if threshold is not None:
            return float(threshold)

        return float(self.metadata.get("threshold", 0.5))

    def _ensure_transformer_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make serving robust to partial raw payloads.

        apply_feature_transformers expects the train-time numeric/categorical
        columns to exist. If a request misses some of them, create them before
        transformation so numeric columns get median-imputed and categorical
        columns become all-zero dummies.
        """
        self._require_ready()
        assert self.transformers is not None

        df = df.copy()

        for col in self.transformers.get("numeric_cols", []):
            if col not in df.columns:
                df[col] = np.nan

        for col in self.transformers.get("categorical_cols", []):
            if col not in df.columns:
                df[col] = pd.NA

        return df

    def _prepare_features(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, list[str | None]]:
        self._require_ready()
        assert self.metadata is not None
        assert self.transformers is not None

        if not rows:
            raise ValueError("Request must contain at least one row.")

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

        engineered_df = self._ensure_transformer_columns(engineered_df)

        feature_df = apply_feature_transformers(
            engineered_df,
            self.transformers,
        )

        X = feature_df.drop(
            columns=[id_col, target_col],
            errors="ignore",
        )

        X = X.reindex(columns=feature_columns, fill_value=0)

        return X, ids

    def _predict_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        self._require_ready()
        assert self.model is not None

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]

        prediction_output = self.model.predict(X)

        if isinstance(prediction_output, pd.DataFrame):
            values = prediction_output.iloc[:, 0].to_numpy()
        elif isinstance(prediction_output, pd.Series):
            values = prediction_output.to_numpy()
        else:
            values = np.asarray(prediction_output).reshape(-1)

        return values.astype(float)

    def status(self) -> dict[str, Any]:
        return {
            "model_ready": self.ready,
            "model_uri": MODEL_URI,
            "metadata_path": str(METADATA_PATH),
            "feature_transformers_path": str(TRANSFORMERS_PATH),
            "error": self.error,
        }

    def metadata_info(self) -> dict[str, Any]:
        self._require_ready()
        assert self.metadata is not None

        return {
            "model_uri": MODEL_URI,
            "selected_model_family": self.metadata.get("selected_model_family"),
            "threshold": self.metadata.get("threshold"),
            "fallback_threshold": self.metadata.get("fallback_threshold"),
            "threshold_strategy": self.metadata.get("threshold_strategy"),
            "selection_metric": self.metadata.get("selection_metric"),
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
        self._require_ready()
        assert self.metadata is not None

        X, ids = self._prepare_features(rows)
        probabilities = self._predict_probabilities(X)

        threshold_value = self._get_threshold(threshold)
        predictions = (probabilities >= threshold_value).astype(int)

        selected_model_family = self.metadata.get("selected_model_family")

        results = []

        for loan_id, probability, prediction in zip(ids, probabilities, predictions):
            results.append(
                {
                    "loan_sequence_number": loan_id,
                    "prediction": int(prediction),
                    "probability_default": float(probability),
                    "threshold": threshold_value,
                    "selected_model_family": selected_model_family,
                    "model_uri": MODEL_URI,
                }
            )

        return results