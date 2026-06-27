import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import cloudpickle

class ModelNotReadyError(RuntimeError):
    """Raised when the API is running but no trained model is available."""

    pass


class ModelService:
    """Model loading and prediction service for mortgage default serving.

    Our current architecture:

    1. model_train pipeline selects the best model.
    2. It saves the selected model to:
       data/06_models/mlflow_model
    3. It saves training metadata to:
       data/08_reporting/model_training_metadata.json
    4. The serving API loads both files and uses the saved feature columns
       to align incoming request data.

    Environment variables:
        MODEL_URI:
            Defaults to data/06_models/mlflow_model

        MODEL_METADATA_PATH:
            Defaults to data/08_reporting/model_training_metadata.json

        MLFLOW_TRACKING_URI:
            Optional. Only needed if MODEL_URI is a runs:/ or models:/ URI.

        ALLOW_DUMMY_MODEL:
            Optional. If true, API can return dummy predictions when no model exists.
    """

    def __init__(self) -> None:
        self.model_uri = os.getenv("MODEL_URI", "data/06_models/mlflow_model")
        self.metadata_path = Path(
            os.getenv(
                "MODEL_METADATA_PATH",
                "data/08_reporting/model_training_metadata.json",
            )
        )
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.allow_dummy_model = (
            os.getenv("ALLOW_DUMMY_MODEL", "false").lower() == "true"
        )

        self.load_error: str | None = None
        self.model: Any | None = None
        self.metadata: dict[str, Any] | None = None

        self._load()

    def _load(self) -> None:
        """Load metadata and model without crashing the API process."""
        try:
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)

            self.metadata = self._load_metadata()
            self.model = self._load_model()

        except Exception as exc:
            self.load_error = str(exc)
            self.model = None
            self.metadata = None

    def _load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata file not found: {self.metadata_path}"
            )

        with self.metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        if "feature_columns" not in metadata:
            raise ValueError(
                "Model metadata is missing required key: 'feature_columns'"
            )

        return metadata

    def _load_model(self) -> Any:
        """Load the selected model.

        Main path:
            data/06_models/mlflow_model

        Prefer MLflow's sklearn loader because it preserves predict_proba.
        If that fails locally, fall back to directly loading the pickle artifact
        inside the MLflow model folder.
        """
        model_uri = self.model_uri

        if self._is_local_path(model_uri):
            model_path = Path(model_uri)

            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

            if not (model_path / "MLmodel").exists():
                raise FileNotFoundError(
                    f"Path exists but is not an MLflow model folder: {model_path}"
                )

            try:
                return mlflow.sklearn.load_model(model_uri)
            except Exception as mlflow_exc:
                direct_model = self._load_local_mlflow_pickle(model_path)

                if direct_model is not None:
                    return direct_model

                raise RuntimeError(
                    "Failed to load local MLflow model with mlflow.sklearn.load_model, "
                    "and no local pickle artifact could be loaded. "
                    f"Original MLflow error: {mlflow_exc}"
                ) from mlflow_exc

        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception:
            return mlflow.pyfunc.load_model(model_uri)

    def _is_local_path(self, uri: str) -> bool:
        """Return True for local paths, False for MLflow runs:/ or models:/ URIs."""
        return not uri.startswith(("runs:/", "models:/"))

    def _load_local_mlflow_pickle(self, model_path: Path) -> Any | None:
        """Load sklearn model directly from a local MLflow model folder.

        This fallback is only for local serving. It reads the MLmodel file to find
        the actual sklearn pickle artifact, then skips invalid/empty pickle files.
        """
        candidate_paths: list[Path] = []

        mlmodel_path = model_path / "MLmodel"

        if mlmodel_path.exists():
            mlmodel_text = mlmodel_path.read_text(encoding="utf-8")

            for line in mlmodel_text.splitlines():
                stripped = line.strip()

                if stripped.startswith("pickled_model:"):
                    relative_path = stripped.split(":", 1)[1].strip().strip("'\"")
                    candidate_paths.append(model_path / relative_path)

                if stripped.startswith("data:"):
                    relative_path = stripped.split(":", 1)[1].strip().strip("'\"")
                    if relative_path.endswith(".pkl"):
                        candidate_paths.append(model_path / relative_path)

        candidate_paths.extend(
            [
                model_path / "model.pkl",
                model_path / "data" / "model.pkl",
                model_path / "sklearn_model.pkl",
                model_path / "data" / "sklearn_model.pkl",
            ]
        )

        candidate_paths.extend(model_path.rglob("*.pkl"))

        seen: set[Path] = set()
        errors: list[str] = []

        for candidate_path in candidate_paths:
            candidate_path = candidate_path.resolve()

            if candidate_path in seen:
                continue

            seen.add(candidate_path)

            if not candidate_path.exists() or not candidate_path.is_file():
                continue

            if candidate_path.stat().st_size == 0:
                errors.append(f"{candidate_path}: empty file")
                continue

            try:
                with candidate_path.open("rb") as file:
                    return cloudpickle.load(file)
            except Exception as exc:
                errors.append(f"{candidate_path}: {exc}")
                continue

        if errors:
            self.load_error = "Could not load any pickle candidate. Tried: " + " | ".join(
                errors
            )

        return None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.metadata is not None

    def status(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "model_uri": self.model_uri,
            "metadata_path": str(self.metadata_path),
            "selected_model_family": (
                self.metadata.get("selected_model_family")
                if self.metadata
                else None
            ),
            "threshold": self.get_threshold() if self.metadata else None,
            "load_error": self.load_error,
        }

    def get_threshold(self, override_threshold: float | None = None) -> float:
        if override_threshold is not None:
            return float(override_threshold)

        if self.metadata is None:
            return 0.5

        return float(self.metadata.get("threshold", 0.5))

    def _prepare_input(self, rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.metadata is None:
            raise ModelNotReadyError("Model metadata is not loaded.")

        if not rows:
            raise ValueError("No rows provided for prediction.")

        input_df = pd.DataFrame(rows)

        feature_columns = self.metadata["feature_columns"]
        target_column = self.metadata.get("target_column", "default")
        id_column = self.metadata.get("id_column", "loan_sequence_number")

        output_df = pd.DataFrame(index=input_df.index)

        if id_column in input_df.columns:
            output_df[id_column] = input_df[id_column].to_numpy()

        X = input_df.copy()

        for col in [target_column, id_column]:
            if col in X.columns:
                X = X.drop(columns=[col])

        # Critical architecture point:
        # serving data must match the exact training feature columns.
        # Missing columns become 0; extra request columns are ignored.
        X = X.reindex(columns=feature_columns, fill_value=0)

        return X, output_df

    def predict_many(
        self,
        rows: list[dict[str, Any]],
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if self.model is None or self.metadata is None:
            if self.allow_dummy_model:
                return [self._dummy_prediction(row) for row in rows]

            raise ModelNotReadyError(
                "No trained model is currently loaded. "
                f"Load error: {self.load_error}"
            )

        X, output_df = self._prepare_input(rows)
        threshold_value = self.get_threshold(threshold)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[:, 1]
            predictions = (probabilities >= threshold_value).astype(int)
        else:
            # Fallback for pyfunc models where predict_proba is unavailable.
            prediction_output = self.model.predict(X)
            predictions = self._normalise_prediction_output(prediction_output)
            probabilities = np.full(shape=len(predictions), fill_value=np.nan)

        output_df["prediction"] = predictions.astype(int)
        output_df["probability_default"] = probabilities
        output_df["threshold"] = threshold_value
        output_df["selected_model_family"] = self.metadata.get(
            "selected_model_family"
        )
        output_df["model_uri"] = self.model_uri
        output_df["model_status"] = "loaded"

        records = output_df.to_dict(orient="records")
        return [self._json_safe_record(record) for record in records]

    def predict_one(
        self,
        features: dict[str, Any],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        return self.predict_many([features], threshold=threshold)[0]

    def _normalise_prediction_output(self, prediction_output: Any) -> np.ndarray:
        if isinstance(prediction_output, pd.DataFrame):
            return prediction_output.iloc[:, 0].to_numpy()

        if isinstance(prediction_output, pd.Series):
            return prediction_output.to_numpy()

        return np.asarray(prediction_output).reshape(-1)

    def _dummy_prediction(self, row: dict[str, Any]) -> dict[str, Any]:
        id_column = "loan_sequence_number"

        result = {
            "prediction": 0,
            "probability_default": None,
            "threshold": None,
            "selected_model_family": None,
            "model_uri": self.model_uri,
            "model_status": "dummy_no_model_loaded",
        }

        if id_column in row:
            result[id_column] = row[id_column]

        return result

    def _json_safe_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {key: self._json_safe_value(value) for key, value in record.items()}

    def _json_safe_value(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if pd.isna(value):
            return None

        return value