"""Nodes for the model_evaluate pipeline.

Applies an already-trained model (from model_train) to the temporal test
set exactly once. Kept as a separate pipeline so it can be re-run on its
own — e.g. against a different test slice, or after a threshold change —
without repeating the expensive random search in model_train.
"""
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _safe_metric(metric_fn, y_true, y_score_or_pred) -> float | None:
    try:
        return float(metric_fn(y_true, y_score_or_pred))
    except Exception:
        return None


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    prefix: str,
    beta: float = 2.0,
) -> dict[str, float | None]:
    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, predictions)),
        f"{prefix}_precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        f"{prefix}_recall": float(recall_score(y_true, predictions, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, predictions, zero_division=0)),
        f"{prefix}_fbeta": float(
            fbeta_score(y_true, predictions, beta=beta, zero_division=0)
        ),
        f"{prefix}_average_precision": _safe_metric(
            average_precision_score, y_true, probabilities
        ),
    }

    if y_true.nunique() == 2:
        metrics[f"{prefix}_roc_auc"] = _safe_metric(
            roc_auc_score, y_true, probabilities
        )
    else:
        metrics[f"{prefix}_roc_auc"] = None

    return metrics


def _prepare_xy(
    data: pd.DataFrame,
    target_column: str,
    id_column: str | None,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series | None]:
    df = data.copy()
    y = None

    if target_column in df.columns:
        y = df[target_column].astype(int)
        df = df.drop(columns=[target_column])

    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])

    df = df.reindex(columns=feature_columns, fill_value=0)
    return df, y


def _predict_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        score_min = np.min(scores)
        score_max = np.max(scores)
        if score_max == score_min:
            return np.full(shape=len(scores), fill_value=0.5)
        return (scores - score_min) / (score_max - score_min)
    return model.predict(X).astype(float)


def _save_sklearn_model(model, path: Path, signature: Any, input_example: pd.DataFrame) -> None:
    import shutil

    if path.exists():
        shutil.rmtree(path)

    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(path),
        signature=signature,
        input_example=input_example,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )


def evaluate_model(
    selected_model: Any,
    features_test: pd.DataFrame,
    train_metadata: dict[str, Any],
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Evaluate the already-trained model on the temporal test set.

    This is the single point of contact with features_test — used once,
    for final reporting, never during model/hyperparameter/threshold
    selection (that all happens in model_train using only train/val data).

    Can be re-run independently of model_train as long as selected_model
    and train_metadata are persisted in the catalog (Pickle / JSON).

    Args:
        selected_model: fitted sklearn Pipeline, produced by model_train.
        features_test: transformed test data (years >= test_start_year),
            target column included.
        train_metadata: metadata dict produced by model_train (feature
            columns, threshold, model family, MLflow run ids, etc.)
        parameters: model_train parameter group (for model_output_path,
            tracking_uri).
    Returns:
        metrics_output: final test metrics + run metadata.
        full_metadata: train_metadata merged with evaluation info, ready
            for the model_explainability pipeline to consume.
        test_output: per-loan predictions DataFrame.
    """
    target_column = train_metadata["target_column"]
    id_column = train_metadata["id_column"]
    feature_columns = train_metadata["feature_columns"]
    selected_threshold = float(train_metadata["threshold"])
    beta = float(train_metadata.get("beta", 2.0))

    X_test, y_test = _prepare_xy(
        features_test, target_column=target_column,
        id_column=id_column, feature_columns=feature_columns,
    )

    if y_test is None:
        raise ValueError(f"Target column '{target_column}' not found in features_test.")
    if y_test.nunique() < 2:
        raise ValueError("features_test must contain both classes 0 and 1.")

    final_test_probabilities = _predict_probabilities(selected_model, X_test)
    final_test_predictions = (final_test_probabilities >= selected_threshold).astype(int)

    final_test_metrics = _classification_metrics(
        y_true=y_test, probabilities=final_test_probabilities,
        predictions=final_test_predictions, prefix="final_test", beta=beta,
    )

    logger.info(
        "Avaliação final — AUC: %.4f | Recall: %.4f | Precision: %.4f | F1: %.4f",
        final_test_metrics.get("final_test_roc_auc") or float("nan"),
        final_test_metrics["final_test_recall"],
        final_test_metrics["final_test_precision"],
        final_test_metrics["final_test_f1"],
    )

    tracking_uri = parameters.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    parent_run_id = train_metadata.get("parent_run_id")
    if parent_run_id:
        with mlflow.start_run(run_id=parent_run_id):
            mlflow.log_metrics({
                key: value for key, value in final_test_metrics.items()
                if value is not None
            })

    model_output_path = Path(
        parameters.get("model_output_path", "data/06_models/mlflow_model")
    )

    input_example = X_test.head(5)
    signature = infer_signature(input_example, selected_model.predict(input_example))
    _save_sklearn_model(
        model=selected_model, path=model_output_path,
        signature=signature, input_example=input_example,
    )

    metrics_output = {
        **final_test_metrics,
        "parent_run_id": train_metadata.get("parent_run_id"),
        "run_name": train_metadata.get("run_name"),
        "run_timestamp": train_metadata.get("run_timestamp"),
        "selected_model_family": train_metadata.get("selected_model_family"),
        "selected_child_run_id": train_metadata.get("selected_child_run_id"),
        "selected_child_run_name": train_metadata.get("selected_child_run_name"),
        "selection_metric": train_metadata.get("selection_metric"),
        "selected_threshold": selected_threshold,
        "threshold_strategy": train_metadata.get("threshold_strategy"),
        "beta": beta,
        "model_output_path": str(model_output_path),
        "n_features": len(feature_columns),
        "n_test_rows": int(len(X_test)),
    }

    full_metadata = {
        **train_metadata,
        "model_output_path": str(model_output_path),
        "final_test_metrics": final_test_metrics,
    }

    test_output = pd.DataFrame({
        "actual_default": y_test.to_numpy(),
        "prediction": final_test_predictions,
        "probability_default": final_test_probabilities,
        "selected_model_family": train_metadata.get("selected_model_family"),
    })

    if id_column in features_test.columns:
        test_output.insert(0, id_column, features_test[id_column].to_numpy())

    return _to_builtin(metrics_output), _to_builtin(full_metadata), test_output
