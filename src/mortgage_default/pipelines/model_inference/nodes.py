from typing import Any

import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def _normalise_prediction_output(prediction_output: Any) -> np.ndarray:
    if isinstance(prediction_output, pd.DataFrame):
        return prediction_output.iloc[:, 0].to_numpy()

    if isinstance(prediction_output, pd.Series):
        return prediction_output.to_numpy()

    return np.asarray(prediction_output).reshape(-1)


def run_model_inference(
    features_inference: pd.DataFrame,
    model_training_metadata: dict[str, Any],
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Load selected MLflow model and generate predictions."""

    model_uri = parameters.get(
        "model_uri",
        model_training_metadata.get("model_output_path", "data/06_models/mlflow_model"),
    )
    target_column = parameters.get(
        "target_column",
        model_training_metadata.get("target_column", "default"),
    )
    id_column = parameters.get(
        "id_column",
        model_training_metadata.get("id_column", "loan_sequence_number"),
    )
    threshold = float(
        parameters.get(
            "threshold",
            model_training_metadata.get("threshold", 0.5),
        )
    )

    feature_columns = model_training_metadata["feature_columns"]

    output = pd.DataFrame(index=features_inference.index)

    if id_column in features_inference.columns:
        output[id_column] = features_inference[id_column].to_numpy()

    X = features_inference.copy()

    for col in [target_column, id_column]:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Match training columns exactly.
    # Missing columns become 0; extra columns are ignored.
    X = X.reindex(columns=feature_columns, fill_value=0)

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        model = mlflow.pyfunc.load_model(model_uri)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
    else:
        prediction_output = model.predict(X)
        predictions = _normalise_prediction_output(prediction_output).astype(int)
        probabilities = np.full(shape=len(predictions), fill_value=np.nan)

    output["prediction"] = predictions
    output["probability_default"] = probabilities
    output["threshold"] = threshold

    if "selected_model_family" in model_training_metadata:
        output["selected_model_family"] = model_training_metadata[
            "selected_model_family"
        ]

    return output.reset_index(drop=True)


def evaluate_labeled_inference(
    features_inference: pd.DataFrame,
    model_inference_predictions: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare inference predictions against the known target."""

    target_column = parameters.get("target_column", "default")

    if target_column not in features_inference.columns:
        raise ValueError(
            f"Cannot evaluate labeled inference because '{target_column}' "
            "is missing from features_inference."
        )

    predictions = model_inference_predictions.copy()
    y_true = features_inference[target_column].astype(int).reset_index(drop=True)

    if "prediction" not in predictions.columns:
        raise ValueError("model_inference_predictions must contain 'prediction'.")

    y_pred = predictions["prediction"].astype(int).reset_index(drop=True)

    if "probability_default" in predictions.columns:
        y_prob = predictions["probability_default"].astype(float).reset_index(drop=True)
    else:
        y_prob = y_pred.astype(float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "n_rows": int(len(y_true)),
        "actual_default_rate": float(y_true.mean()),
        "predicted_default_rate": float(y_pred.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    metrics["test_year"] = parameters.get("test_year")

    if y_true.nunique() == 2:
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["average_precision"] = None
        metrics["roc_auc"] = None

    evaluated_predictions = predictions.copy()
    evaluated_predictions["actual_default"] = y_true.to_numpy()
    evaluated_predictions["is_correct"] = (
        evaluated_predictions["prediction"].astype(int)
        == evaluated_predictions["actual_default"].astype(int)
    )

    return metrics, evaluated_predictions