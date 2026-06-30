"""Nodes for the model_predict pipeline.

End-to-end inference pipeline starting from raw origination data only —
no performance file. This is the realistic production scenario: a brand
new loan application has origination details (credit score, LTV, etc.)
but no performance history, because it hasn't happened yet. The model
predicts risk purely from origination features.

Combines:
- Raw ingestion (origination only) + cleaning, reusing the same logic
  as data_ingestion / data_cleaning so inference data is preprocessed
  identically to training data.
- Feature preparation, reusing model_train's stateless transformations
  plus the already-fitted feature_transformers.
- Scoring — contributed by a teammate, kept mostly unchanged. The
  evaluation step (comparing against ground truth) is no longer part of
  this pipeline by default, since real new loans have no known outcome;
  it's kept as an optional node for cases where you deliberately feed in
  historical data to sanity-check the model (see evaluate_labeled_inference).
"""
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

from mortgage_default.pipelines.data_ingestion.nodes import (
    assign_origination_columns_names,
)
from mortgage_default.pipelines.model_train.nodes import (
    apply_feature_transformers,
    drop_unused_columns,
    engineer_date_features,
)


def ingest_inference_data(raw_origination_inference: pd.DataFrame) -> pd.DataFrame:
    """Assigns column names to raw origination-only data — no performance
    file, no target variable. This mirrors a real production scenario:
    a new loan application only has origination details available.

    Args:
        raw_origination_inference: raw origination CSV (pipe-delimited,
            no header) for the inference batch.
    Returns:
        Named origination DataFrame, no 'default' column.
    """
    return assign_origination_columns_names(raw_origination_inference)


def clean_inference_data(origination_data_inference_named: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cleans origination-only inference data using the same sentinel
    handling and type coercion logic as data_cleaning, adapted to work
    without a 'default' column (since this data has no known outcome).

    Args:
        origination_data_inference_named: named origination data, no
            target column.
    Returns:
        cleaned DataFrame (no 'default' column), cleaning statistics dict.
    """
    from mortgage_default.pipelines.data_cleaning.nodes import (
        CATEGORICAL_COLUMNS,
        CATEGORICAL_UNKNOWN_MAP,
        COLUMNS_TO_DROP,
        NUMERIC_COLUMNS,
        NUMERIC_SENTINEL_MAP,
        _count_missing,
    )

    df = origination_data_inference_named.copy()
    stats: dict[str, Any] = {
        "rows_before": int(len(df)),
        "columns_before": int(df.shape[1]),
    }

    df.columns = df.columns.str.strip().str.lower()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, sentinels in NUMERIC_SENTINEL_MAP.items():
        if col not in df.columns:
            continue
        mask = df[col].isin(sentinels)
        stats[f"{col}_sentinel_count"] = int(mask.sum())
        df.loc[mask, col] = pd.NA

    integer_like_columns = [
        "number_of_units", "number_of_borrowers", "original_loan_term", "postal_code",
    ]
    for col in integer_like_columns:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    for col in ["first_payment_date", "maturity_date"]:
        if col not in df.columns:
            continue
        df[col] = pd.to_datetime(
            df[col].astype("string").str.strip(), format="%Y%m", errors="coerce",
        )
        stats[f"{col}_parse_failed_count"] = int(df[col].isna().sum())

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.upper()

    for col, sentinels in CATEGORICAL_UNKNOWN_MAP.items():
        if col not in df.columns:
            continue
        sentinels_as_string = {str(value) for value in sentinels}
        mask = df[col].isin(sentinels_as_string)
        stats[f"{col}_unknown_count"] = int(mask.sum())
        df.loc[mask, col] = "UNKNOWN"

    if "msa" in df.columns:
        stats["msa_missing_count"] = int(df["msa"].isna().sum())
        df["msa"] = df["msa"].astype("Int64")

    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    stats["dropped_columns"] = cols_to_drop
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if "loan_sequence_number" in df.columns:
        stats["duplicate_loan_sequence_number_count"] = int(
            df["loan_sequence_number"].duplicated().sum()
        )

    stats["missing_values_after_cleaning"] = _count_missing(df)
    stats["rows_after"] = int(len(df))
    stats["columns_after"] = int(df.shape[1])
    stats["rows_removed"] = int(stats["rows_before"] - stats["rows_after"])

    return df, stats


def prepare_inference_features(
    origination_data_inference_cleaned: pd.DataFrame,
    feature_transformers: dict,
) -> pd.DataFrame:
    """Applies the exact same stateless transformations used in
    model_train (drop columns, date engineering) plus the already-fitted
    feature_transformers (imputation medians, label encodings learned
    only on 2000-2002). No target column involved at any point.

    Args:
        origination_data_inference_cleaned: cleaned inference data, no
            target column.
        feature_transformers: artifact from model_train.
    Returns:
        Transformed DataFrame ready for run_model_inference.
    """
    df = origination_data_inference_cleaned.copy()

    id_col = "loan_sequence_number"
    loan_ids = df[id_col].copy() if id_col in df.columns else None

    df = drop_unused_columns(df)
    df = engineer_date_features(df)
    df = apply_feature_transformers(df, feature_transformers)

    if loan_ids is not None and id_col not in df.columns:
        df.insert(0, id_col, loan_ids.reset_index(drop=True))

    return df


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
    """Load selected MLflow model and generate predictions.

    Works whether or not features_inference contains the target column —
    real production data won't have it; historical sanity-check data
    optionally fed through this same pipeline might.
    """

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
        parameters.get("threshold") or
        model_training_metadata.get("threshold", 0.5)
    )

    feature_columns = model_training_metadata["feature_columns"]

    output = pd.DataFrame(index=features_inference.index)

    if id_column in features_inference.columns:
        output[id_column] = features_inference[id_column].to_numpy()

    X = features_inference.copy()

    for col in [target_column, id_column]:
        if col in X.columns:
            X = X.drop(columns=[col])

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
    """Optional: compare inference predictions against a known target,
    for cases where historical data was deliberately fed through this
    pipeline to sanity-check the model. Real production data has no
    target column, so this node would simply not be wired into the
    pipeline for genuine new-loan scoring.
    """

    target_column = parameters.get("target_column", "default")

    if target_column not in features_inference.columns:
        raise ValueError(
            f"Cannot evaluate labeled inference because '{target_column}' "
            "is missing from features_inference. This is expected for "
            "genuine new-loan scoring — only use this node when "
            "deliberately feeding historical data with known outcomes."
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
