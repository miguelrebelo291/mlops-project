"""Nodes for the model_explainability pipeline.

Generates SHAP feature importance for the model selected and evaluated
by model_train / model_evaluate. Reads the model_uri dynamically from
train_metadata (written by model_train), since it changes on every run.
"""
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap


def _extract_shap_values(explanation: Any) -> np.ndarray:
    """Convert SHAP output to a 2D array: rows x features."""
    values = explanation.values

    if isinstance(values, list):
        values = values[1]

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 1]

    return values


def generate_shap_report(
    features_test: pd.DataFrame,
    train_metadata: dict,
    parameters: dict,
) -> dict:
    """Generate SHAP feature importance table and summary plot.

    Args:
        features_test: transformed test data (from model_train), target
            column included.
        train_metadata: metadata dict produced by model_train, containing
            model_uri, feature_columns, target_column, id_column.
        parameters: model_explainability parameter group (output_dir,
            sample_size, tracking_uri, etc.)
    Returns:
        Dictionary summarizing the SHAP run, with paths to saved artifacts.
    """
    tracking_uri = parameters.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_uri = train_metadata["model_uri"]
    target_col = train_metadata.get("target_column", "default")
    id_col = train_metadata.get("id_column", "loan_sequence_number")
    feature_columns = train_metadata["feature_columns"]

    output_dir = Path(parameters.get("output_dir", "data/08_reporting"))
    sample_size = int(parameters.get("sample_size", 200))
    random_state = int(parameters.get("random_state", 42))
    max_display = int(parameters.get("max_display", 20))

    output_dir.mkdir(parents=True, exist_ok=True)

    model = mlflow.sklearn.load_model(model_uri)

    data = features_test.copy()

    X = data.drop(columns=[target_col, id_col], errors="ignore")
    X = X.reindex(columns=feature_columns, fill_value=0)

    # SHAP needs numeric model input only — handles pandas nullable
    # dtypes (Int64, etc.) that break shap's internal np.isclose checks.
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    X = X.astype("float64")

    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X

    def predict_default_probability(input_data):
        input_df = pd.DataFrame(input_data, columns=X_sample.columns)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(input_df)[:, 1]
        return model.predict(input_df)

    background_size = min(50, len(X_sample))
    background = shap.sample(X_sample, background_size, random_state=random_state)

    explainer = shap.Explainer(
        predict_default_probability,
        background,
    )

    explanation = explainer(X_sample)
    shap_values = _extract_shap_values(explanation)

    importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    importance_path = output_dir / "shap_feature_importance.csv"
    plot_path = output_dir / "shap_summary_plot.png"

    importance_df.to_csv(importance_path, index=False)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "model_uri": model_uri,
        "selected_model_family": train_metadata.get("selected_model_family"),
        "n_rows_explained": int(len(X_sample)),
        "n_features": int(X_sample.shape[1]),
        "top_feature": str(importance_df.iloc[0]["feature"]),
        "top_feature_mean_abs_shap": float(importance_df.iloc[0]["mean_abs_shap"]),
        "importance_path": str(importance_path),
        "summary_plot_path": str(plot_path),
    }
