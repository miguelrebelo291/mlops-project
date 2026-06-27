from typing import Any

import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd


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
