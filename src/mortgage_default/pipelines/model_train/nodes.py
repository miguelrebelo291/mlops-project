import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _safe_metric(metric_fn, y_true, y_score_or_pred) -> float | None:
    try:
        return float(metric_fn(y_true, y_score_or_pred))
    except Exception:
        return None


def _to_builtin(value: Any) -> Any:
    """Convert numpy/pandas values to JSON-safe Python values."""
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
) -> dict[str, float | None]:
    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, predictions)),
        f"{prefix}_precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        f"{prefix}_recall": float(recall_score(y_true, predictions, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, predictions, zero_division=0)),
        f"{prefix}_average_precision": _safe_metric(
            average_precision_score,
            y_true,
            probabilities,
        ),
    }

    if y_true.nunique() == 2:
        metrics[f"{prefix}_roc_auc"] = _safe_metric(
            roc_auc_score,
            y_true,
            probabilities,
        )
    else:
        metrics[f"{prefix}_roc_auc"] = None

    return metrics


def _prepare_xy(
    data: pd.DataFrame,
    target_column: str,
    id_column: str | None,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, list[str], list[str]]:
    df = data.copy()

    y = None
    if target_column in df.columns:
        y = df[target_column].astype(int)
        df = df.drop(columns=[target_column])

    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])

    if feature_columns is None:
        non_numeric_columns = df.select_dtypes(
            exclude=["number", "bool"]
        ).columns.tolist()
        df = df.drop(columns=non_numeric_columns)
        feature_columns = df.columns.tolist()
    else:
        non_numeric_columns = []
        df = df.reindex(columns=feature_columns, fill_value=0)

    return df, y, feature_columns, non_numeric_columns


def _predict_probabilities(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return probability of default/class 1."""
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


def _build_model_spaces(random_state: int) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Define model families and random-search parameter spaces."""
    logistic_regression = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )

    extra_trees = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": (
            logistic_regression,
            {
                "classifier__C": np.logspace(-3, 2, 20).tolist(),
                "classifier__solver": ["lbfgs", "liblinear"],
                "classifier__penalty": ["l2"],
                "classifier__class_weight": [None, "balanced"],
                "classifier__max_iter": [500, 1000, 2000],
            },
        ),
        "random_forest": (
            random_forest,
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 5, 10, 20, 30],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4, 8],
                "classifier__max_features": ["sqrt", "log2", None],
                "classifier__class_weight": [None, "balanced", "balanced_subsample"],
            },
        ),
        "extra_trees": (
            extra_trees,
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 5, 10, 20, 30],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4, 8],
                "classifier__max_features": ["sqrt", "log2", None],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }


def _search_results_to_frame(
    search: RandomizedSearchCV,
    model_family: str,
) -> pd.DataFrame:
    """Convert sklearn cv_results_ to a stable dataframe."""
    results = pd.DataFrame(search.cv_results_)
    results.insert(0, "model_family", model_family)

    results["params_json"] = results["params"].apply(
        lambda params: json.dumps(params, default=str, sort_keys=True)
    )
    results["is_best_candidate"] = False
    results.loc[search.best_index_, "is_best_candidate"] = True

    results = results.drop(columns=["params"])

    # Parquet does not like arbitrary Python objects. Make object columns safe.
    for col in results.columns:
        if results[col].dtype == "object":
            results[col] = results[col].astype(str)

    return results


def _log_sklearn_model(
    model: Pipeline,
    signature: Any,
    input_example: pd.DataFrame,
    artifact_name: str = "model",
) -> None:
    """Log sklearn model with MLflow compatibility across versions."""
    kwargs = {
        "sk_model": model,
        "signature": signature,
        "input_example": input_example,
        "serialization_format": mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    }

    try:
        mlflow.sklearn.log_model(
            name=artifact_name,
            **kwargs,
        )
    except TypeError:
        mlflow.sklearn.log_model(
            artifact_path=artifact_name,
            **kwargs,
        )


def _save_sklearn_model(
    model: Pipeline,
    path: Path,
    signature: Any,
    input_example: pd.DataFrame,
) -> None:
    """Save selected sklearn model as an MLflow model folder."""
    if path.exists():
        shutil.rmtree(path)

    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(path),
        signature=signature,
        input_example=input_example,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )


def train_validate_test_model(
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run random search for three model families and export the best model.

    Data split:
    - features_train/features_test come from feature_engineering
    - features_train is split again into train + validation
    - RandomizedSearchCV runs on train
    - Best candidate per family is evaluated on validation and test
    - Final model is selected by validation metric
    """
    target_column = parameters.get("target_column", "default")
    id_column = parameters.get("id_column", "loan_sequence_number")
    validation_size = parameters.get("validation_size", 0.2)
    random_state = parameters.get("random_state", 42)
    threshold = parameters.get("threshold", 0.5)

    experiment_name = parameters.get("experiment_name", "mortgage_default_model_search")
    run_name = parameters.get("run_name", "random_search_three_models")
    model_output_path = Path(
        parameters.get("model_output_path", "data/06_models/mlflow_model")
    )

    n_iter = int(parameters.get("n_iter", 8))
    requested_cv = int(parameters.get("cv", 3))
    n_jobs = int(parameters.get("n_jobs", -1))
    scoring = parameters.get("scoring", "roc_auc")
    selection_metric = parameters.get("selection_metric", "val_roc_auc")
    requested_models = parameters.get(
        "models",
        ["logistic_regression", "random_forest", "extra_trees"],
    )

    X_train_full, y_train_full, feature_columns, dropped_non_numeric_columns = (
        _prepare_xy(
            features_train,
            target_column=target_column,
            id_column=id_column,
        )
    )

    X_test, y_test, _, _ = _prepare_xy(
        features_test,
        target_column=target_column,
        id_column=id_column,
        feature_columns=feature_columns,
    )

    if y_train_full is None:
        raise ValueError(f"Target column '{target_column}' not found in features_train.")

    if y_test is None:
        raise ValueError(f"Target column '{target_column}' not found in features_test.")

    stratify = y_train_full if y_train_full.nunique() == 2 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify,
    )

    min_class_count = int(y_train.value_counts().min())
    cv = min(requested_cv, min_class_count)

    if cv < 2:
        raise ValueError(
            "Not enough examples in the minority class to run cross-validation."
        )

    model_spaces = _build_model_spaces(random_state=random_state)

    unknown_models = set(requested_models) - set(model_spaces)
    if unknown_models:
        raise ValueError(f"Unknown model names in parameters: {sorted(unknown_models)}")

    tracking_uri = parameters.get("tracking_uri")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    all_search_results: list[pd.DataFrame] = []
    model_summaries: list[dict[str, Any]] = []

    with mlflow.start_run(run_name=run_name) as parent_run:
        parent_run_id = parent_run.info.run_id

        mlflow.log_params(
            {
                "validation_size": validation_size,
                "threshold": threshold,
                "random_state": random_state,
                "n_iter": n_iter,
                "cv": cv,
                "requested_cv": requested_cv,
                "n_jobs": n_jobs,
                "scoring": scoring,
                "selection_metric": selection_metric,
                "n_features": len(feature_columns),
                "n_train_rows": len(X_train),
                "n_val_rows": len(X_val),
                "n_test_rows": len(X_test),
                "models": ",".join(requested_models),
            }
        )

        for model_family in requested_models:
            base_model, param_distributions = model_spaces[model_family]

            with mlflow.start_run(
                run_name=f"{model_family}_random_search",
                nested=True,
            ) as child_run:
                child_run_id = child_run.info.run_id

                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_distributions,
                    n_iter=n_iter,
                    scoring=scoring,
                    cv=cv,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    refit=True,
                    return_train_score=True,
                    verbose=1,
                )

                search.fit(X_train, y_train)

                best_model = search.best_estimator_

                val_probabilities = _predict_probabilities(best_model, X_val)
                val_predictions = (val_probabilities >= threshold).astype(int)

                test_probabilities = _predict_probabilities(best_model, X_test)
                test_predictions = (test_probabilities >= threshold).astype(int)

                val_metrics = _classification_metrics(
                    y_true=y_val,
                    probabilities=val_probabilities,
                    predictions=val_predictions,
                    prefix="val",
                )
                test_metrics = _classification_metrics(
                    y_true=y_test,
                    probabilities=test_probabilities,
                    predictions=test_predictions,
                    prefix="test",
                )

                model_metrics = {
                    **val_metrics,
                    **test_metrics,
                    "best_cv_score": float(search.best_score_),
                    "best_cv_rank": int(search.best_index_ + 1),
                }

                mlflow.log_params(
                    {
                        "model_family": model_family,
                        "best_params": json.dumps(search.best_params_, default=str),
                    }
                )
                mlflow.log_metrics(
                    {
                        key: value
                        for key, value in model_metrics.items()
                        if value is not None
                    }
                )

                input_example = X_train.head(5)
                signature = infer_signature(
                    input_example,
                    best_model.predict(input_example),
                )

                _log_sklearn_model(
                    model=best_model,
                    signature=signature,
                    input_example=input_example,
                    artifact_name="model",
                )

                search_results = _search_results_to_frame(
                    search=search,
                    model_family=model_family,
                )
                search_results["child_run_id"] = child_run_id
                search_results["parent_run_id"] = parent_run_id

                all_search_results.append(search_results)

                model_summaries.append(
                    {
                        "model_family": model_family,
                        "child_run_id": child_run_id,
                        "best_cv_score": float(search.best_score_),
                        "best_params": search.best_params_,
                        **model_metrics,
                    }
                )

        model_search_results = pd.concat(all_search_results, ignore_index=True)

        with TemporaryDirectory() as tmp_dir:
            search_results_path = Path(tmp_dir) / "model_search_results.csv"
            model_search_results.to_csv(search_results_path, index=False)
            mlflow.log_artifact(
                str(search_results_path),
                artifact_path="search_results",
            )

        def _selection_value(summary: dict[str, Any]) -> float:
            value = summary.get(selection_metric)
            if value is None:
                return float("-inf")
            return float(value)

        best_summary = max(model_summaries, key=_selection_value)
        best_model_family = best_summary["model_family"]

        # Refit the selected model family on train + validation using its best params.
        selected_base_model, _ = model_spaces[best_model_family]
        selected_base_model.set_params(**best_summary["best_params"])

        X_train_plus_val = pd.concat([X_train, X_val], axis=0)
        y_train_plus_val = pd.concat([y_train, y_val], axis=0)

        selected_base_model.fit(X_train_plus_val, y_train_plus_val)

        final_test_probabilities = _predict_probabilities(selected_base_model, X_test)
        final_test_predictions = (final_test_probabilities >= threshold).astype(int)

        final_test_metrics = _classification_metrics(
            y_true=y_test,
            probabilities=final_test_probabilities,
            predictions=final_test_predictions,
            prefix="final_test",
        )

        mlflow.log_metrics(
            {
                key: value
                for key, value in final_test_metrics.items()
                if value is not None
            }
        )
        mlflow.log_params(
            {
                "selected_model_family": best_model_family,
                "selected_child_run_id": best_summary["child_run_id"],
                "selected_best_params": json.dumps(
                    best_summary["best_params"],
                    default=str,
                ),
            }
        )

        input_example = X_train_plus_val.head(5)
        signature = infer_signature(
            input_example,
            selected_base_model.predict(input_example),
        )

        _log_sklearn_model(
            model=selected_base_model,
            signature=signature,
            input_example=input_example,
            artifact_name="selected_model",
        )

        _save_sklearn_model(
            model=selected_base_model,
            path=model_output_path,
            signature=signature,
            input_example=input_example,
        )

    metrics_output = {
        **final_test_metrics,
        "parent_run_id": parent_run_id,
        "selected_model_family": best_model_family,
        "selected_child_run_id": best_summary["child_run_id"],
        "selection_metric": selection_metric,
        "selection_metric_value": best_summary.get(selection_metric),
        "model_output_path": str(model_output_path),
        "n_features": len(feature_columns),
        "n_train_rows": int(len(X_train)),
        "n_val_rows": int(len(X_val)),
        "n_test_rows": int(len(X_test)),
        "n_random_search_rows": int(len(model_search_results)),
    }

    metadata = {
        "parent_run_id": parent_run_id,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "model_output_path": str(model_output_path),
        "target_column": target_column,
        "id_column": id_column,
        "threshold": threshold,
        "random_state": random_state,
        "n_iter": n_iter,
        "cv": cv,
        "scoring": scoring,
        "selection_metric": selection_metric,
        "selected_model_family": best_model_family,
        "selected_child_run_id": best_summary["child_run_id"],
        "selected_best_params": best_summary["best_params"],
        "feature_columns": feature_columns,
        "dropped_non_numeric_columns": dropped_non_numeric_columns,
        "model_summaries": model_summaries,
    }

    test_output = pd.DataFrame(
        {
            "actual_default": y_test.to_numpy(),
            "prediction": final_test_predictions,
            "probability_default": final_test_probabilities,
            "selected_model_family": best_model_family,
        }
    )

    if id_column in features_test.columns:
        test_output.insert(0, id_column, features_test[id_column].to_numpy())

    return (
        _to_builtin(metrics_output),
        _to_builtin(metadata),
        test_output,
        model_search_results,
    )