import json
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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
        non_numeric_columns = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        df = df.drop(columns=non_numeric_columns)
        feature_columns = df.columns.tolist()
    else:
        non_numeric_columns = []
        df = df.reindex(columns=feature_columns, fill_value=0)

    return df, y, feature_columns, non_numeric_columns


def _predict_probabilities(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
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


def _choose_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    strategy: str,
    fallback_threshold: float,
    beta: float = 2.0,
) -> tuple[float, dict[str, float | None]]:
    if strategy == "fixed":
        threshold = float(fallback_threshold)
        predictions = (probabilities >= threshold).astype(int)

        metrics = _classification_metrics(
            y_true=y_true,
            probabilities=probabilities,
            predictions=predictions,
            prefix="threshold_selection",
            beta=beta,
        )

        metrics["threshold_selection_beta"] = float(beta)

        return threshold, metrics

    if strategy not in {"maximize_f1", "maximize_f2", "maximize_fbeta"}:
        raise ValueError(f"Unknown threshold_strategy: {strategy}")

    if strategy == "maximize_f1":
        beta = 1.0
    elif strategy == "maximize_f2":
        beta = 2.0

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)

    if len(thresholds) == 0:
        threshold = float(fallback_threshold)
    else:
        beta_squared = beta**2

        fbeta_scores = (
            (1 + beta_squared)
            * precision
            * recall
            / ((beta_squared * precision) + recall + 1e-12)
        )

        best_idx = int(np.nanargmax(fbeta_scores[:-1]))
        threshold = float(thresholds[best_idx])

    predictions = (probabilities >= threshold).astype(int)

    metrics = _classification_metrics(
        y_true=y_true,
        probabilities=probabilities,
        predictions=predictions,
        prefix="threshold_selection",
        beta=beta,
    )

    metrics["threshold_selection_beta"] = float(beta)

    return threshold, metrics


def _build_model_spaces(
    random_state: int,
    n_jobs: int,
) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    logistic_regression = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    decision_tree = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                DecisionTreeClassifier(
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
                    n_jobs=n_jobs,
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
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )

    gradient_boosting = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=random_state,
                ),
            ),
        ]
    )

    hist_gradient_boosting = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": (
            logistic_regression,
            {
                "classifier__C": np.logspace(-3, 2, 30).tolist(),
                "classifier__solver": ["lbfgs", "liblinear"],
                "classifier__penalty": ["l2"],
                "classifier__class_weight": [None, "balanced"],
                "classifier__max_iter": [1000, 2000, 4000],
            },
        ),
        "decision_tree": (
            decision_tree,
            {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [None, 3, 5, 10, 20, 40],
                "classifier__min_samples_split": [2, 5, 10, 25, 50],
                "classifier__min_samples_leaf": [1, 2, 5, 10, 25, 50],
                "classifier__max_features": [None, "sqrt", "log2", 0.5],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            random_forest,
            {
                "classifier__n_estimators": [200, 400, 600, 800],
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [None, 5, 10, 20, 40],
                "classifier__min_samples_split": [2, 5, 10, 25],
                "classifier__min_samples_leaf": [1, 2, 5, 10, 25],
                "classifier__max_features": ["sqrt", "log2", None, 0.5],
                "classifier__bootstrap": [True, False],
                "classifier__class_weight": [None, "balanced", "balanced_subsample"],
            },
        ),
        "extra_trees": (
            extra_trees,
            {
                "classifier__n_estimators": [200, 400, 600, 800],
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [None, 5, 10, 20, 40],
                "classifier__min_samples_split": [2, 5, 10, 25],
                "classifier__min_samples_leaf": [1, 2, 5, 10, 25],
                "classifier__max_features": ["sqrt", "log2", None, 0.5],
                "classifier__bootstrap": [False, True],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "gradient_boosting": (
            gradient_boosting,
            {
                "classifier__n_estimators": [100, 200, 400, 600],
                "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "classifier__max_depth": [2, 3, 4, 5],
                "classifier__min_samples_split": [2, 5, 10, 25],
                "classifier__min_samples_leaf": [1, 2, 5, 10, 25, 50],
                "classifier__subsample": [0.6, 0.8, 1.0],
                "classifier__max_features": [None, "sqrt", "log2"],
            },
        ),
        "hist_gradient_boosting": (
            hist_gradient_boosting,
            {
                "classifier__max_iter": [100, 200, 400, 600],
                "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "classifier__max_leaf_nodes": [15, 31, 63, 127],
                "classifier__max_depth": [None, 3, 5, 10],
                "classifier__min_samples_leaf": [10, 20, 50, 100],
                "classifier__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
        ),
    }


def _candidate_results_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    results = pd.DataFrame(rows)

    if results.empty:
        return results

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
    kwargs = {
        "sk_model": model,
        "signature": signature,
        "input_example": input_example,
        "serialization_format": mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    }

    try:
        mlflow.sklearn.log_model(name=artifact_name, **kwargs)
    except TypeError:
        mlflow.sklearn.log_model(artifact_path=artifact_name, **kwargs)


def _save_sklearn_model(
    model: Pipeline,
    path: Path,
    signature: Any,
    input_example: pd.DataFrame,
) -> None:
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
    """
    Holdout random search for model families and export the best model.

    Compatibility:
    - Same Kedro inputs as before: features_train, features_test, params:model_train.
    - Same Kedro outputs as before:
      model_training_metrics,
      model_training_metadata,
      model_test_predictions,
      model_search_results.

    Anti-leakage design:
    - features_train/features_test already come from feature_engineering.
    - features_train is split again into train + validation.
    - Random candidates are fit on train only.
    - Hyperparameters are selected on validation only.
    - Threshold is selected on validation only.
    - features_test is used once at the very end for final reporting.
    """

    target_column = parameters.get("target_column", "default")
    id_column = parameters.get("id_column", "loan_sequence_number")

    validation_size = float(parameters.get("validation_size", 0.2))
    random_state = int(parameters.get("random_state", 42))

    fallback_threshold = float(parameters.get("threshold", 0.5))
    threshold_strategy = parameters.get("threshold_strategy", "maximize_f2")
    beta = float(parameters.get("beta", 2.0))

    experiment_name = parameters.get("experiment_name", "mortgage_default_model_search")
    base_run_name = parameters.get("run_name", "holdout_random_search_tree_models")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = f"{timestamp}_{uuid4().hex[:8]}"

    if parameters.get("add_timestamp_to_run_name", True):
        run_name = f"{base_run_name}_{run_suffix}"
    else:
        run_name = base_run_name

    model_output_path = Path(
        parameters.get("model_output_path", "data/06_models/mlflow_model")
    )

    n_iter = int(parameters.get("n_iter", 20))
    n_jobs = int(parameters.get("n_jobs", -1))

    scoring = parameters.get("scoring", "average_precision")
    selection_metric = parameters.get("selection_metric", "val_average_precision")

    requested_models = parameters.get(
        "models",
        [
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "hist_gradient_boosting",
            "decision_tree",
            "logistic_regression",
        ],
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

    if y_train_full.nunique() < 2:
        raise ValueError("features_train must contain both classes 0 and 1.")

    if y_test.nunique() < 2:
        raise ValueError("features_test must contain both classes 0 and 1.")

    stratify = y_train_full if y_train_full.nunique() == 2 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify,
    )

    model_spaces = _build_model_spaces(
        random_state=random_state,
        n_jobs=n_jobs,
    )

    unknown_models = set(requested_models) - set(model_spaces)
    if unknown_models:
        raise ValueError(f"Unknown model names in parameters: {sorted(unknown_models)}")

    if selection_metric not in {
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_fbeta",
        "val_average_precision",
        "val_roc_auc",
    }:
        raise ValueError(
            "selection_metric must be one of: "
            "val_accuracy, val_precision, val_recall, val_f1, val_fbeta, "
            "val_average_precision, val_roc_auc"
        )

    tracking_uri = parameters.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment name: {experiment_name}")
    print(f"MLflow parent run name: {run_name}")
    print(f"Search strategy: holdout random search, no CV")
    print(f"Selection metric: {selection_metric}")
    print(f"Threshold strategy: {threshold_strategy}")
    print(f"F-beta beta: {beta}")

    candidate_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []

    with mlflow.start_run(run_name=run_name) as parent_run:
        parent_run_id = parent_run.info.run_id

        print(f"MLflow parent run ID: {parent_run_id}")
        print(f"MLflow artifact URI: {parent_run.info.artifact_uri}")

        mlflow.log_params(
            {
                "validation_size": validation_size,
                "fallback_threshold": fallback_threshold,
                "threshold_strategy": threshold_strategy,
                "beta": beta,
                "random_state": random_state,
                "n_iter": n_iter,
                "n_jobs": n_jobs,
                "scoring": scoring,
                "selection_metric": selection_metric,
                "search_strategy": "holdout_random_search_no_cv",
                "n_features": len(feature_columns),
                "n_train_rows": len(X_train),
                "n_val_rows": len(X_val),
                "n_test_rows": len(X_test),
                "models": ",".join(requested_models),
                "run_timestamp": timestamp,
            }
        )

        for model_family in requested_models:
            base_model, param_distributions = model_spaces[model_family]

            sampled_params = list(
                ParameterSampler(
                    param_distributions,
                    n_iter=n_iter,
                    random_state=random_state,
                )
            )

            child_run_name = f"{model_family}_holdout_search_{run_suffix}"

            with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                child_run_id = child_run.info.run_id

                print(f"MLflow child run ID for {model_family}: {child_run_id}")
                print(f"MLflow child run name for {model_family}: {child_run_name}")

                best_family_summary: dict[str, Any] | None = None
                best_family_model: Pipeline | None = None

                for candidate_idx, params in enumerate(sampled_params):
                    candidate_model = clone(base_model)
                    candidate_model.set_params(**params)

                    candidate_model.fit(X_train, y_train)

                    train_probabilities = _predict_probabilities(
                        candidate_model,
                        X_train,
                    )

                    train_threshold, _ = _choose_threshold(
                        y_true=y_train,
                        probabilities=train_probabilities,
                        strategy=threshold_strategy,
                        fallback_threshold=fallback_threshold,
                        beta=beta,
                    )

                    train_predictions = (
                        train_probabilities >= train_threshold
                    ).astype(int)

                    val_probabilities = _predict_probabilities(
                        candidate_model,
                        X_val,
                    )

                    selected_threshold, threshold_metrics = _choose_threshold(
                        y_true=y_val,
                        probabilities=val_probabilities,
                        strategy=threshold_strategy,
                        fallback_threshold=fallback_threshold,
                        beta=beta,
                    )

                    val_predictions = (
                        val_probabilities >= selected_threshold
                    ).astype(int)

                    train_metrics = _classification_metrics(
                        y_true=y_train,
                        probabilities=train_probabilities,
                        predictions=train_predictions,
                        prefix="train",
                        beta=beta,
                    )

                    val_metrics = _classification_metrics(
                        y_true=y_val,
                        probabilities=val_probabilities,
                        predictions=val_predictions,
                        prefix="val",
                        beta=beta,
                    )

                    candidate_summary = {
                        "model_family": model_family,
                        "candidate_idx": candidate_idx,
                        "child_run_id": child_run_id,
                        "child_run_name": child_run_name,
                        "params": params,
                        "params_json": json.dumps(params, default=str, sort_keys=True),
                        "selected_threshold": selected_threshold,
                        "train_threshold": train_threshold,
                        **train_metrics,
                        **val_metrics,
                        **threshold_metrics,
                    }

                    candidate_rows.append(
                        {
                            "model_family": model_family,
                            "candidate_idx": candidate_idx,
                            "child_run_id": child_run_id,
                            "parent_run_id": parent_run_id,
                            "child_run_name": child_run_name,
                            "params_json": candidate_summary["params_json"],
                            "selected_threshold": selected_threshold,
                            "train_threshold": train_threshold,
                            "is_best_candidate": False,
                            **train_metrics,
                            **val_metrics,
                        }
                    )

                    current_value = candidate_summary.get(selection_metric)

                    if current_value is None:
                        continue

                    best_value = (
                        best_family_summary.get(selection_metric)
                        if best_family_summary is not None
                        else None
                    )

                    if best_value is None or float(current_value) > float(best_value):
                        best_family_summary = candidate_summary
                        best_family_model = candidate_model

                if best_family_summary is None or best_family_model is None:
                    raise ValueError(f"No valid candidate found for {model_family}.")

                mlflow.log_params(
                    {
                        "model_family": model_family,
                        "child_run_name": child_run_name,
                        "best_params": json.dumps(
                            best_family_summary["params"],
                            default=str,
                        ),
                        "best_candidate_idx": best_family_summary["candidate_idx"],
                        "best_selected_threshold": best_family_summary[
                            "selected_threshold"
                        ],
                    }
                )

                mlflow.log_metrics(
                    {
                        key: value
                        for key, value in best_family_summary.items()
                        if isinstance(value, (int, float, np.integer, np.floating))
                        and value is not None
                        and not isinstance(value, bool)
                    }
                )

                input_example = X_train.head(5)
                signature = infer_signature(
                    input_example,
                    best_family_model.predict(input_example),
                )

                _log_sklearn_model(
                    model=best_family_model,
                    signature=signature,
                    input_example=input_example,
                    artifact_name="model",
                )

                model_summaries.append(best_family_summary)

        model_search_results = _candidate_results_to_frame(candidate_rows)

        if not model_search_results.empty:
            for summary in model_summaries:
                mask = (
                    (model_search_results["model_family"] == summary["model_family"])
                    & (
                        model_search_results["candidate_idx"]
                        == summary["candidate_idx"]
                    )
                )
                model_search_results.loc[mask, "is_best_candidate"] = True

        with TemporaryDirectory() as tmp_dir:
            search_results_path = Path(tmp_dir) / "model_search_results.csv"
            model_search_results.to_csv(search_results_path, index=False)
            mlflow.log_artifact(str(search_results_path), artifact_path="search_results")

        def _selection_value(summary: dict[str, Any]) -> float:
            value = summary.get(selection_metric)
            if value is None:
                return float("-inf")
            return float(value)

        best_summary = max(model_summaries, key=_selection_value)
        best_model_family = best_summary["model_family"]
        selected_threshold = float(best_summary["selected_threshold"])

        selected_base_model, _ = model_spaces[best_model_family]
        selected_base_model = clone(selected_base_model)
        selected_base_model.set_params(**best_summary["params"])

        X_train_plus_val = pd.concat([X_train, X_val], axis=0)
        y_train_plus_val = pd.concat([y_train, y_val], axis=0)

        selected_base_model.fit(X_train_plus_val, y_train_plus_val)

        final_test_probabilities = _predict_probabilities(selected_base_model, X_test)
        final_test_predictions = (
            final_test_probabilities >= selected_threshold
        ).astype(int)

        final_test_metrics = _classification_metrics(
            y_true=y_test,
            probabilities=final_test_probabilities,
            predictions=final_test_predictions,
            prefix="final_test",
            beta=beta,
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
                "selected_child_run_name": best_summary["child_run_name"],
                "selected_best_params": json.dumps(
                    best_summary["params"],
                    default=str,
                ),
                "selected_threshold": selected_threshold,
                "threshold_strategy": threshold_strategy,
                "beta": beta,
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
            "run_name": run_name,
            "run_timestamp": timestamp,
            "selected_model_family": best_model_family,
            "selected_child_run_id": best_summary["child_run_id"],
            "selected_child_run_name": best_summary["child_run_name"],
            "selection_metric": selection_metric,
            "selection_metric_value": best_summary.get(selection_metric),
            "selected_threshold": selected_threshold,
            "threshold_strategy": threshold_strategy,
            "beta": beta,
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
            "run_timestamp": timestamp,
            "model_output_path": str(model_output_path),
            "target_column": target_column,
            "id_column": id_column,
            "threshold": selected_threshold,
            "fallback_threshold": fallback_threshold,
            "threshold_strategy": threshold_strategy,
            "beta": beta,
            "random_state": random_state,
            "n_iter": n_iter,
            "scoring": scoring,
            "selection_metric": selection_metric,
            "search_strategy": "holdout_random_search_no_cv",
            "selected_model_family": best_model_family,
            "selected_child_run_id": best_summary["child_run_id"],
            "selected_child_run_name": best_summary["child_run_name"],
            "selected_best_params": best_summary["params"],
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