"""Nodes for the model_train pipeline.

Split into two clear stages:
- train_search: temporal data prep + holdout random search across model
  families, selecting the best one. Does NOT touch features_test.
- evaluate_model: applies the selected, already-trained model to
  features_test and computes final metrics. Can be re-run independently
  (e.g. for testing, or to re-evaluate against a different test slice)
  without repeating the expensive random search.

This separation also means evaluate_model's transformation logic is the
exact code path model_predict will reuse in production.
"""
import json
import logging
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

logger = logging.getLogger(__name__)

TARGET_COL = "default"
ID_COL = "loan_sequence_number"
COLS_TO_DROP = ["seller_name", "servicer_name", "postal_code", "msa"]
DATE_COLS = ["first_payment_date", "maturity_date"]
CAT_COLS = [
    "first_time_homebuyer_flag",
    "occupancy_status",
    "channel",
    "prepayment_penalty_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "loan_purpose",
    "interest_only_indicator",
]


# ──────────────────────────────────────────────────────────────────────────
# Data preparation: stateless steps (reusable identically in model_predict)
# ──────────────────────────────────────────────────────────────────────────

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID and low-value columns. Stateless — safe to reuse in inference."""
    cols_to_drop = [c for c in [ID_COL] + COLS_TO_DROP if c in df.columns]
    return df.drop(columns=cols_to_drop)


def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai features numéricas das colunas de data. Stateless."""
    df = df.copy()

    for col in DATE_COLS:
        if col not in df.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_year"] = df[col].astype("Int64") // 100
            df[f"{col}_month"] = df[col].astype("Int64") % 100

    if {"first_payment_date", "maturity_date"}.issubset(df.columns):
        fp = df["first_payment_date"]
        mt = df["maturity_date"]
        if pd.api.types.is_datetime64_any_dtype(fp):
            df["loan_term_months"] = (
                (mt.dt.year - fp.dt.year) * 12 + (mt.dt.month - fp.dt.month)
            )
        else:
            fp_int = pd.to_numeric(fp, errors="coerce")
            mt_int = pd.to_numeric(mt, errors="coerce")
            df["loan_term_months"] = (
                (mt_int // 100 - fp_int // 100) * 12
                + (mt_int % 100 - fp_int % 100)
            )

    cols_to_drop = [c for c in DATE_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def split_temporal(df: pd.DataFrame, parameters: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal — treino nos anos <= train_cutoff_year,
    teste nos anos >= test_start_year.
    """
    train_cutoff = parameters["train_cutoff_year"]
    test_start = parameters["test_start_year"]

    train_df = df[df["year"] <= train_cutoff].drop(columns=["year"])
    test_df = df[df["year"] >= test_start].drop(columns=["year"])

    logger.info(
        "Split temporal: treino até %d (%d linhas), teste a partir de %d (%d linhas)",
        train_cutoff, len(train_df), test_start, len(test_df),
    )
    logger.info(
        "Taxa de default — treino: %.2f%%, teste: %.2f%%",
        train_df[TARGET_COL].mean() * 100,
        test_df[TARGET_COL].mean() * 100,
    )

    return train_df, test_df


def fit_feature_transformers(train_df: pd.DataFrame) -> dict:
    """Aprende as transformações SÓ no treino. Artefacto reutilizável."""
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    numeric_cols = train_df[feature_cols].select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in CAT_COLS if c in train_df.columns]

    impute_values = {c: float(train_df[c].median()) for c in numeric_cols}

    label_mappings = {}
    for col in cat_cols:
        values = train_df[col].astype(str).fillna("unknown")
        classes = sorted(values.unique().tolist())
        if "unknown" not in classes:
            classes.append("unknown")
        label_mappings[col] = {cls: idx for idx, cls in enumerate(classes)}

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "impute_values": impute_values,
        "label_mappings": label_mappings,
    }


def apply_feature_transformers(df: pd.DataFrame, transformers: dict) -> pd.DataFrame:
    """Aplica imputação + label encoding. Reutilizado por model_predict."""
    df = df.copy()

    for col, median in transformers["impute_values"].items():
        if col in df.columns:
            df[col] = df[col].fillna(median)

    for col, mapping in transformers["label_mappings"].items():
        if col not in df.columns:
            continue
        values = df[col].astype(str).fillna("unknown")
        values = values.apply(lambda x: x if x in mapping else "unknown")
        df[col] = values.map(mapping)

    return df


# ──────────────────────────────────────────────────────────────────────────
# Shared helpers (used by both train_search and evaluate_model)
# ──────────────────────────────────────────────────────────────────────────

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
            y_true=y_true, probabilities=probabilities,
            predictions=predictions, prefix="threshold_selection", beta=beta,
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
            (1 + beta_squared) * precision * recall
            / ((beta_squared * precision) + recall + 1e-12)
        )
        best_idx = int(np.nanargmax(fbeta_scores[:-1]))
        threshold = float(thresholds[best_idx])

    predictions = (probabilities >= threshold).astype(int)
    metrics = _classification_metrics(
        y_true=y_true, probabilities=probabilities,
        predictions=predictions, prefix="threshold_selection", beta=beta,
    )
    metrics["threshold_selection_beta"] = float(beta)

    return threshold, metrics


def _build_model_spaces(
    random_state: int,
    n_jobs: int,
) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    logistic_regression = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
    ])

    decision_tree = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(random_state=random_state)),
    ])

    random_forest = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)),
    ])

    extra_trees = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", ExtraTreesClassifier(random_state=random_state, n_jobs=n_jobs)),
    ])

    gradient_boosting = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", GradientBoostingClassifier(random_state=random_state)),
    ])

    hist_gradient_boosting = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", HistGradientBoostingClassifier(random_state=random_state)),
    ])

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


# ──────────────────────────────────────────────────────────────────────────
# Stage 1: train_search — random search + model selection. Never touches
# features_test. Returns the fitted model and everything needed to
# evaluate it later (independently, without retraining).
# ──────────────────────────────────────────────────────────────────────────

def train_search(
    features_train: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """Holdout random search across model families, selecting the best one.

    Anti-leakage design:
    - Random candidates are fit on an inner train split only.
    - Hyperparameters and threshold are selected on an inner validation
      split only.
    - features_test is never seen here — evaluation happens separately
      in evaluate_model, so this stage can be re-run, cached, or tested
      without needing the test set at all.

    Args:
        features_train: transformed training data (temporal split, years
            <= train_cutoff_year), target column included.
        parameters: model_train parameter group.
    Returns:
        selected_model: fitted sklearn Pipeline (refit on train+val).
        train_metadata: dict with model family, params, threshold,
            feature_columns, MLflow run ids, etc.
        model_search_results: DataFrame with every candidate tried.
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

    n_iter = int(parameters.get("n_iter", 20))
    n_jobs = int(parameters.get("n_jobs", -1))

    selection_metric = parameters.get("selection_metric", "val_average_precision")

    requested_models = parameters.get(
        "models",
        [
            "random_forest", "extra_trees", "gradient_boosting",
            "hist_gradient_boosting", "decision_tree", "logistic_regression",
        ],
    )

    X_train_full, y_train_full, feature_columns, dropped_non_numeric_columns = (
        _prepare_xy(features_train, target_column=target_column, id_column=id_column)
    )

    if y_train_full is None:
        raise ValueError(f"Target column '{target_column}' not found in features_train.")
    if y_train_full.nunique() < 2:
        raise ValueError("features_train must contain both classes 0 and 1.")

    stratify = y_train_full if y_train_full.nunique() == 2 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=validation_size,
        random_state=random_state, stratify=stratify,
    )

    model_spaces = _build_model_spaces(random_state=random_state, n_jobs=n_jobs)

    unknown_models = set(requested_models) - set(model_spaces)
    if unknown_models:
        raise ValueError(f"Unknown model names in parameters: {sorted(unknown_models)}")

    valid_selection_metrics = {
        "val_accuracy", "val_precision", "val_recall", "val_f1", "val_fbeta",
        "val_average_precision", "val_roc_auc",
    }
    if selection_metric not in valid_selection_metrics:
        raise ValueError(f"selection_metric must be one of: {sorted(valid_selection_metrics)}")

    tracking_uri = parameters.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    logger.info("Search strategy: holdout random search, no CV")
    logger.info("Selection metric: %s | Threshold strategy: %s", selection_metric, threshold_strategy)

    candidate_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []

    with mlflow.start_run(run_name=run_name) as parent_run:
        parent_run_id = parent_run.info.run_id

        mlflow.log_params({
            "validation_size": validation_size,
            "fallback_threshold": fallback_threshold,
            "threshold_strategy": threshold_strategy,
            "beta": beta,
            "random_state": random_state,
            "n_iter": n_iter,
            "n_jobs": n_jobs,
            "selection_metric": selection_metric,
            "search_strategy": "holdout_random_search_no_cv",
            "n_features": len(feature_columns),
            "n_train_rows": len(X_train),
            "n_val_rows": len(X_val),
            "models": ",".join(requested_models),
            "run_timestamp": timestamp,
            "train_cutoff_year": parameters.get("train_cutoff_year"),
            "test_start_year": parameters.get("test_start_year"),
        })

        for model_family in requested_models:
            base_model, param_distributions = model_spaces[model_family]
            sampled_params = list(ParameterSampler(
                param_distributions, n_iter=n_iter, random_state=random_state,
            ))
            child_run_name = f"{model_family}_holdout_search_{run_suffix}"

            with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                child_run_id = child_run.info.run_id
                logger.info("A treinar %s (%d candidatos)...", model_family, len(sampled_params))

                best_family_summary: dict[str, Any] | None = None
                best_family_model: Pipeline | None = None

                for candidate_idx, params in enumerate(sampled_params):
                    candidate_model = clone(base_model)
                    candidate_model.set_params(**params)
                    candidate_model.fit(X_train, y_train)

                    train_probabilities = _predict_probabilities(candidate_model, X_train)
                    train_threshold, _ = _choose_threshold(
                        y_true=y_train, probabilities=train_probabilities,
                        strategy=threshold_strategy, fallback_threshold=fallback_threshold, beta=beta,
                    )
                    train_predictions = (train_probabilities >= train_threshold).astype(int)

                    val_probabilities = _predict_probabilities(candidate_model, X_val)
                    selected_threshold, threshold_metrics = _choose_threshold(
                        y_true=y_val, probabilities=val_probabilities,
                        strategy=threshold_strategy, fallback_threshold=fallback_threshold, beta=beta,
                    )
                    val_predictions = (val_probabilities >= selected_threshold).astype(int)

                    train_metrics = _classification_metrics(
                        y_true=y_train, probabilities=train_probabilities,
                        predictions=train_predictions, prefix="train", beta=beta,
                    )
                    val_metrics = _classification_metrics(
                        y_true=y_val, probabilities=val_probabilities,
                        predictions=val_predictions, prefix="val", beta=beta,
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
                        **train_metrics, **val_metrics, **threshold_metrics,
                    }

                    candidate_rows.append({
                        "model_family": model_family,
                        "candidate_idx": candidate_idx,
                        "child_run_id": child_run_id,
                        "parent_run_id": parent_run_id,
                        "child_run_name": child_run_name,
                        "params_json": candidate_summary["params_json"],
                        "selected_threshold": selected_threshold,
                        "train_threshold": train_threshold,
                        "is_best_candidate": False,
                        **train_metrics, **val_metrics,
                    })

                    current_value = candidate_summary.get(selection_metric)
                    if current_value is None:
                        continue

                    best_value = (
                        best_family_summary.get(selection_metric)
                        if best_family_summary is not None else None
                    )
                    if best_value is None or float(current_value) > float(best_value):
                        best_family_summary = candidate_summary
                        best_family_model = candidate_model

                if best_family_summary is None or best_family_model is None:
                    raise ValueError(f"No valid candidate found for {model_family}.")

                mlflow.log_params({
                    "model_family": model_family,
                    "child_run_name": child_run_name,
                    "best_params": json.dumps(best_family_summary["params"], default=str),
                    "best_candidate_idx": best_family_summary["candidate_idx"],
                    "best_selected_threshold": best_family_summary["selected_threshold"],
                })
                mlflow.log_metrics({
                    key: value for key, value in best_family_summary.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                    and value is not None and not isinstance(value, bool)
                })

                input_example = X_train.head(5)
                signature = infer_signature(input_example, best_family_model.predict(input_example))
                _log_sklearn_model(
                    model=best_family_model, signature=signature,
                    input_example=input_example, artifact_name="model",
                )

                model_summaries.append(best_family_summary)
                logger.info(
                    "%s — melhor %s: %.4f", model_family, selection_metric,
                    best_family_summary.get(selection_metric, float("nan")),
                )

        model_search_results = _candidate_results_to_frame(candidate_rows)

        if not model_search_results.empty:
            for summary in model_summaries:
                mask = (
                    (model_search_results["model_family"] == summary["model_family"])
                    & (model_search_results["candidate_idx"] == summary["candidate_idx"])
                )
                model_search_results.loc[mask, "is_best_candidate"] = True

        with TemporaryDirectory() as tmp_dir:
            search_results_path = Path(tmp_dir) / "model_search_results.csv"
            model_search_results.to_csv(search_results_path, index=False)
            mlflow.log_artifact(str(search_results_path), artifact_path="search_results")

        def _selection_value(summary: dict[str, Any]) -> float:
            value = summary.get(selection_metric)
            return float("-inf") if value is None else float(value)

        best_summary = max(model_summaries, key=_selection_value)
        best_model_family = best_summary["model_family"]
        selected_threshold = float(best_summary["selected_threshold"])

        logger.info(
            "Melhor modelo: %s (%s: %.4f)", best_model_family, selection_metric,
            best_summary.get(selection_metric, float("nan")),
        )

        selected_base_model, _ = model_spaces[best_model_family]
        selected_base_model = clone(selected_base_model)
        selected_base_model.set_params(**best_summary["params"])

        X_train_plus_val = pd.concat([X_train, X_val], axis=0)
        y_train_plus_val = pd.concat([y_train, y_val], axis=0)
        selected_base_model.fit(X_train_plus_val, y_train_plus_val)

        mlflow.log_params({
            "selected_model_family": best_model_family,
            "selected_child_run_id": best_summary["child_run_id"],
            "selected_child_run_name": best_summary["child_run_name"],
            "selected_best_params": json.dumps(best_summary["params"], default=str),
            "selected_threshold": selected_threshold,
            "threshold_strategy": threshold_strategy,
            "beta": beta,
        })

        input_example = X_train_plus_val.head(5)
        signature = infer_signature(input_example, selected_base_model.predict(input_example))

        _log_sklearn_model(
            model=selected_base_model, signature=signature,
            input_example=input_example, artifact_name="selected_model",
        )

    train_metadata = {
        "parent_run_id": parent_run_id,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "run_timestamp": timestamp,
        "model_uri": f"runs:/{best_summary['child_run_id']}/model",
        "target_column": target_column,
        "id_column": id_column,
        "threshold": selected_threshold,
        "fallback_threshold": fallback_threshold,
        "threshold_strategy": threshold_strategy,
        "beta": beta,
        "random_state": random_state,
        "n_iter": n_iter,
        "selection_metric": selection_metric,
        "search_strategy": "holdout_random_search_no_cv",
        "selected_model_family": best_model_family,
        "selected_child_run_id": best_summary["child_run_id"],
        "selected_child_run_name": best_summary["child_run_name"],
        "selected_best_params": best_summary["params"],
        "feature_columns": feature_columns,
        "dropped_non_numeric_columns": dropped_non_numeric_columns,
        "model_summaries": model_summaries,
        "train_cutoff_year": parameters.get("train_cutoff_year"),
        "test_start_year": parameters.get("test_start_year"),
    }

    return selected_base_model, _to_builtin(train_metadata), model_search_results

