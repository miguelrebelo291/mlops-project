"""Nodes for the model_train pipeline."""
import logging

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

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


def _engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai features numéricas das colunas de data.

    O cleaning converte as datas para datetime — aqui extraímos
    ano, mês e calculamos o prazo do empréstimo em meses.
    """
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


def prepare_features(
    df: pd.DataFrame, parameters: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepara features para treino: drop, engenharia de datas, split temporal,
    imputação e encoding.

    O split é temporal — treino nos anos <= train_cutoff_year,
    teste nos anos >= test_start_year. Isto evita data leakage e
    simula o cenário real de prever defaults futuros.

    Args:
        df: dataset limpo com todos os anos e coluna 'year'.
        parameters: parâmetros do pipeline (train_cutoff_year, test_start_year).
    Returns:
        X_train, X_test, y_train, y_test
    """
    df = df.copy()

    cols_to_drop = [c for c in [ID_COL] + COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df = _engineer_date_features(df)

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

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    cat_cols = [c for c in CAT_COLS if c in X_train.columns]
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str).fillna("unknown")
        X_test[col] = X_test[col].astype(str).fillna("unknown")
        le = LabelEncoder()
        le.fit(X_train[col])
        X_test[col] = X_test[col].apply(
            lambda x: x if x in le.classes_ else "unknown"
        )
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    return X_train, X_test, y_train, y_test


def _get_best_threshold(
    model, X_val: pd.DataFrame, y_val: pd.Series, recall_floor: float = 0.70
) -> float:
    """Escolhe o threshold que maximiza precisão mantendo recall >= recall_floor."""
    proba = model.predict_proba(X_val)[:, 1]
    prec, rec, thresh = precision_recall_curve(y_val, proba)
    valid = rec[:-1] >= recall_floor
    if not valid.any():
        return float(thresh[0])
    return float(thresh[valid].max())


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    parameters: dict,
) -> dict:
    """Treina Random Forest e XGBoost, regista no MLflow e devolve os modelos."""
    mlflow.set_experiment(parameters["mlflow_experiment_name"])

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    scale_pos_weight = int((y_tr == 0).sum() / (y_tr == 1).sum())

    models_config = {
        "random_forest": RandomForestClassifier(
            n_estimators=parameters.get("rf_n_estimators", 200),
            max_depth=parameters.get("rf_max_depth", 10),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=parameters.get("xgb_n_estimators", 200),
            max_depth=parameters.get("xgb_max_depth", 6),
            learning_rate=parameters.get("xgb_learning_rate", 0.1),
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }

    trained_models = {}

    for name, model in models_config.items():
        logger.info("A treinar %s...", name)

        with mlflow.start_run(run_name=name):
            model.fit(X_tr, y_tr)

            threshold = _get_best_threshold(model, X_val, y_val, recall_floor=0.70)
            mlflow.log_param("decision_threshold", threshold)

            proba = model.predict_proba(X_test)[:, 1]
            preds = (proba >= threshold).astype(int)

            metrics = {
                "test_auc_roc": roc_auc_score(y_test, proba),
                "test_recall_default": recall_score(y_test, preds),
                "test_precision_default": precision_score(y_test, preds, zero_division=0),
                "test_f1_default": f1_score(y_test, preds, zero_division=0),
            }
            mlflow.log_metrics(metrics)
            mlflow.log_params({
                "model_type": name,
                "train_cutoff_year": parameters["train_cutoff_year"],
                "test_start_year": parameters["test_start_year"],
            })
            mlflow.sklearn.log_model(model, artifact_path=name)

            logger.info(
                "%s — AUC: %.4f | Recall: %.4f | Precision: %.4f | F1: %.4f",
                name,
                metrics["test_auc_roc"],
                metrics["test_recall_default"],
                metrics["test_precision_default"],
                metrics["test_f1_default"],
            )

            trained_models[name] = {
                "model": model,
                "threshold": threshold,
                "metrics": metrics,
            }

    return trained_models


def compute_shap_values(
    trained_models: dict,
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
) -> dict:
    """Calcula SHAP values para explicabilidade dos modelos."""
    shap_results = {}

    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    for name, model_data in trained_models.items():
        model = model_data["model"]
        logger.info("A calcular SHAP values para %s...", name)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_sample)

            if hasattr(shap_values, "values"):
                vals = shap_values.values
                if vals.ndim == 3:
                    vals = vals[:, :, 1]
            else:
                vals = shap_values

            mean_abs_shap = np.abs(vals).mean(axis=0)
            feature_importance = dict(zip(X_sample.columns, mean_abs_shap.tolist()))
            feature_importance_sorted = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            shap_results[name] = feature_importance_sorted

            logger.info(
                "%s — Top 5 features: %s",
                name,
                list(feature_importance_sorted.keys())[:5],
            )

        except Exception as e:
            logger.warning("SHAP falhou para %s: %s", name, str(e))
            shap_results[name] = {}

    return shap_results


def select_best_model(trained_models: dict, shap_results: dict) -> dict:
    """Seleciona o melhor modelo com base no AUC-ROC."""
    best_name = max(
        trained_models,
        key=lambda name: trained_models[name]["metrics"]["test_auc_roc"],
    )
    best = trained_models[best_name]

    logger.info(
        "Melhor modelo: %s (AUC: %.4f)",
        best_name,
        best["metrics"]["test_auc_roc"],
    )

    return {
        "best_model_name": best_name,
        "best_model": best["model"],
        "threshold": best["threshold"],
        "metrics": best["metrics"],
        "shap_feature_importance": shap_results.get(best_name, {}),
        "all_metrics": {
            name: data["metrics"] for name, data in trained_models.items()
        },
    }
