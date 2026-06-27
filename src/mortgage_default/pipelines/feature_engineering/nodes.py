"""Nodes for the feature_engineering pipeline."""
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

ID_COL = "loan_sequence_number"
TARGET_COL = "default"
COLS_TO_DROP = ["seller_name", "servicer_name", "msa"]
DATE_COLS = ["first_payment_date", "maturity_date"]


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop low-value columns while always keeping the ID and target."""
    to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    return df.drop(columns=to_drop)


def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn YYYYMM dates into numeric features and derive the loan term."""
    df = df.copy()
    for col in DATE_COLS:
        if col in df.columns:
            df[f"{col}_year"] = df[col].astype(int) // 100
            df[f"{col}_month"] = df[col].astype(int) % 100

    if {"first_payment_date", "maturity_date"}.issubset(df.columns):
        fp = df["first_payment_date"].astype(int)
        mt = df["maturity_date"].astype(int)
        df["loan_term_months"] = (mt // 100 - fp // 100) * 12 + (mt % 100 - fp % 100)

    return df.drop(columns=[c for c in DATE_COLS if c in df.columns])


def split_data(df: pd.DataFrame, parameters: dict) -> tuple:
    """Train/test split stratified by the target (imbalanced classes).

    The split happens BEFORE any fit so there is no leakage.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=df[TARGET_COL],
    )
    return train_df, test_df


def fit_feature_transformers(train_df: pd.DataFrame) -> dict:
    """Learn the transformations ONLY on the training set (imputation medians and categories).

    Returns a dictionary (artifact) reusable by the inference pipeline.
    """
    feature_cols = [c for c in train_df.columns if c not in (ID_COL, TARGET_COL)]
    numeric_cols = train_df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "impute_values": {c: float(train_df[c].median()) for c in numeric_cols},
        "categories": {
            c: sorted(train_df[c].dropna().unique().tolist()) for c in categorical_cols
        },
    }


def apply_feature_transformers(df: pd.DataFrame, transformers: dict) -> pd.DataFrame:
    """Apply imputation + one-hot using parameters learned on the training set.

    Guarantees the same columns across train, test and inference.
    """
    df = df.copy()

    # numeric imputation with the training medians
    for col, val in transformers["impute_values"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # one-hot encode the categoricals
    cat_cols = transformers["categorical_cols"]
    df = pd.get_dummies(df, columns=cat_cols)

    # force exactly the dummy columns seen during training
    expected = [
        f"{c}_{cat}" for c in cat_cols for cat in transformers["categories"][c]
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = 0
    prefixes = tuple(f"{c}_" for c in cat_cols)
    for col in list(df.columns):
        if col.startswith(prefixes) and col not in expected:
            df = df.drop(columns=col)

    return df


def log_feature_engineering(
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    transformers: dict,
    parameters: dict,
) -> dict:
    """Log the feature engineering parameters and metrics to MLflow.

    Creates a single run with the split configuration, the number of features
    produced and the default rate in train/test (a stratification sanity check).

    Returns:
        Dictionary with the logged metrics (also stored in the catalog).
    """
    feature_cols = [
        c for c in features_train.columns if c not in (ID_COL, TARGET_COL)
    ]
    metrics = {
        "n_train_rows": int(len(features_train)),
        "n_test_rows": int(len(features_test)),
        "n_features": int(len(feature_cols)),
        "train_default_rate": round(float(features_train[TARGET_COL].mean()), 4),
        "test_default_rate": round(float(features_test[TARGET_COL].mean()), 4),
    }

    mlflow.set_experiment("feature_engineering")
    with mlflow.start_run(run_name="feature_engineering"):
        mlflow.log_param("test_size", parameters["test_size"])
        mlflow.log_param("random_state", parameters["random_state"])
        mlflow.log_param("n_numeric_features", len(transformers["numeric_cols"]))
        mlflow.log_param(
            "n_categorical_features", len(transformers["categorical_cols"])
        )
        mlflow.log_param("dropped_columns", COLS_TO_DROP)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    return metrics
