"""Testes unitários para os nodes do pipeline feature_engineering."""
import mlflow
import numpy as np
import pandas as pd
import pytest

from mortgage_default.pipelines.feature_engineering.nodes import (
    apply_feature_transformers,
    drop_unused_columns,
    engineer_date_features,
    fit_feature_transformers,
    log_feature_engineering,
    split_data,
)


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "loan_sequence_number": [f"F{i:03d}" for i in range(10)],
            "credit_score": [700, 680, np.nan, 720, 710, 690, 730, 660, 705, np.nan],
            "first_payment_date": [200503] * 10,
            "maturity_date": [203502] * 10,
            "occupancy_status": ["O", "I", "O", "S", "O", "I", "O", "S", "O", "I"],
            "seller_name": ["BANK_A"] * 10,
            "servicer_name": ["SERV_A"] * 10,
            "msa": [0] * 10,
            "default": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        }
    )


def test_drop_unused_columns(raw_df):
    out = drop_unused_columns(raw_df)
    for col in ["seller_name", "servicer_name", "msa"]:
        assert col not in out.columns
    # ID e target nunca são removidos
    assert "loan_sequence_number" in out.columns
    assert "default" in out.columns


def test_engineer_date_features(raw_df):
    out = engineer_date_features(raw_df)
    assert "first_payment_date" not in out.columns
    assert "maturity_date" not in out.columns
    assert out["first_payment_date_year"].iloc[0] == 2005
    assert out["first_payment_date_month"].iloc[0] == 3
    # 200503 -> 203502 = 30 anos = 359 meses
    assert out["loan_term_months"].iloc[0] == 359


def test_split_data_sizes(raw_df):
    params = {"test_size": 0.2, "random_state": 42}
    train_df, test_df = split_data(raw_df, params)
    assert len(train_df) == 8
    assert len(test_df) == 2
    # sem sobreposição de empréstimos entre treino e teste
    assert set(train_df["loan_sequence_number"]).isdisjoint(
        test_df["loan_sequence_number"]
    )


def test_fit_only_uses_train(raw_df):
    # imputação aprendida só no treino
    transformers = fit_feature_transformers(raw_df.drop(columns=["msa"]))
    assert "credit_score" in transformers["impute_values"]
    assert "loan_sequence_number" not in transformers["numeric_cols"]
    assert "default" not in transformers["numeric_cols"]


def test_apply_no_missing_and_consistent_columns(raw_df):
    df = engineer_date_features(drop_unused_columns(raw_df))
    transformers = fit_feature_transformers(df)
    out = apply_feature_transformers(df, transformers)
    # imputação resolveu os NaN
    assert out.isna().sum().sum() == 0
    # ID e target preservados
    assert "loan_sequence_number" in out.columns
    assert "default" in out.columns
    # one-hot criou colunas das categorias do treino
    assert "occupancy_status_O" in out.columns


def test_apply_handles_unseen_category(raw_df):
    df = engineer_date_features(drop_unused_columns(raw_df))
    transformers = fit_feature_transformers(df)
    # introduz uma categoria nunca vista no "treino"
    df_infer = df.copy()
    df_infer.loc[0, "occupancy_status"] = "Z"
    out = apply_feature_transformers(df_infer, transformers)
    # colunas iguais às do treino, sem coluna nova para "Z"
    train_out = apply_feature_transformers(df, transformers)
    assert list(out.columns) == list(train_out.columns)
    assert "occupancy_status_Z" not in out.columns


def test_log_feature_engineering(raw_df, tmp_path):
    # tracking isolado para o teste não tocar no MLflow real
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    df = engineer_date_features(drop_unused_columns(raw_df))
    transformers = fit_feature_transformers(df)
    features = apply_feature_transformers(df, transformers)
    params = {"test_size": 0.2, "random_state": 42}

    metrics = log_feature_engineering(features, features, transformers, params)

    assert metrics["n_train_rows"] == len(features)
    assert metrics["n_features"] == len(
        [c for c in features.columns if c not in ("loan_sequence_number", "default")]
    )
    assert 0.0 <= metrics["train_default_rate"] <= 1.0
