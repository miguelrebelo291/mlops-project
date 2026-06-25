"""Testes do pipeline feature_engineering_inference.

Garante que a inferência reutiliza os transformers do treino e produz
exatamente as mesmas features (mesmas colunas, sem fit nem leakage).
"""
import numpy as np
import pandas as pd
import pytest

from mortgage_default.pipelines.feature_engineering.nodes import (
    apply_feature_transformers,
    drop_unused_columns,
    engineer_date_features,
    fit_feature_transformers,
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


def _prepare(df):
    return engineer_date_features(drop_unused_columns(df))


def test_inference_matches_training_columns(raw_df):
    # transformers aprendidos no "treino"
    train = _prepare(raw_df)
    transformers = fit_feature_transformers(train)
    features_train = apply_feature_transformers(train, transformers)

    # dados de inferência diferentes (outros valores), mas mesmo schema
    infer_raw = raw_df.copy()
    infer_raw["credit_score"] = [800, 640, 700, 710, np.nan, 690, 720, 650, 730, 660]
    features_inf = apply_feature_transformers(_prepare(infer_raw), transformers)

    # mesmas colunas e mesma ordem que o treino
    assert list(features_inf.columns) == list(features_train.columns)
    # sem missing (imputação com as medianas do treino)
    assert features_inf.isna().sum().sum() == 0


def test_inference_without_target(raw_df):
    # na inferência real pode não existir a coluna 'default' (é o que se prevê)
    train = _prepare(raw_df)
    transformers = fit_feature_transformers(train)

    infer_raw = raw_df.drop(columns=["default"])
    features_inf = apply_feature_transformers(_prepare(infer_raw), transformers)

    # o ID mantém-se e não rebenta sem o target
    assert "loan_sequence_number" in features_inf.columns
    assert features_inf.isna().sum().sum() == 0
