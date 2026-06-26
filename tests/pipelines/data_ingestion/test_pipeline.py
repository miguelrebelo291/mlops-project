"""
This is a boilerplate test file for pipeline 'data_ingestion'
generated using Kedro 1.3.1.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pandas as pd
import pytest
from mortgage_default.pipelines.data_ingestion.nodes import (
    assign_origination_columns_names,
    assign_performance_columns_names,
    create_target_variable,
    join_origination_with_target,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_origination():
    """Synthetic raw origination DataFrame with 32 unnamed columns."""
    return pd.DataFrame([[None] * 32])


@pytest.fixture
def raw_performance():
    """Synthetic raw performance DataFrame with 32 unnamed columns."""
    return pd.DataFrame([[None] * 32])

@pytest.fixture
def named_performance_default():
    """Performance data where loan A defaulted (code 09) and loan B did not (code 01)."""
    return pd.DataFrame({
        "loan_sequence_number": ["A", "A", "B", "B"],
        "zero_balance_code":    [None, 9.0, None, 1.0],
    }).astype({"zero_balance_code": "Float64"})


@pytest.fixture
def named_performance_no_code():
    """Performance data where loan C never got a zero_balance_code (still active)."""
    return pd.DataFrame({
        "loan_sequence_number": ["C", "C"],
        "zero_balance_code": [None, None],
    })


@pytest.fixture
def origination_named():
    """Origination data with loan_sequence_number column."""
    return pd.DataFrame({
        "loan_sequence_number": ["A", "B", "C"],
        "credit_score": [700, 650, 720],
    })


@pytest.fixture
def target_data():
    """Target data — loan A defaulted, loan B did not."""
    return pd.DataFrame({
        "loan_sequence_number": ["A", "B"],
        "default": [1, 0],
    })


# ── assign_origination_columns_names ────────────────────────────────────────

def test_origination_has_32_columns(raw_origination):
    result = assign_origination_columns_names(raw_origination)
    assert result.shape[1] == 32


def test_origination_first_column_is_credit_score(raw_origination):
    result = assign_origination_columns_names(raw_origination)
    assert result.columns[0] == "credit_score"


def test_origination_last_column_is_mi_cancellation(raw_origination):
    result = assign_origination_columns_names(raw_origination)
    assert result.columns[-1] == "mi_cancellation_indicator"


def test_origination_contains_loan_sequence_number(raw_origination):
    result = assign_origination_columns_names(raw_origination)
    assert "loan_sequence_number" in result.columns


# ── assign_performance_columns_names ────────────────────────────────────────

def test_performance_has_32_columns(raw_performance):
    result = assign_performance_columns_names(raw_performance)
    assert result.shape[1] == 32


def test_performance_first_column_is_loan_sequence_number(raw_performance):
    result = assign_performance_columns_names(raw_performance)
    assert result.columns[0] == "loan_sequence_number"


def test_performance_delinquency_status_is_string(raw_performance):
    result = assign_performance_columns_names(raw_performance)
    assert pd.api.types.is_string_dtype(result["current_loan_delinquency_status"])


def test_performance_contains_zero_balance_code(raw_performance):
    result = assign_performance_columns_names(raw_performance)
    assert "zero_balance_code" in result.columns


# ── create_target_variable ───────────────────────────────────────────────────

def test_target_one_row_per_loan(named_performance_default):
    result = create_target_variable(named_performance_default)
    assert len(result) == 2


def test_target_columns(named_performance_default):
    result = create_target_variable(named_performance_default)
    assert set(result.columns) == {"loan_sequence_number", "default"}


def test_target_default_only_zero_or_one(named_performance_default):
    result = create_target_variable(named_performance_default)
    assert result["default"].isin([0, 1]).all()


def test_target_loan_with_code_09_is_default(named_performance_default):
    result = create_target_variable(named_performance_default)
    assert result.loc[result["loan_sequence_number"] == "A", "default"].values[0] == 1


def test_target_loan_with_code_01_is_not_default(named_performance_default):
    result = create_target_variable(named_performance_default)
    assert result.loc[result["loan_sequence_number"] == "B", "default"].values[0] == 0


def test_target_loan_with_no_code_is_not_default(named_performance_no_code):
    result = create_target_variable(named_performance_no_code)
    assert result.loc[result["loan_sequence_number"] == "C", "default"].values[0] == 0


# ── join_origination_with_target ─────────────────────────────────────────────

def test_join_has_default_column(origination_named, target_data):
    result = join_origination_with_target(origination_named, target_data)
    assert "default" in result.columns


def test_join_preserves_all_origination_rows(origination_named, target_data):
    result = join_origination_with_target(origination_named, target_data)
    assert len(result) == len(origination_named)


def test_join_loan_without_target_gets_zero(origination_named, target_data):
    result = join_origination_with_target(origination_named, target_data)
    assert result.loc[result["loan_sequence_number"] == "C", "default"].values[0] == 0


def test_join_default_is_integer(origination_named, target_data):
    result = join_origination_with_target(origination_named, target_data)
    assert result["default"].dtype == int