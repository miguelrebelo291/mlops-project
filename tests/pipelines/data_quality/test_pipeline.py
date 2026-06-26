"""
This is a boilerplate test file for pipeline 'data_quality'
generated using Kedro 1.3.1.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
"""
Unit tests for the data_quality pipeline nodes.
"""
import pandas as pd
import pytest
from mortgage_default.pipelines.data_quality.nodes import run_data_quality

VALID_COLUMNS = [
    "credit_score", "first_payment_date", "first_time_homebuyer_flag",
    "maturity_date", "msa", "mi_percentage", "number_of_units",
    "occupancy_status", "original_cltv", "original_dti", "original_upb",
    "original_ltv", "original_interest_rate", "channel", "prepayment_penalty_flag",
    "amortization_type", "property_state", "property_type", "postal_code",
    "loan_sequence_number", "loan_purpose", "original_loan_term",
    "number_of_borrowers", "seller_name", "servicer_name", "super_conforming_flag",
    "pre_relief_refinance_loan_sequence_number", "special_eligibility_program",
    "relief_refinance_indicator", "property_valuation_method",
    "interest_only_indicator", "mi_cancellation_indicator", "default",
]


@pytest.fixture
def valid_df():
    """Minimal valid DataFrame that passes all expectations."""
    return pd.DataFrame({
        "credit_score":           [700, 650],
        "first_payment_date":     [200501, 200502],
        "first_time_homebuyer_flag": ["Y", "N"],
        "maturity_date":          [203501, 203502],
        "msa":                    [35620.0, 31080.0],
        "mi_percentage":          [0, 25],
        "number_of_units":        [1, 1],
        "occupancy_status":       ["P", "P"],
        "original_cltv":          [80, 75],
        "original_dti":           [35, 40],
        "original_upb":           [200000, 150000],
        "original_ltv":           [80, 75],
        "original_interest_rate": [6.5, 5.75],
        "channel":                ["R", "B"],
        "prepayment_penalty_flag":["N", "N"],
        "amortization_type":      ["FRM", "FRM"],
        "property_state":         ["CA", "TX"],
        "property_type":          ["SF", "SF"],
        "postal_code":            [90200, 75200],
        "loan_sequence_number":   ["F05Q10000001", "F05Q10000002"],
        "loan_purpose":           ["P", "P"],
        "original_loan_term":     [360, 360],
        "number_of_borrowers":    [1, 2],
        "seller_name":            ["BANK A", "BANK B"],
        "servicer_name":          ["SERVICER A", "SERVICER B"],
        "super_conforming_flag":  [None, None],
        "pre_relief_refinance_loan_sequence_number": [None, None],
        "special_eligibility_program": [9, 9],
        "relief_refinance_indicator":  [None, None],
        "property_valuation_method":   [2, 2],
        "interest_only_indicator":     ["N", "N"],
        "mi_cancellation_indicator":   [7, 7],
        "default":                [0, 1],
    })


# Valid scenarios

def test_valid_data_passes(valid_df):
    result = run_data_quality(valid_df)
    assert result is not None


def test_valid_data_returns_same_shape(valid_df):
    result = run_data_quality(valid_df)
    assert result.shape == valid_df.shape


def test_valid_data_returns_same_columns(valid_df):
    result = run_data_quality(valid_df)
    assert list(result.columns) == list(valid_df.columns)


# Critical failure scenarios

def test_missing_default_column_raises(valid_df):
    df = valid_df.drop(columns=["default"])
    with pytest.raises(ValueError):
        run_data_quality(df)


def test_invalid_default_values_raises(valid_df):
    df = valid_df.copy()
    df["default"] = [2, 3]
    with pytest.raises(ValueError):
        run_data_quality(df)


def test_missing_loan_sequence_number_raises(valid_df):
    df = valid_df.drop(columns=["loan_sequence_number"])
    with pytest.raises(ValueError):
        run_data_quality(df)


def test_duplicate_loan_sequence_number_raises(valid_df):
    df = valid_df.copy()
    df["loan_sequence_number"] = ["F05Q10000001", "F05Q10000001"]
    with pytest.raises(ValueError):
        run_data_quality(df)


def test_wrong_column_count_raises(valid_df):
    df = valid_df.copy()
    df["extra_column"] = [0, 0]
    with pytest.raises(ValueError):
        run_data_quality(df)