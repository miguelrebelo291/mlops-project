import numpy as np
import pandas as pd
import pytest

from mortgage_default.pipelines.data_cleaning.nodes import clean_origination_data


@pytest.fixture
def model_input_dirty():
    """Synthetic model input with dirty values that should be cleaned."""
    return pd.DataFrame(
        {
            "credit_score": [9999, 650],
            "first_payment_date": [200503, 200505],
            "first_time_homebuyer_flag": ["9", "N"],
            "maturity_date": [203502, 203504],
            "msa": [np.nan, 17140.0],
            "mi_percentage": [999, 0],
            "number_of_units": [99, 1],
            "occupancy_status": ["P", "I"],
            "original_cltv": [999, 80],
            "original_dti": [999, 43],
            "original_upb": [100000, 200000],
            "original_ltv": [999, 77],
            "original_interest_rate": [5.5, 6.0],
            "channel": ["9", "R"],
            "prepayment_penalty_flag": ["N", "Y"],
            "amortization_type": ["FRM", "FRM"],
            "property_state": ["CA", "NY"],
            "property_type": ["99", "SF"],
            "postal_code": [94100, 10000],
            "loan_sequence_number": ["F05Q10000001", "F05Q10000002"],
            "loan_purpose": ["9", "P"],
            "original_loan_term": [360, 360],
            "number_of_borrowers": [99, 2],
            "seller_name": ["Other sellers", "Some seller"],
            "servicer_name": ["Other servicers", "Some servicer"],
            "super_conforming_flag": [np.nan, np.nan],
            "pre_relief_refinance_loan_sequence_number": [np.nan, np.nan],
            "special_eligibility_program": [9, 9],
            "relief_refinance_indicator": [np.nan, np.nan],
            "property_valuation_method": [7, 7],
            "interest_only_indicator": ["N", "N"],
            "mi_cancellation_indicator": [9, 9],
            "default": [0, 1],
        }
    )


@pytest.fixture
def model_input_with_leaky_column(model_input_dirty):
    """Synthetic input with a performance column that should be dropped."""
    df = model_input_dirty.copy()
    df["zero_balance_code"] = ["02", np.nan]
    return df


def test_cleaning_preserves_row_count(model_input_dirty):
    cleaned_df, stats = clean_origination_data(model_input_dirty)

    assert len(cleaned_df) == len(model_input_dirty)
    assert stats["rows_before"] == 2
    assert stats["rows_after"] == 2
    assert stats["rows_removed"] == 0


def test_cleaning_drops_expected_columns(model_input_dirty):
    cleaned_df, stats = clean_origination_data(model_input_dirty)

    expected_dropped_columns = {
        "super_conforming_flag",
        "pre_relief_refinance_loan_sequence_number",
        "relief_refinance_indicator",
        "property_valuation_method",
        "mi_cancellation_indicator",
        "special_eligibility_program",
    }

    for col in expected_dropped_columns:
        assert col not in cleaned_df.columns

    assert set(stats["dropped_columns"]) == expected_dropped_columns


def test_cleaning_drops_leaky_performance_columns(model_input_with_leaky_column):
    cleaned_df, stats = clean_origination_data(model_input_with_leaky_column)

    assert "zero_balance_code" not in cleaned_df.columns
    assert stats["leaky_performance_columns_dropped"] == ["zero_balance_code"]


def test_numeric_sentinels_become_missing(model_input_dirty):
    cleaned_df, stats = clean_origination_data(model_input_dirty)

    assert pd.isna(cleaned_df.loc[0, "credit_score"])
    assert pd.isna(cleaned_df.loc[0, "mi_percentage"])
    assert pd.isna(cleaned_df.loc[0, "number_of_units"])
    assert pd.isna(cleaned_df.loc[0, "original_cltv"])
    assert pd.isna(cleaned_df.loc[0, "original_dti"])
    assert pd.isna(cleaned_df.loc[0, "original_ltv"])
    assert pd.isna(cleaned_df.loc[0, "number_of_borrowers"])

    assert stats["credit_score_sentinel_count"] == 1
    assert stats["mi_percentage_sentinel_count"] == 1
    assert stats["number_of_units_sentinel_count"] == 1
    assert stats["original_cltv_sentinel_count"] == 1
    assert stats["original_dti_sentinel_count"] == 1
    assert stats["original_ltv_sentinel_count"] == 1
    assert stats["number_of_borrowers_sentinel_count"] == 1


def test_categorical_sentinels_become_unknown(model_input_dirty):
    cleaned_df, stats = clean_origination_data(model_input_dirty)

    assert cleaned_df.loc[0, "first_time_homebuyer_flag"] == "UNKNOWN"
    assert cleaned_df.loc[0, "channel"] == "UNKNOWN"
    assert cleaned_df.loc[0, "property_type"] == "UNKNOWN"
    assert cleaned_df.loc[0, "loan_purpose"] == "UNKNOWN"

    assert stats["first_time_homebuyer_flag_unknown_count"] == 1
    assert stats["channel_unknown_count"] == 1
    assert stats["property_type_unknown_count"] == 1
    assert stats["loan_purpose_unknown_count"] == 1


def test_dates_are_parsed(model_input_dirty):
    cleaned_df, stats = clean_origination_data(model_input_dirty)

    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["first_payment_date"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["maturity_date"])

    assert stats["first_payment_date_parse_failed_count"] == 0
    assert stats["maturity_date_parse_failed_count"] == 0


def test_date_consistency_is_reported(model_input_dirty):
    _, stats = clean_origination_data(model_input_dirty)

    assert stats["date_checks"]["available"] is True
    assert stats["date_checks"]["maturity_after_first_payment_all"] is True
    assert stats["date_checks"]["maturity_not_after_first_payment_count"] == 0
    assert stats["date_checks"]["term_difference_counts"] == {"0": 2}


def test_output_integer_like_columns_are_nullable_int(model_input_dirty):
    cleaned_df, _ = clean_origination_data(model_input_dirty)

    assert str(cleaned_df["msa"].dtype) == "Int64"
    assert str(cleaned_df["number_of_units"].dtype) == "Int64"
    assert str(cleaned_df["number_of_borrowers"].dtype) == "Int64"
    assert str(cleaned_df["original_loan_term"].dtype) == "Int64"
    assert str(cleaned_df["postal_code"].dtype) == "Int64"


def test_output_categorical_columns_are_strings(model_input_dirty):
    cleaned_df, _ = clean_origination_data(model_input_dirty)

    assert str(cleaned_df["first_time_homebuyer_flag"].dtype) == "string"
    assert str(cleaned_df["occupancy_status"].dtype) == "string"
    assert str(cleaned_df["channel"].dtype) == "string"
    assert str(cleaned_df["loan_sequence_number"].dtype) == "string"


def test_range_checks_are_valid(model_input_dirty):
    _, stats = clean_origination_data(model_input_dirty)

    assert stats["range_checks"]["credit_score_valid"] is True
    assert stats["range_checks"]["original_dti_valid"] is True
    assert stats["range_checks"]["original_ltv_valid"] is True
    assert stats["range_checks"]["original_cltv_valid"] is True
    assert stats["range_checks"]["mi_percentage_valid"] is True
    assert stats["range_checks"]["number_of_units_valid"] is True
    assert stats["range_checks"]["number_of_borrowers_valid"] is True
    assert stats["range_checks"]["default_valid"] is True


def test_categorical_checks_are_valid(model_input_dirty):
    _, stats = clean_origination_data(model_input_dirty)

    assert stats["categorical_check_results"]["first_time_homebuyer_flag"]["valid"] is True
    assert stats["categorical_check_results"]["property_type"]["valid"] is True
    assert stats["categorical_check_results"]["loan_purpose"]["valid"] is True


def test_id_and_target_statistics(model_input_dirty):
    cleaned_df, stats = clean_origination_data(model_input_dirty)

    assert stats["duplicate_loan_sequence_number_count"] == 0
    assert stats["loan_sequence_number_invalid_count"] == 0
    assert stats["property_state_invalid_count"] == 0

    assert cleaned_df["default"].tolist() == [0, 1]
    assert stats["target_distribution"] == {0: 1, 1: 1}