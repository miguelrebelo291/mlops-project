"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 1.3.1
"""
from typing import Any

import pandas as pd


NUMERIC_SENTINEL_MAP = {
    "credit_score": [9999],
    "mi_percentage": [999],
    "number_of_units": [99],
    "original_cltv": [999],
    "original_dti": [999],
    "original_ltv": [999],
    "number_of_borrowers": [99],
}


CATEGORICAL_UNKNOWN_MAP = {
    "first_time_homebuyer_flag": ["9"],
    "occupancy_status": ["9"],
    "channel": ["9"],
    "property_type": ["99"],
    "loan_purpose": ["9"],
}


CATEGORICAL_COLUMNS = [
    "first_time_homebuyer_flag",
    "occupancy_status",
    "channel",
    "prepayment_penalty_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "loan_sequence_number",
    "loan_purpose",
    "seller_name",
    "servicer_name",
    "interest_only_indicator",
]


NUMERIC_COLUMNS = [
    "credit_score",
    "msa",
    "mi_percentage",
    "number_of_units",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "original_loan_term",
    "number_of_borrowers",
]


COLUMNS_TO_DROP = [
    "super_conforming_flag",
    "pre_relief_refinance_loan_sequence_number",
    "relief_refinance_indicator",
    "property_valuation_method",
    "mi_cancellation_indicator",
    "special_eligibility_program",
]


LEAKY_PERFORMANCE_COLUMNS = [
    "monthly_reporting_period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "due_date_last_paid_installment",
    "mi_recoveries",
    "net_sale_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "maintenance_preservation_costs",
    "taxes_insurance",
    "miscellaneous_expenses",
    "actual_loss_calculation",
    "cumulative_modification_cost",
    "interest_rate_step_indicator",
    "payment_deferral_flag",
    "estimated_ltv",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_status_code",
    "current_month_modification_cost",
    "interest_bearing_upb",
]


ALLOWED_CATEGORY_VALUES = {
    "first_time_homebuyer_flag": {"Y", "N", "UNKNOWN"},
    "occupancy_status": {"P", "I", "S", "UNKNOWN"},
    "channel": {"R", "B", "C", "T", "UNKNOWN"},
    "prepayment_penalty_flag": {"Y", "N"},
    "amortization_type": {"FRM", "ARM"},
    "property_type": {"CO", "PU", "MH", "SF", "CP", "UNKNOWN"},
    "loan_purpose": {"P", "C", "N", "R", "UNKNOWN"},
    "interest_only_indicator": {"Y", "N"},
}


def _count_missing(df: pd.DataFrame) -> dict[str, int]:
    return df.isna().sum().astype(int).to_dict()


def _safe_date_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value.date())


def _get_range_checks(df: pd.DataFrame) -> dict[str, bool]:
    checks: dict[str, bool] = {}

    if "credit_score" in df.columns:
        checks["credit_score_valid"] = bool(
            df["credit_score"].dropna().between(300, 850).all()
        )

    if "original_dti" in df.columns:
        checks["original_dti_valid"] = bool(
            df["original_dti"].dropna().between(1, 65).all()
        )

    if "original_ltv" in df.columns:
        checks["original_ltv_valid"] = bool(
            df["original_ltv"].dropna().between(6, 105).all()
        )

    if "original_cltv" in df.columns:
        checks["original_cltv_valid"] = bool(
            df["original_cltv"].dropna().between(6, 200).all()
        )

    if "mi_percentage" in df.columns:
        checks["mi_percentage_valid"] = bool(
            df["mi_percentage"].dropna().between(0, 55).all()
        )

    if "number_of_units" in df.columns:
        checks["number_of_units_valid"] = bool(
            df["number_of_units"].dropna().isin([1, 2, 3, 4]).all()
        )

    if "number_of_borrowers" in df.columns:
        checks["number_of_borrowers_valid"] = bool(
            df["number_of_borrowers"].dropna().isin([1, 2]).all()
        )

    if "default" in df.columns:
        checks["default_valid"] = bool(df["default"].dropna().isin([0, 1]).all())

    return checks


def _get_categorical_checks(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    for col, allowed_values in ALLOWED_CATEGORY_VALUES.items():
        if col not in df.columns:
            continue

        observed = set(df[col].dropna().astype(str).str.strip().str.upper())
        unexpected = sorted(observed - allowed_values)

        results[col] = {
            "observed_values": sorted(observed),
            "unexpected_values": unexpected,
            "valid": len(unexpected) == 0,
        }

    return results


def _get_date_checks(df: pd.DataFrame) -> dict[str, Any]:
    required_cols = {"first_payment_date", "maturity_date", "original_loan_term"}

    if not required_cols.issubset(df.columns):
        return {
            "available": False,
            "reason": "Missing one or more required date columns.",
        }

    months_between = (
        (df["maturity_date"].dt.year - df["first_payment_date"].dt.year) * 12
        + (df["maturity_date"].dt.month - df["first_payment_date"].dt.month)
        + 1
    )

    term_difference = months_between - df["original_loan_term"]

    term_difference_counts = {
        str(int(key)): int(value)
        for key, value in term_difference.value_counts().sort_index().items()
        if pd.notna(key)
    }

    maturity_after_first_payment = df["maturity_date"] > df["first_payment_date"]

    return {
        "available": True,
        "first_payment_date_min": _safe_date_string(df["first_payment_date"].min()),
        "first_payment_date_max": _safe_date_string(df["first_payment_date"].max()),
        "maturity_date_min": _safe_date_string(df["maturity_date"].min()),
        "maturity_date_max": _safe_date_string(df["maturity_date"].max()),
        "maturity_after_first_payment_all": bool(maturity_after_first_payment.all()),
        "maturity_not_after_first_payment_count": int(
            (~maturity_after_first_payment).sum()
        ),
        "term_difference_counts": term_difference_counts,
        "note": (
            "Dates are parsed and checked for internal consistency only. "
            "Rows are not removed or corrected based on expected file year."
        ),
    }


def clean_origination_data(
    model_input: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the validated model input dataset.

    This function is safe for both training and inference:
    - during training, the input may include the target column `default`;
    - during inference, `default` may be absent;
    - performance columns are dropped if they accidentally appear, to reduce leakage risk.

    Args:
        model_input: Validated model input data.

    Returns:
        A tuple containing:
            1. cleaned DataFrame
            2. cleaning statistics dictionary
    """
    df = model_input.copy()

    stats: dict[str, Any] = {
        "rows_before": int(len(df)),
        "columns_before": int(df.shape[1]),
    }

    # Normalize column names.
    df.columns = df.columns.str.strip().str.lower()

    # Drop accidental performance/leakage columns if present.
    leaky_cols_present = [col for col in LEAKY_PERFORMANCE_COLUMNS if col in df.columns]
    stats["leaky_performance_columns_dropped"] = leaky_cols_present

    if leaky_cols_present:
        df = df.drop(columns=leaky_cols_present)

    # Convert numeric columns.
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean numeric sentinel values.
    for col, sentinels in NUMERIC_SENTINEL_MAP.items():
        if col not in df.columns:
            continue

        mask = df[col].isin(sentinels)
        stats[f"{col}_sentinel_count"] = int(mask.sum())

        df.loc[mask, col] = pd.NA
    
    integer_like_columns = [
    "number_of_units",
    "number_of_borrowers",
    "original_loan_term",
    "postal_code"]

    for col in integer_like_columns:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    # Parse date columns.
    for col in ["first_payment_date", "maturity_date"]:
        if col not in df.columns:
            continue

        df[col] = pd.to_datetime(
            df[col].astype("string").str.strip(),
            format="%Y%m",
            errors="coerce",
        )

        stats[f"{col}_parse_failed_count"] = int(df[col].isna().sum())

    # Normalize categoricals.
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.upper()

    # Clean categorical sentinel values.
    for col, sentinels in CATEGORICAL_UNKNOWN_MAP.items():
        if col not in df.columns:
            continue

        sentinels_as_string = {str(value) for value in sentinels}
        mask = df[col].isin(sentinels_as_string)

        stats[f"{col}_unknown_count"] = int(mask.sum())

        df.loc[mask, col] = "UNKNOWN"

    # MSA is geographic and can be missing. Keep missing values for later imputation.
    if "msa" in df.columns:
        stats["msa_missing_count"] = int(df["msa"].isna().sum())
        df["msa"] = df["msa"].astype("Int64")

    # Target exists in training but not in inference.
    if "default" in df.columns:
        df["default"] = pd.to_numeric(df["default"], errors="coerce")
        stats["default_missing_count"] = int(df["default"].isna().sum())

        if df["default"].isna().any():
            raise ValueError("Missing target values found. Do not fill labels with 0.")

        if not df["default"].isin([0, 1]).all():
            raise ValueError("Target column must contain only 0/1 values.")

        df["default"] = df["default"].astype(int)

    # Drop columns that are not useful/safe for this baseline cleaned modeling dataset.
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    stats["dropped_columns"] = cols_to_drop

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Do not drop duplicate loans automatically, but report them.
    if "loan_sequence_number" in df.columns:
        stats["duplicate_loan_sequence_number_count"] = int(
            df["loan_sequence_number"].duplicated().sum()
        )

        loan_id_pattern = r"^[FA]\d{2}Q[1-4]\d{7}$"
        loan_id_valid = df["loan_sequence_number"].astype(str).str.match(
            loan_id_pattern
        )
        stats["loan_sequence_number_invalid_count"] = int((~loan_id_valid).sum())

    # Basic property state format check.
    if "property_state" in df.columns:
        property_state_valid = df["property_state"].astype(str).str.match(r"^[A-Z]{2}$")
        stats["property_state_invalid_count"] = int((~property_state_valid).sum())

    # Store validation summaries.
    stats["range_checks"] = _get_range_checks(df)
    stats["categorical_check_results"] = _get_categorical_checks(df)
    stats["date_checks"] = _get_date_checks(df)

    stats["missing_values_after_cleaning"] = _count_missing(df)

    if "default" in df.columns:
        stats["target_distribution"] = (
            df["default"].value_counts(dropna=False).astype(int).to_dict()
        )

    stats["rows_after"] = int(len(df))
    stats["columns_after"] = int(df.shape[1])
    stats["rows_removed"] = int(stats["rows_before"] - stats["rows_after"])

    return df, stats