"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.3.1
"""
import pandas as pd

ORIGINATION_COLUMNS = [
    "credit_score",
    "first_payment_date",
    "first_time_homebuyer_flag",
    "maturity_date",
    "msa",
    "mi_percentage",
    "number_of_units",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "prepayment_penalty_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_sequence_number",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "servicer_name",
    "super_conforming_flag",
    "pre_relief_refinance_loan_sequence_number",
    "special_eligibility_program",
    "relief_refinance_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "mi_cancellation_indicator",
]

PERFORMANCE_COLUMNS = [
    "loan_sequence_number",
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

DEFAULT_CODES = ["02", "03", "09"]

def assign_origination_columns_names(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Assign column names to the origination data."""
    raw_data.columns = ORIGINATION_COLUMNS
    return raw_data


def assign_performance_columns_names(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Assign column names to the performance data.

    Supports both:
    - full 32-column performance files
    - lightweight 2-column files loaded with usecols [0, 8]
    """
    if raw_data.shape[1] == 2:
        raw_data.columns = [
            "loan_sequence_number",
            "zero_balance_code",
        ]
        return raw_data

    raw_data.columns = PERFORMANCE_COLUMNS

    if "current_loan_delinquency_status" in raw_data.columns:
        raw_data["current_loan_delinquency_status"] = (
            raw_data["current_loan_delinquency_status"].astype("string")
        )

    return raw_data


def create_target_variable(performance_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance data to loan level and create binary default target.

    A loan is considered defaulted if zero_balance_code is ever 02, 03, or 09.
    """

    zero_balance_code_clean = (
        pd.to_numeric(performance_data["zero_balance_code"], errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.zfill(2)
    )

    target_source = pd.DataFrame(
        {
            "loan_sequence_number": performance_data["loan_sequence_number"],
            "zero_balance_code_clean": zero_balance_code_clean,
        }
    )

    target = (
        target_source.groupby("loan_sequence_number")["zero_balance_code_clean"]
        .apply(lambda codes: int(codes.isin(DEFAULT_CODES).any()))
        .reset_index()
        .rename(columns={"zero_balance_code_clean": "default"})
    )

    return target


def join_origination_with_target(
    origination_data: pd.DataFrame,
    target_data: pd.DataFrame,
) -> pd.DataFrame:
    """Join origination data with known loan-level target labels."""
    joined = origination_data.merge(
        target_data,
        on="loan_sequence_number",
        how="inner",
    )

    joined["default"] = joined["default"].astype(int)

    return joined


def concatenate_years(*dfs: pd.DataFrame, years: list[int] = None) -> pd.DataFrame:
    """Concatenate datasets from multiple years, adding year column."""
    if years is None:
        from mortgage_default.pipelines.data_ingestion.pipeline import YEARS

        years = YEARS

    labeled = []

    for df, year in zip(dfs, years):
        df["year"] = year
        labeled.append(df)

    return pd.concat(labeled, ignore_index=True)