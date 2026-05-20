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

def assign_origination_columns_names (raw_data: pd.DataFrame) -> pd.DataFrame:
    """Assign column names to the origination data.
    
    Args:
        raw_data: raw origination dataset without column names.

    Returns:
        DataFrame with assigned column names.
    
    """

    raw_data.columns = ORIGINATION_COLUMNS
    return raw_data   