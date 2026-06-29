import pandas as pd

from mortgage_default.pipelines.data_ingestion.nodes import ORIGINATION_COLUMNS


def assign_inference_origination_columns(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Assign Freddie Mac origination column names for unlabeled inference data."""
    df = raw_data.copy()

    # If already named correctly, keep as-is.
    if list(df.columns) == ORIGINATION_COLUMNS:
        return df

    if df.shape[1] != len(ORIGINATION_COLUMNS):
        raise ValueError(
            f"Inference origination data has {df.shape[1]} columns, "
            f"but expected {len(ORIGINATION_COLUMNS)}."
        )

    df.columns = ORIGINATION_COLUMNS
    return df