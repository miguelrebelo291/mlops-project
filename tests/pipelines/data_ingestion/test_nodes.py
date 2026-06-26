import pandas as pd
import pytest
from mortgage_default.pipelines.data_ingestion.nodes import  assign_origination_columns_names

def test_assign_origination_columns_names():
    "Test that column names are correctly assigned to the origination data."

    # Create a sample raw DataFrame without column names
    df = pd.DataFrame([[0] *32])

    # run function
    result = assign_origination_columns_names(df)

    # check if 1st column name is correct 
    assert result.columns[0] == "credit_score"

    #check last column name
    assert result.columns[-1] == "mi_cancellation_indicator"

    # check total number of cols
    assert len(result.columns) == 32
