"""Testes unitários para os nodes do pipeline feature_store."""
import pandas as pd
import pytest

from mortgage_default.pipelines.feature_store.nodes import (
    _sanitize_feature_names,
    upload_to_feature_store,
)


def test_sanitize_feature_names_lowercases():
    df = pd.DataFrame(columns=["loan_sequence_number", "occupancy_status_O", "channel_R"])
    out = _sanitize_feature_names(df)
    assert list(out.columns) == [
        "loan_sequence_number",
        "occupancy_status_o",
        "channel_r",
    ]


def test_upload_requires_id_column():
    df = pd.DataFrame({"occupancy_status_o": [1, 0]})  # sem loan_sequence_number
    params = {"feature_group_name": "fg", "feature_group_version": 1}
    with pytest.raises(ValueError, match="loan_sequence_number"):
        upload_to_feature_store(df, params)
