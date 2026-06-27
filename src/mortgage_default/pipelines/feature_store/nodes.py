"""Nodes for the feature_store pipeline (Hopsworks Feature Store)."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

ID_COL = "loan_sequence_number"


def _sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Hopsworks only accepts feature names in lowercase with underscores.

    One-hot encoding produces columns like 'occupancy_status_O' (with uppercase),
    so we normalise everything before inserting into the feature group.
    """
    df = df.copy()
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns
    ]
    return df


def upload_to_feature_store(features: pd.DataFrame, parameters: dict) -> dict:
    """Insert the features into a Hopsworks Feature Group.

    The API key is read from the HOPSWORKS_API_KEY environment variable (never in code).

    Args:
        features: feature table (must include loan_sequence_number).
        parameters: feature group config (parameters_feature_store.yml).

    Returns:
        Dictionary with the feature group name/version and number of inserted rows.
    """
    import hopsworks  # lazy import: the package is only needed when uploading

    features = _sanitize_feature_names(features)
    if ID_COL not in features.columns:
        raise ValueError(f"'{ID_COL}' must exist to be the primary key.")

    project = hopsworks.login(
        project=parameters.get("project") or None,
    )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=parameters["feature_group_name"],
        version=parameters["feature_group_version"],
        description=parameters.get(
            "description", "Mortgage default origination features"
        ),
        primary_key=[ID_COL],
        online_enabled=parameters.get("online_enabled", False),
        time_travel_format=parameters.get("time_travel_format", "NONE"),
    )

    fg.insert(features, write_options={"wait_for_job": True})

    result = {
        "feature_group": parameters["feature_group_name"],
        "version": parameters["feature_group_version"],
        "n_rows": int(len(features)),
        "n_features": int(features.shape[1]),
    }
    logger.info("Inserted into Hopsworks Feature Store: %s", result)
    return result
