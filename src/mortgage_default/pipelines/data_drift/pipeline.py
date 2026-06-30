"""Pipeline 'data_drift'.

Detects feature drift year by year (2003-2008) comparing against the
training distribution (2000-2002) using Evidently's DataDriftPreset.

Uses origination_data_cleaned (Standard + NSD) as the source — not
features_inference — so the comparison includes the full loan population,
including ARM/interest-only products from the NSD that are most relevant
for detecting pre-crisis drift.

Date-derived features are excluded to avoid trivial drift from time passing.

Run independently after model_train:
    kedro run --pipeline data_drift
"""
from kedro.pipeline import Pipeline, node

from .nodes import analyze_data_drift


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=analyze_data_drift,
                inputs=[
                    "origination_data_cleaned",
                    "features_train",
                    "feature_transformers",
                    "train_metadata",
                    "params:data_drift",
                ],
                outputs=[
                    "data_drift_report",
                    "data_drift_metrics",
                ],
                name="analyze_data_drift_node",
            )
        ]
    )
