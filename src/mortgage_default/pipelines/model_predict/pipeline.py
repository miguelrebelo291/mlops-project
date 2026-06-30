"""Pipeline 'model_predict'.

Genuine production inference simulation: a new loan application has only
origination data — no performance history (it hasn't happened yet), no
target label. This pipeline scores such loans using the model trained
by model_train, ending at the prediction step.

The evaluate_labeled_inference node is intentionally NOT wired into this
default pipeline, since real new loans have no known outcome to compare
against. If you want to sanity-check the model against historical loans
with known labels, feed in data that does have a 'default' column and
call evaluate_labeled_inference directly/separately.
"""
from kedro.pipeline import Pipeline, node

from .nodes import (
    clean_inference_data,
    ingest_inference_data,
    prepare_inference_features,
    run_model_inference,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=ingest_inference_data,
                inputs="raw_origination_data_inference",
                outputs="origination_data_inference_named",
                name="ingest_inference_data_node",
            ),
            node(
                func=clean_inference_data,
                inputs="origination_data_inference_named",
                outputs=[
                    "origination_data_inference_cleaned",
                    "inference_cleaning_statistics",
                ],
                name="clean_inference_data_node",
            ),
            node(
                func=prepare_inference_features,
                inputs=[
                    "origination_data_inference_cleaned",
                    "feature_transformers",
                ],
                outputs="features_inference",
                name="prepare_inference_features_node",
            ),
            node(
                func=run_model_inference,
                inputs=[
                    "features_inference",
                    "model_training_metadata",
                    "params:model_predict",
                ],
                outputs="model_inference_predictions",
                name="run_model_inference_node",
            ),
        ]
    )
