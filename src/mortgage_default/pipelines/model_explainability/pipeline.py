"""Pipeline 'model_explainability'.

Generates SHAP feature importance for the model produced by model_train.
Depends on features_test and train_metadata, both persisted by
model_train, so this pipeline can be re-run independently without
repeating the random search or evaluation.
"""
from kedro.pipeline import Pipeline, node

from .nodes import generate_shap_report


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=generate_shap_report,
                inputs=[
                    "features_test",
                    "train_metadata",
                    "params:model_explainability",
                ],
                outputs="shap_metrics",
                name="generate_shap_report_node",
            )
        ]
    )
