"""Feature engineering for labeled inference.

Uses the transformers fitted during training.
The default target is allowed to pass through so evaluation can happen later,
but model_inference will drop it before prediction.
"""

from kedro.pipeline import Node, Pipeline

from mortgage_default.pipelines.feature_engineering.nodes import (
    apply_feature_transformers,
    drop_unused_columns,
    engineer_date_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=drop_unused_columns,
                inputs="model_input_data_inference_cleaned",
                outputs="inference_no_unused",
                name="fe_inf_drop_unused_columns_node",
            ),
            Node(
                func=engineer_date_features,
                inputs="inference_no_unused",
                outputs="inference_dated",
                name="fe_inf_engineer_date_features_node",
            ),
            Node(
                func=apply_feature_transformers,
                inputs=["inference_dated", "feature_transformers"],
                outputs="features_inference",
                name="fe_inf_apply_transformers_node",
            ),
        ]
    )