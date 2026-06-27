"""Pipeline 'feature_engineering_inference'.

Reuses the training pipeline nodes, but ONLY the transformation part:
- drop_unused_columns and engineer_date_features (same as training)
- apply_feature_transformers loading 'feature_transformers' from disk

There is no fit or split here: the parameters (medians, categories, columns) come
from training, which guarantees exactly the same features and zero leakage.
"""
from kedro.pipeline import Node, Pipeline

from ..feature_engineering.nodes import (
    apply_feature_transformers,
    drop_unused_columns,
    engineer_date_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=drop_unused_columns,
                inputs="origination_data_inference",
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
