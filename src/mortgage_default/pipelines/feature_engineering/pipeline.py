"""Pipeline 'feature_engineering'."""
from kedro.pipeline import Node, Pipeline

from .nodes import (
    apply_feature_transformers,
    drop_unused_columns,
    engineer_date_features,
    fit_feature_transformers,
    log_feature_engineering,
    split_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=drop_unused_columns,
                inputs="origination_data_cleaned",
                outputs="features_no_unused",
                name="drop_unused_columns_node",
            ),
            Node(
                func=engineer_date_features,
                inputs="features_no_unused",
                outputs="features_dated",
                name="engineer_date_features_node",
            ),
            Node(
                func=split_data,
                inputs=["features_dated", "params:feature_engineering"],
                outputs=["train_raw", "test_raw"],
                name="fe_split_data_node",
            ),
            Node(
                func=fit_feature_transformers,
                inputs="train_raw",
                outputs="feature_transformers",
                name="fit_transformers_node",
            ),
            Node(
                func=apply_feature_transformers,
                inputs=["train_raw", "feature_transformers"],
                outputs="features_train",
                name="apply_transformers_train_node",
            ),
            Node(
                func=apply_feature_transformers,
                inputs=["test_raw", "feature_transformers"],
                outputs="features_test",
                name="apply_transformers_test_node",
            ),
            Node(
                func=log_feature_engineering,
                inputs=[
                    "features_train",
                    "features_test",
                    "feature_transformers",
                    "params:feature_engineering",
                ],
                outputs="feature_engineering_metrics",
                name="log_feature_engineering_node",
            ),
        ]
    )
