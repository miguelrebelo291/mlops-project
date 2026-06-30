"""Pipeline 'model_train'.

Temporal data preparation + holdout random search across model families.
Never touches the test set — that happens separately in model_evaluate,
so this pipeline can be run, cached, and tested on its own.
"""
from kedro.pipeline import Node, Pipeline

from .nodes import (
    apply_feature_transformers,
    drop_unused_columns,
    engineer_date_features,
    fit_feature_transformers,
    split_temporal,
    train_search,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=drop_unused_columns,
                inputs="origination_data_cleaned",
                outputs="model_train_features_no_unused",
                name="model_train_drop_unused_columns_node",
            ),
            Node(
                func=engineer_date_features,
                inputs="model_train_features_no_unused",
                outputs="model_train_features_dated",
                name="model_train_engineer_date_features_node",
            ),
            Node(
                func=split_temporal,
                inputs=["model_train_features_dated", "params:model_train"],
                outputs=["train_raw", "test_raw"],
                name="split_temporal_node",
            ),
            Node(
                func=fit_feature_transformers,
                inputs="train_raw",
                outputs="feature_transformers",
                name="fit_feature_transformers_node",
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
                func=train_search,
                inputs=["features_train", "params:model_train"],
                outputs=[
                    "selected_model",
                    "train_metadata",
                    "model_search_results",
                ],
                name="train_search_node",
            ),
        ]
    )
