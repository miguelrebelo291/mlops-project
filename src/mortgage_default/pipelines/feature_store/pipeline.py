"""Pipeline 'feature_store' — upload das features para o Hopsworks."""
from kedro.pipeline import Node, Pipeline

from .nodes import upload_to_feature_store


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=upload_to_feature_store,
                inputs=["features_train", "params:feature_store"],
                outputs="feature_store_metadata",
                name="upload_to_feature_store_node",
            ),
        ]
    )
