from kedro.pipeline import Pipeline, node

from .nodes import train_validate_test_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_validate_test_model,
                inputs=[
                    "features_train",
                    "features_test",
                    "params:model_train",
                ],
                outputs=[
                    "model_training_metrics",
                    "model_training_metadata",
                    "model_test_predictions",
                    "model_search_results",
                ],
                name="train_validate_test_model_node",
            )
        ]
    )