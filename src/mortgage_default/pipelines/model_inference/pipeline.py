from kedro.pipeline import Pipeline, node

from .nodes import run_model_inference


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_model_inference,
                inputs=[
                    "features_inference",
                    "model_training_metadata",
                    "params:model_inference",
                ],
                outputs="model_inference_predictions",
                name="run_model_inference_node",
            )
        ]
    )
