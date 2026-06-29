from kedro.pipeline import Pipeline, node

from .nodes import evaluate_labeled_inference, run_model_inference


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
            ),
            node(
                func=evaluate_labeled_inference,
                inputs=[
                    "features_inference",
                    "model_inference_predictions",
                    "params:model_inference",
                ],
                outputs=[
                    "inference_evaluation_metrics",
                    "evaluated_inference_predictions",
                ],
                name="evaluate_labeled_inference_node",
            ),
        ]
    )