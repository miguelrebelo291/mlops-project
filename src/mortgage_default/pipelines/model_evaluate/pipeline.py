"""Pipeline 'model_evaluate'.

Applies the model trained by model_train to the temporal test set,
exactly once. Separated from model_train so evaluation can be re-run
independently (e.g. testing a different threshold or test slice)
without repeating the expensive random search.

Requires selected_model, features_test, and train_metadata to be
available — either from the same Kedro session as model_train, or
persisted in the catalog (Pickle / Parquet / JSON) from a previous run.
"""
from kedro.pipeline import Node, Pipeline

from .nodes import evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=evaluate_model,
                inputs=[
                    "selected_model",
                    "features_test",
                    "train_metadata",
                    "params:model_train",
                ],
                outputs=[
                    "model_training_metrics",
                    "model_training_metadata",
                    "model_test_predictions",
                ],
                name="evaluate_model_node",
            ),
        ]
    )
