"""Pipeline 'model_train'."""
from kedro.pipeline import Node, Pipeline

from .nodes import (
    compute_shap_values,
    prepare_features,
    select_best_model,
    train_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=prepare_features,
                inputs=["model_input_data_all", "params:model_train"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="prepare_features_node",
            ),
            Node(
                func=train_models,
                inputs=["X_train", "X_test", "y_train", "y_test", "params:model_train"],
                outputs="trained_models",
                name="train_models_node",
            ),
            Node(
                func=compute_shap_values,
                inputs=["trained_models", "X_test", "X_train"],
                outputs="shap_results",
                name="compute_shap_values_node",
            ),
            Node(
                func=select_best_model,
                inputs=["trained_models", "shap_results"],
                outputs="model_selection_result",
                name="select_best_model_node",
            ),
        ]
    )
