"""
This is a boilerplate pipeline 'data_quality'
generated using Kedro 1.3.1
"""
from kedro.pipeline import Node, Pipeline
from .nodes import run_data_quality


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=run_data_quality,
            inputs="model_input_data_all",
            outputs="model_input_validated",
            name="run_data_quality_node",
        ),
    ])
