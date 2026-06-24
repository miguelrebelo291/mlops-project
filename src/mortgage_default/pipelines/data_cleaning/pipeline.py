"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Node, Pipeline
from .nodes import clean_origination_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=clean_origination_data,
                inputs="model_input_validated",
                outputs=[
                    "origination_data_cleaned",
                    "origination_cleaning_statistics",
                ],
                name="clean_origination_data_node",
            )
        ]
    )