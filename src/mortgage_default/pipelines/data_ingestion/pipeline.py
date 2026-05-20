"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import assign_origination_columns_names


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func =  assign_origination_columns_names,
            inputs = "raw_origination_data",
            outputs = "origination_data_named",
            name = "assign_origination_columns_node" 
        ),

    ])
