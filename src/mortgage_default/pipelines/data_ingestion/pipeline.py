"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.3.1
"""

"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.3.1
"""
from kedro.pipeline import Node, Pipeline
from .nodes import (
    assign_origination_columns_names,
    assign_performance_columns_names,
    create_target_variable,
    join_origination_with_target,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
       Node(
    func=assign_origination_columns_names,
    inputs="raw_origination_data_2005",
    outputs="origination_data_named",
    name="assign_origination_columns_node",
        ),
        Node(
            func=assign_performance_columns_names,
            inputs="raw_performance_data_2005",
            outputs="performance_data_named",
            name="assign_performance_columns_node",
        ),
        Node(
            func=create_target_variable,
            inputs="performance_data_named",
            outputs="target_data",
            name="create_target_variable_node",
        ),
        Node(
            func=join_origination_with_target,
            inputs=["origination_data_named", "target_data"],
            outputs="model_input_data",
            name="join_origination_with_target_node",
        ),
    ])