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
    concatenate_years,
)

YEARS = list(range(2000, 2009))  # 2000 a 2008


def create_pipeline(**kwargs) -> Pipeline:
    nodes = []

    for year in YEARS:
        nodes += [
            Node(
                func=assign_origination_columns_names,
                inputs=f"raw_origination_data_{year}",
                outputs=f"origination_data_named_{year}",
                name=f"assign_origination_columns_node_{year}",
            ),
            Node(
                func=assign_performance_columns_names,
                inputs=f"raw_performance_data_{year}",
                outputs=f"performance_data_named_{year}",
                name=f"assign_performance_columns_node_{year}",
            ),
            Node(
                func=create_target_variable,
                inputs=f"performance_data_named_{year}",
                outputs=f"target_data_{year}",
                name=f"create_target_variable_node_{year}",
            ),
            Node(
                func=join_origination_with_target,
                inputs=[f"origination_data_named_{year}", f"target_data_{year}"],
                outputs=f"model_input_data_{year}",
                name=f"join_origination_with_target_node_{year}",
            ),
        ]

    # Junta todos os anos num único dataset com coluna 'year'
    nodes.append(
        Node(
            func=concatenate_years,
            inputs=[f"model_input_data_{year}" for year in YEARS],
            outputs="model_input_data_all",
            name="concatenate_years_node",
        )
    )

    return Pipeline(nodes)
