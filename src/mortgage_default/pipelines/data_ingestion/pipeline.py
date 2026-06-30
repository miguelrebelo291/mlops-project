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
    prepare_nsd_sample,
)
import os


def _get_years() -> list[int]:
    """Years to ingest."""
    raw_years = os.getenv("MORTGAGE_DATA_YEARS", "2005")
    return [int(year.strip()) for year in raw_years.split(",") if year.strip()]


YEARS = _get_years()


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

    # Prepara o NSD sample (já tem target e coluna 'year') para entrar
    # na mesma concatenação que os anos do Standard Dataset.
    nodes.append(
        Node(
            func=prepare_nsd_sample,
            inputs="nsd_sample_raw",
            outputs="nsd_sample_prepared",
            name="prepare_nsd_sample_node",
        )
    )

    # Junta todos os anos do Standard + o NSD num único dataset
    standard_inputs = [f"model_input_data_{year}" for year in YEARS]

    nodes.append(
        Node(
            func=concatenate_years,
            inputs=standard_inputs + ["nsd_sample_prepared"],
            outputs="model_input_data_all",
            name="concatenate_years_node",
        )
    )

    return Pipeline(nodes)
