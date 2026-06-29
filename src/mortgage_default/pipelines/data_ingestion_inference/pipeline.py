from kedro.pipeline import Pipeline, node

from mortgage_default.pipelines.data_ingestion.nodes import (
    assign_origination_columns_names,
    assign_performance_columns_names,
    create_target_variable,
    join_origination_with_target,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=assign_origination_columns_names,
                inputs="raw_origination_data_inference",
                outputs="origination_data_inference_named",
                name="assign_inference_origination_columns_node",
            ),
            node(
                func=assign_performance_columns_names,
                inputs="raw_performance_data_inference",
                outputs="performance_data_inference_named",
                name="assign_inference_performance_columns_node",
            ),
            node(
                func=create_target_variable,
                inputs="performance_data_inference_named",
                outputs="target_data_inference",
                name="create_inference_target_variable_node",
            ),
            node(
                func=join_origination_with_target,
                inputs=[
                    "origination_data_inference_named",
                    "target_data_inference",
                ],
                outputs="model_input_data_inference_labeled",
                name="join_inference_origination_with_target_node",
            ),
        ]
    )