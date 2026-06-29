from kedro.pipeline import Pipeline, node

from mortgage_default.pipelines.data_cleaning.nodes import clean_origination_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_origination_data,
                inputs="model_input_data_inference_labeled",
                outputs=[
                    "model_input_data_inference_cleaned",
                    "inference_cleaning_statistics",
                ],
                name="clean_labeled_inference_data_node",
            )
        ]
    )