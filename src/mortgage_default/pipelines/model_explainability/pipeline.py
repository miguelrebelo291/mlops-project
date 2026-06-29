from kedro.pipeline import Pipeline, node

from .nodes import generate_shap_report


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=generate_shap_report,
                inputs=[
                    "features_test",
                    "params:model_explainability",
                ],
                outputs="shap_metrics",
                name="generate_shap_report_node",
            )
        ]
    )