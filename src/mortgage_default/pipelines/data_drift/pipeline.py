from kedro.pipeline import Pipeline, node

from .nodes import analyze_data_drift


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=analyze_data_drift,
                inputs=[
                    "features_train",
                    "features_inference",
                    "params:data_drift",
                ],
                outputs=[
                    "data_drift_report",
                    "data_drift_metrics",
                ],
                name="analyze_data_drift_node",
            )
        ]
    )