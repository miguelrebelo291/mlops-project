from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
 
 
def _combine(pipelines: dict[str, Pipeline], names: list[str]) -> Pipeline:
    missing = [name for name in names if name not in pipelines]
    if missing:
        raise KeyError(f"Missing expected pipelines: {missing}")
 
    result = Pipeline([])
    for name in names:
        result += pipelines[name]
    return result
 
 
def register_pipelines() -> dict[str, Pipeline]:
    pipelines = find_pipelines(raise_errors=True)
 
    pipelines["training"] = _combine(
        pipelines,
        [
            "data_ingestion",
            "data_quality",
            "data_cleaning",
            "model_train",
        ],
    )
 
    pipelines["__default__"] = pipelines["training"]
 
    return pipelines
 