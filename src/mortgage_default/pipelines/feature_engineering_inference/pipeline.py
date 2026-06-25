"""Pipeline 'feature_engineering_inference'.

Reutiliza os nodes do pipeline de treino, mas SÓ a parte de transformação:
- drop_unused_columns e engineer_date_features (iguais ao treino)
- apply_feature_transformers carregando 'feature_transformers' do disco

Não há fit nem split aqui: os parâmetros (medianas, categorias, colunas) vêm
do treino, o que garante exatamente as mesmas features e zero leakage.
"""
from kedro.pipeline import Node, Pipeline

from ..feature_engineering.nodes import (
    apply_feature_transformers,
    drop_unused_columns,
    engineer_date_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=drop_unused_columns,
                inputs="origination_data_inference",
                outputs="inference_no_unused",
                name="fe_inf_drop_unused_columns_node",
            ),
            Node(
                func=engineer_date_features,
                inputs="inference_no_unused",
                outputs="inference_dated",
                name="fe_inf_engineer_date_features_node",
            ),
            Node(
                func=apply_feature_transformers,
                inputs=["inference_dated", "feature_transformers"],
                outputs="features_inference",
                name="fe_inf_apply_transformers_node",
            ),
        ]
    )
