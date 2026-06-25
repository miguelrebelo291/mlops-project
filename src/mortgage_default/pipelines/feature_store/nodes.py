"""Nodes for the feature_store pipeline (Hopsworks Feature Store)."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

ID_COL = "loan_sequence_number"


def _sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Hopsworks só aceita nomes de features em minúsculas e com underscores.

    O one-hot gera colunas como 'occupancy_status_O' (com maiúsculas), por isso
    normalizamos tudo antes de inserir no feature group.
    """
    df = df.copy()
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns
    ]
    return df


def upload_to_feature_store(features: pd.DataFrame, parameters: dict) -> dict:
    """Insere as features num Feature Group do Hopsworks.

    A API key é lida da variável de ambiente HOPSWORKS_API_KEY (nunca em código).

    Args:
        features: tabela de features (tem de incluir loan_sequence_number).
        parameters: config do feature group (parameters_feature_store.yml).

    Returns:
        Dicionário com o nome/versão do feature group e nº de linhas inseridas.
    """
    import hopsworks  # import tardio: só precisa do pacote quando se faz upload

    features = _sanitize_feature_names(features)
    if ID_COL not in features.columns:
        raise ValueError(f"'{ID_COL}' tem de existir para ser a primary key.")

    project = hopsworks.login(
        project=parameters.get("project") or None,
    )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=parameters["feature_group_name"],
        version=parameters["feature_group_version"],
        description=parameters.get(
            "description", "Mortgage default origination features"
        ),
        primary_key=[ID_COL],
        online_enabled=parameters.get("online_enabled", False),
        time_travel_format=parameters.get("time_travel_format", "NONE"),
    )

    fg.insert(features, write_options={"wait_for_job": True})

    result = {
        "feature_group": parameters["feature_group_name"],
        "version": parameters["feature_group_version"],
        "n_rows": int(len(features)),
        "n_features": int(features.shape[1]),
    }
    logger.info("Inserido no Hopsworks Feature Store: %s", result)
    return result
