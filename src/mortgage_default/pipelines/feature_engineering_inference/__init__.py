"""
Pipeline 'feature_engineering_inference'.

Aplica a empréstimos novos as MESMAS transformações aprendidas no treino,
carregando o artefacto 'feature_transformers' (sem fit, sem split) para
garantir consistência e evitar leakage.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
