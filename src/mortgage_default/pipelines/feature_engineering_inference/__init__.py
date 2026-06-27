"""
Pipeline 'feature_engineering_inference'.

Applies to new loans the SAME transformations learned during training,
loading the 'feature_transformers' artifact (no fit, no split) to
ensure consistency and avoid leakage.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
