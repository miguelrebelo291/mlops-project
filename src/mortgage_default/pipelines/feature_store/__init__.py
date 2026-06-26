"""
This is a pipeline 'feature_store'
that uploads engineered features to the Hopsworks Feature Store.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
