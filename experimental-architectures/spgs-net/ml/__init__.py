"""
SPGS-Net ML Package
===================
Classical ML for patch-level defect awareness.
"""

from .xgboost_classifier import PatchClassifier, create_anomaly_heatmap

__all__ = [
    "PatchClassifier",
    "create_anomaly_heatmap",
]
