"""
SPGS-Net DINOv2 Feature Extraction Package
===========================================
Frozen self-supervised feature extraction using DINOv2.
"""

from .feature_extractor import DINOv2Extractor, extract_patch_features

__all__ = [
    "DINOv2Extractor",
    "extract_patch_features",
]
