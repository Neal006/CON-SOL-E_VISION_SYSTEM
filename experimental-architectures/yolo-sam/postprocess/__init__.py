"""
Post-processing module for defect detection pipeline.

Contains:
- Mask thresholding
- Connected component analysis
- Geometry-derived bounding boxes
- Area computation (pixel to mm²)
"""

from .thresholding import apply_threshold, adaptive_threshold
from .connected_components import find_components, separate_instances
from .geometry import mask_to_bbox, compute_bbox
from .area_computation import compute_pixel_area, pixel_to_mm2

__all__ = [
    "apply_threshold",
    "adaptive_threshold",
    "find_components",
    "separate_instances",
    "mask_to_bbox",
    "compute_bbox",
    "compute_pixel_area",
    "pixel_to_mm2"
]
