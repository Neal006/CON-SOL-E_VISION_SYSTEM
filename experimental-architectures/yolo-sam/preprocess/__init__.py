"""
Preprocessing module for YOLO + SAM Defect Detection Pipeline.

Contains:
- Label decoding (polygon to bounding box conversion)
- Channel handling (grayscale to 3-channel expansion)
"""

from .label_decoder import (
    parse_yolo_polygon,
    polygon_to_bbox,
    decode_label_file,
    create_binary_mask
)
from .channel_handler import (
    expand_grayscale,
    prepare_for_model,
    normalize_image
)

__all__ = [
    "parse_yolo_polygon",
    "polygon_to_bbox",
    "decode_label_file",
    "create_binary_mask",
    "expand_grayscale",
    "prepare_for_model",
    "normalize_image"
]
