"""
Evaluation module for defect detection pipeline.

Contains:
- Detection metrics (Recall, Precision, Box IoU, Miss rate)
- Segmentation metrics (Mask IoU, Dice score, Boundary accuracy)
- Physical metrics (Area error, Median mm² deviation)
- Main evaluator orchestrator
"""

from .detection_metrics import (
    compute_recall,
    compute_precision,
    compute_box_iou,
    compute_miss_rate,
    compute_map
)
from .segmentation_metrics import (
    compute_mask_iou,
    compute_dice_score,
    compute_boundary_accuracy
)
from .physical_metrics import (
    compute_area_error,
    compute_median_deviation,
    compute_per_class_accuracy
)
from .evaluator import Evaluator

__all__ = [
    "compute_recall",
    "compute_precision",
    "compute_box_iou",
    "compute_miss_rate",
    "compute_map",
    "compute_mask_iou",
    "compute_dice_score",
    "compute_boundary_accuracy",
    "compute_area_error",
    "compute_median_deviation",
    "compute_per_class_accuracy",
    "Evaluator"
]
