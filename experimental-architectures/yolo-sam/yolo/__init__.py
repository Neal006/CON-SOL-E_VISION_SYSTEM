"""
YOLO module for defect detection.

Contains:
- YOLOv8 detector wrapper
- Training configuration
- Inference pipeline
"""

from .detector import YOLODetector
from .trainer import YOLOTrainer
from .inference import run_inference, batch_inference

__all__ = [
    "YOLODetector",
    "YOLOTrainer",
    "run_inference",
    "batch_inference"
]
