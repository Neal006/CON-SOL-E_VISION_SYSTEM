"""
SAM (Segment Anything Model) module for instance segmentation.

Contains:
- SAM segmentor wrapper
- Prompt builder for YOLO bbox to SAM prompt conversion
- Mask generation pipeline
"""

from .segmentor import SAMSegmentor
from .prompt_builder import bbox_to_prompt, build_prompts
from .mask_generator import MaskGenerator

__all__ = [
    "SAMSegmentor",
    "bbox_to_prompt",
    "build_prompts",
    "MaskGenerator"
]
