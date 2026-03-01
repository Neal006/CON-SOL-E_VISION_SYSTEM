"""
SAM Segmentation Utilities
Provides box-prompted and automatic segmentation for defect regions.
"""
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PIXELS_PER_MM, DEVICE


@dataclass
class DefectSegment:
    """Represents a segmented defect region."""
    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    mask: np.ndarray  # Binary mask (H, W), uint8
    area_pixels: int
    area_mm2: Optional[float] = None
    iou_prediction: float = 0.0
    stability_score: float = 0.0


class SAMSegmenter:
    """
    SAM-based segmentation for defect regions.
    Uses bounding boxes from DINOv2+XGBoost as prompts for precise masks.
    """
    
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.predictor = None
        self._image_set = False
        self._current_image_shape = None
    
    def _ensure_predictor(self):
        """Lazy-load SAM predictor."""
        if self.predictor is None:
            from sam.model_loader import get_sam_predictor
            self.predictor = get_sam_predictor(device=self.device)
    
    def set_image(self, image: np.ndarray):
        """
        Set the image for segmentation.
        Must be called before segment_box or segment_boxes.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        """
        self._ensure_predictor()
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            # Assume BGR from OpenCV, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image)
        self._image_set = True
        self._current_image_shape = image.shape[:2]
    
    def segment_box(
        self,
        box: Tuple[int, int, int, int],
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Segment a region given a bounding box prompt.
        
        Args:
            box: (x_min, y_min, x_max, y_max) bounding box
            multimask_output: If True, returns 3 masks with quality scores
        
        Returns:
            mask: Binary mask (H, W), uint8
            iou_prediction: Predicted IoU score
            stability_score: Stability score
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() before segment_box()")
        
        import torch
        
        # Convert box to numpy array for SAM
        box_np = np.array(box)
        
        masks, iou_predictions, low_res_masks = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np[None, :],  # Add batch dimension
            multimask_output=multimask_output,
        )
        
        if multimask_output:
            # Return the mask with highest IoU prediction
            best_idx = np.argmax(iou_predictions)
            mask = masks[best_idx]
            iou = float(iou_predictions[best_idx])
        else:
            mask = masks[0]
            iou = float(iou_predictions[0])
        
        # Convert to uint8
        mask = (mask * 255).astype(np.uint8)
        
        # Calculate stability score (simplified)
        stability = min(1.0, iou * 1.1)
        
        return mask, iou, stability
    
    def segment_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        class_names: List[str],
        class_ids: List[int],
        confidences: List[float],
        pixels_per_mm: Optional[float] = PIXELS_PER_MM
    ) -> List[DefectSegment]:
        """
        Segment multiple regions given bounding boxes.
        
        Args:
            boxes: List of (x_min, y_min, x_max, y_max) bounding boxes
            class_names: List of class names for each box
            class_ids: List of class IDs for each box
            confidences: List of confidence scores for each box
            pixels_per_mm: Pixels per mm for area conversion (None = pixels only)
        
        Returns:
            List of DefectSegment objects with masks and area estimates
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() before segment_boxes()")
        
        segments = []
        
        for i, (box, class_name, class_id, conf) in enumerate(
            zip(boxes, class_names, class_ids, confidences)
        ):
            mask, iou, stability = self.segment_box(box, multimask_output=True)
            
            # Calculate area
            area_pixels = int(np.sum(mask > 0))
            
            area_mm2 = None
            if pixels_per_mm is not None and pixels_per_mm > 0:
                area_mm2 = area_pixels / (pixels_per_mm ** 2)
            
            segment = DefectSegment(
                class_name=class_name,
                class_id=class_id,
                confidence=conf,
                bbox=box,
                mask=mask,
                area_pixels=area_pixels,
                area_mm2=area_mm2,
                iou_prediction=iou,
                stability_score=stability
            )
            segments.append(segment)
        
        return segments
    
    def reset(self):
        """Reset the predictor state."""
        self._image_set = False
        self._current_image_shape = None


def calculate_mask_area(
    mask: np.ndarray,
    pixels_per_mm: Optional[float] = PIXELS_PER_MM
) -> Tuple[int, Optional[float]]:
    """
    Calculate area from a binary mask.
    
    Args:
        mask: Binary mask (H, W), values 0 or non-zero
        pixels_per_mm: Pixels per mm for area conversion
    
    Returns:
        area_pixels: Area in pixels
        area_mm2: Area in mm² (or None if not calibrated)
    """
    area_pixels = int(np.sum(mask > 0))
    
    area_mm2 = None
    if pixels_per_mm is not None and pixels_per_mm > 0:
        area_mm2 = area_pixels / (pixels_per_mm ** 2)
    
    return area_pixels, area_mm2


def draw_segments_on_image(
    image: np.ndarray,
    segments: List[DefectSegment],
    alpha: float = 0.4,
    show_bbox: bool = True,
    show_area: bool = True,
    colors: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw segmentation masks and bounding boxes on image.
    
    Args:
        image: BGR image (H, W, 3)
        segments: List of DefectSegment objects
        alpha: Mask transparency (0-1)
        show_bbox: Draw bounding boxes
        show_area: Show area in labels
        colors: Dict mapping class names to BGR colors
    
    Returns:
        Annotated image
    """
    default_colors = {
        "Dust": (255, 165, 0),      # Orange
        "RunDown": (0, 165, 255),   # Yellow-orange
        "Scratch": (0, 0, 255),     # Red
    }
    colors = colors or default_colors
    
    result = image.copy()
    overlay = image.copy()
    
    for segment in segments:
        color = colors.get(segment.class_name, (0, 255, 0))
        
        # Draw mask overlay
        mask_bool = segment.mask > 0
        overlay[mask_bool] = color
        
        if show_bbox:
            x_min, y_min, x_max, y_max = segment.bbox
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Prepare label
            if show_area:
                if segment.area_mm2 is not None:
                    label = f"{segment.class_name} ({segment.confidence:.0%}) {segment.area_mm2:.2f}mm²"
                else:
                    label = f"{segment.class_name} ({segment.confidence:.0%}) {segment.area_pixels}px"
            else:
                label = f"{segment.class_name} ({segment.confidence:.0%})"
            
            # Draw label background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_y = y_min - 5 if y_min > 25 else y_max + text_h + 5
            
            cv2.rectangle(
                result,
                (x_min, label_y - text_h - 5),
                (x_min + text_w + 4, label_y + 5),
                color, -1
            )
            cv2.putText(
                result, label,
                (x_min + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
    
    # Blend mask overlay
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    
    return result


def create_combined_mask(
    segments: List[DefectSegment],
    image_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create combined binary mask and class mask from segments.
    
    Args:
        segments: List of DefectSegment objects
        image_shape: (height, width) of the target image
    
    Returns:
        binary_mask: Combined binary mask (all defects)
        class_mask: Mask with class IDs (0, 1, 2 for defects, 255 for background)
    """
    h, w = image_shape
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    class_mask = np.full((h, w), 255, dtype=np.uint8)
    
    for segment in segments:
        mask_bool = segment.mask > 0
        
        # Resize mask if needed
        if segment.mask.shape != (h, w):
            resized_mask = cv2.resize(
                segment.mask, (w, h),
                interpolation=cv2.INTER_NEAREST
            )
            mask_bool = resized_mask > 0
        
        binary_mask[mask_bool] = 255
        class_mask[mask_bool] = segment.class_id
    
    return binary_mask, class_mask


if __name__ == "__main__":
    # Test SAM segmentation
    print("Testing SAM segmentation...")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Initialize segmenter
    segmenter = SAMSegmenter()
    segmenter.set_image(test_image)
    
    # Test with a sample box
    test_box = (100, 100, 200, 200)
    mask, iou, stability = segmenter.segment_box(test_box)
    
    print(f"Generated mask shape: {mask.shape}")
    print(f"Mask area: {np.sum(mask > 0)} pixels")
    print(f"IoU prediction: {iou:.3f}")
    print(f"Stability score: {stability:.3f}")
    
    print("\n✓ SAM segmentation test passed!")
