import os
import sys
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    _current_dir = Path(__file__).parent.parent
except NameError:
    _current_dir = Path(os.getcwd())
sys.path.insert(0, str(_current_dir))
from config import PIXELS_PER_MM, DEVICE, MASKRCNN_SCORE_THRESH


@dataclass
class DefectSegment:
    """Same interface as SAM's DefectSegment for compatibility."""
    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    area_pixels: int
    area_mm2: Optional[float] = None
    iou_prediction: float = 0.0
    stability_score: float = 0.0


class MaskRCNNSegmenter:
    """
    Mask R-CNN based segmenter with same interface as SAMSegmenter.
    
    Uses pretrained Mask R-CNN as a mask generator given bounding box prompts
    from the XGBoost defect detection stage.
    """
    
    def __init__(self, device: str = DEVICE, score_thresh: float = MASKRCNN_SCORE_THRESH):
        self.device = device
        self.score_thresh = score_thresh
        self.model = None
        self._image_set = False
        self._current_image = None
        self._current_image_shape = None
    
    def _ensure_model(self):
        """Lazy load the Mask R-CNN model."""
        if self.model is None:
            from mask_rcnn.model_loader import get_mask_rcnn_model
            self.model = get_mask_rcnn_model(device=self.device, score_thresh=self.score_thresh)
    
    def set_image(self, image: np.ndarray):
        """
        Set the image for segmentation.
        
        Args:
            image: BGR or RGB image as numpy array
        """
        self._ensure_model()
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self._current_image = image
        self._image_set = True
        self._current_image_shape = image.shape[:2]
    
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to normalized tensor for Mask R-CNN."""
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image.to(self.device)
    
    def _crop_and_segment(self, box: Tuple[int, int, int, int], padding: int = 20) -> Tuple[np.ndarray, float]:
        """
        Crop region around box, run Mask R-CNN, and return the best mask.
        
        Args:
            box: Bounding box (x_min, y_min, x_max, y_max)
            padding: Extra padding around the box
        
        Returns:
            Tuple of (mask, confidence)
        """
        x_min, y_min, x_max, y_max = box
        h, w = self._current_image_shape
        
        pad_x_min = max(0, x_min - padding)
        pad_y_min = max(0, y_min - padding)
        pad_x_max = min(w, x_max + padding)
        pad_y_max = min(h, y_max + padding)
        
        crop = self._current_image[pad_y_min:pad_y_max, pad_x_min:pad_x_max]
        
        if crop.size == 0:
            return self._create_box_mask(box), 0.5
        
        crop_tensor = self._image_to_tensor(crop)
        
        with torch.no_grad():
            predictions = self.model([crop_tensor])
        
        pred = predictions[0]
        
        if len(pred['masks']) == 0:
            return self._create_box_mask(box), 0.5
        
        rel_box = (
            x_min - pad_x_min,
            y_min - pad_y_min,
            x_max - pad_x_min,
            y_max - pad_y_min
        )
        
        best_mask, best_score = self._select_best_mask(pred, rel_box)
        
        full_mask = np.zeros((h, w), dtype=np.uint8)
        if best_mask is not None:
            mask_h, mask_w = best_mask.shape
            full_mask[pad_y_min:pad_y_min+mask_h, pad_x_min:pad_x_min+mask_w] = best_mask
        else:
            full_mask = self._create_box_mask(box)
            best_score = 0.5
        
        return full_mask, best_score
    
    def _select_best_mask(self, pred: Dict, target_box: Tuple[int, int, int, int]) -> Tuple[Optional[np.ndarray], float]:
        """
        Select the mask that best overlaps with the target bounding box.
        
        Args:
            pred: Model predictions dict with 'masks', 'boxes', 'scores'
            target_box: Target bounding box to match
        
        Returns:
            Tuple of (best_mask, best_score)
        """
        masks = pred['masks'].cpu().numpy()
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        if len(masks) == 0:
            return None, 0.0
        
        best_mask = None
        best_iou = 0.0
        best_score = 0.0
        
        tx_min, ty_min, tx_max, ty_max = target_box
        
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            bx_min, by_min, bx_max, by_max = box
            
            inter_x_min = max(tx_min, bx_min)
            inter_y_min = max(ty_min, by_min)
            inter_x_max = min(tx_max, bx_max)
            inter_y_max = min(ty_max, by_max)
            
            if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                target_area = (tx_max - tx_min) * (ty_max - ty_min)
                box_area = (bx_max - bx_min) * (by_max - by_min)
                union_area = target_area + box_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_mask = (mask[0] > 0.5).astype(np.uint8) * 255
                    best_score = float(score)
        
        if best_mask is None and len(masks) > 0:
            best_idx = np.argmax(scores)
            best_mask = (masks[best_idx][0] > 0.5).astype(np.uint8) * 255
            best_score = float(scores[best_idx])
        
        return best_mask, best_score
    
    def _create_box_mask(self, box: Tuple[int, int, int, int]) -> np.ndarray:
        """Create a fallback rectangular mask from bounding box."""
        h, w = self._current_image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        x_min, y_min, x_max, y_max = box
        mask[y_min:y_max, x_min:x_max] = 255
        return mask
    
    def segment_box(self, box: Tuple[int, int, int, int], multimask_output: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Generate a mask for a single bounding box.
        
        Args:
            box: Bounding box (x_min, y_min, x_max, y_max)
            multimask_output: Ignored, kept for API compatibility
        
        Returns:
            Tuple of (mask, iou_prediction, stability_score)
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() before segment_box()")
        
        mask, score = self._crop_and_segment(box)
        
        stability = min(1.0, score * 1.1)
        
        return mask, score, stability
    
    def segment_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        class_names: List[str],
        class_ids: List[int],
        confidences: List[float],
        pixels_per_mm: Optional[float] = PIXELS_PER_MM
    ) -> List[DefectSegment]:
        """
        Generate masks for multiple bounding boxes.
        
        Args:
            boxes: List of bounding boxes
            class_names: Defect class names
            class_ids: Defect class IDs
            confidences: Detection confidences
            pixels_per_mm: For area conversion
        
        Returns:
            List of DefectSegment objects
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() before segment_boxes()")
        
        segments = []
        
        for box, class_name, class_id, conf in zip(boxes, class_names, class_ids, confidences):
            mask, iou, stability = self.segment_box(box)
            
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
        """Reset the segmenter state."""
        self._image_set = False
        self._current_image = None
        self._current_image_shape = None


def calculate_mask_area(mask: np.ndarray, pixels_per_mm: Optional[float] = PIXELS_PER_MM) -> Tuple[int, Optional[float]]:
    """Calculate area in pixels and mm² if calibration is provided."""
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
    """Draw segmentation masks and bounding boxes on image."""
    default_colors = {"Dust": (255, 165, 0), "RunDown": (0, 165, 255), "Scratch": (0, 0, 255)}
    colors = colors or default_colors
    
    result = image.copy()
    overlay = image.copy()
    
    for segment in segments:
        color = colors.get(segment.class_name, (0, 255, 0))
        
        mask_bool = segment.mask > 0
        overlay[mask_bool] = color
        
        if show_bbox:
            x_min, y_min, x_max, y_max = segment.bbox
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, 2)
            
            if show_area:
                if segment.area_mm2 is not None:
                    label = f"{segment.class_name} ({segment.confidence:.0%}) {segment.area_mm2:.2f}mm²"
                else:
                    label = f"{segment.class_name} ({segment.confidence:.0%}) {segment.area_pixels}px"
            else:
                label = f"{segment.class_name} ({segment.confidence:.0%})"
            
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y_min - 5 if y_min > 25 else y_max + text_h + 5
            cv2.rectangle(result, (x_min, label_y - text_h - 5), (x_min + text_w + 4, label_y + 5), color, -1)
            cv2.putText(result, label, (x_min + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    return result


def create_combined_mask(segments: List[DefectSegment], image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Create combined binary and class masks from segments."""
    h, w = image_shape
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    class_mask = np.full((h, w), 255, dtype=np.uint8)
    
    for segment in segments:
        mask_bool = segment.mask > 0
        if segment.mask.shape != (h, w):
            resized_mask = cv2.resize(segment.mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bool = resized_mask > 0
        binary_mask[mask_bool] = 255
        class_mask[mask_bool] = segment.class_id
    
    return binary_mask, class_mask


if __name__ == "__main__":
    print("Testing Mask R-CNN segmentation...")
    
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    segmenter = MaskRCNNSegmenter()
    segmenter.set_image(test_image)
    
    test_box = (100, 100, 200, 200)
    mask, iou, stability = segmenter.segment_box(test_box)
    
    print(f"Generated mask shape: {mask.shape}")
    print(f"Mask area: {np.sum(mask > 0)} pixels")
    print(f"IoU prediction: {iou:.3f}")
    print(f"Stability score: {stability:.3f}")
    
    print("✓ Mask R-CNN segmentation test passed!")
