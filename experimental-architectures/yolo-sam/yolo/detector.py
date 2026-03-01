"""
YOLOv8 Detector Wrapper.

Provides a high-level interface for YOLOv8 detection with:
- Recall-first configuration (low confidence, high NMS IoU)
- Grayscale image support (auto-expanded to 3 channels)
- Structured output format for downstream SAM processing
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class YOLODetector:
    """
    YOLOv8 Detector wrapper optimized for defect detection.
    
    Features:
    - Recall-biased inference settings
    - Support for grayscale images
    - Structured detection output
    - Auto-loads settings from config.py
    
    Attributes:
        model: Loaded YOLOv8 model
        conf_threshold: Confidence threshold
        nms_iou: NMS IoU threshold
        class_names: Mapping of class IDs to names
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_variant: str = None,
        confidence_threshold: float = None,
        nms_iou_threshold: float = None,
        device: str = None,
        class_names: Dict[int, str] = None,
        config = None
    ):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to custom trained weights (.pt file)
            model_variant: YOLO variant if no custom weights (e.g., yolov8n-seg)
            confidence_threshold: Detection confidence threshold (uses config if None)
            nms_iou_threshold: NMS IoU threshold (uses config if None)
            device: Device to run inference on (uses config if None)
            class_names: Optional class name mapping (uses config if None)
            config: Optional Config object (auto-loads if None)
        """
        # Load config if not provided
        if config is None:
            from config import get_config
            config = get_config()
        
        # Use config values as defaults, allow overrides
        self.conf_threshold = confidence_threshold if confidence_threshold is not None else config.yolo.confidence_threshold
        self.nms_iou = nms_iou_threshold if nms_iou_threshold is not None else config.yolo.nms_iou_threshold
        self.device = device if device is not None else config.yolo.device
        self.class_names = class_names if class_names is not None else dict(config.data.class_names)
        
        model_variant = model_variant if model_variant is not None else config.yolo.model_variant
        
        # Load model
        self.model = self._load_model(model_path, model_variant)
        
    def _load_model(self, model_path: Optional[str], model_variant: str):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not found. "
                "Install with: pip install ultralytics"
            )
        
        if model_path and Path(model_path).exists():
            model = YOLO(model_path)
        else:
            # Load pretrained model
            model = YOLO(model_variant)
        
        # Move to device
        model.to(self.device)
        
        return model
    
    def detect(
        self,
        image: Union[np.ndarray, str, Path],
        return_masks: bool = True
    ) -> List[Dict]:
        """
        Run detection on a single image.
        
        Args:
            image: Input image (numpy array or path)
            return_masks: Whether to return segmentation masks
        
        Returns:
            List of detections, each containing:
                - bbox: (x_min, y_min, x_max, y_max)
                - class_id: Integer class identifier
                - class_name: String class name
                - confidence: Detection confidence score
                - mask: Optional segmentation mask (if return_masks=True)
        """
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.nms_iou,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
            
            for i, box in enumerate(boxes):
                # Get bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = xyxy
                
                # Get class and confidence
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                detection = {
                    "bbox": (float(x_min), float(y_min), float(x_max), float(y_max)),
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                    "confidence": confidence
                }
                
                # Add mask if available and requested
                if return_masks and masks is not None and i < len(masks):
                    mask = masks[i].data.cpu().numpy()
                    if mask.ndim == 3:
                        mask = mask[0]  # Remove batch dimension
                    detection["mask"] = mask
                
                detections.append(detection)
        
        return detections
    
    def detect_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        return_masks: bool = True
    ) -> List[List[Dict]]:
        """
        Run detection on a batch of images.
        
        Args:
            images: List of input images
            return_masks: Whether to return segmentation masks
        
        Returns:
            List of detection lists, one per image
        """
        all_detections = []
        
        for image in images:
            detections = self.detect(image, return_masks=return_masks)
            all_detections.append(detections)
        
        return all_detections
    
    def get_bboxes_for_sam(
        self,
        detections: List[Dict]
    ) -> List[Tuple[float, float, float, float]]:
        """
        Extract bounding boxes in format suitable for SAM prompts.
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            List of (x_min, y_min, x_max, y_max) tuples
        """
        return [d["bbox"] for d in detections]
    
    def filter_by_class(
        self,
        detections: List[Dict],
        class_ids: List[int] = None,
        class_names: List[str] = None
    ) -> List[Dict]:
        """
        Filter detections by class.
        
        Args:
            detections: List of detections
            class_ids: List of class IDs to keep
            class_names: List of class names to keep
        
        Returns:
            Filtered list of detections
        """
        if class_ids is None and class_names is None:
            return detections
        
        filtered = []
        for det in detections:
            if class_ids and det["class_id"] in class_ids:
                filtered.append(det)
            elif class_names and det["class_name"] in class_names:
                filtered.append(det)
        
        return filtered
    
    def filter_by_confidence(
        self,
        detections: List[Dict],
        min_confidence: float
    ) -> List[Dict]:
        """
        Filter detections by minimum confidence.
        
        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold
        
        Returns:
            Filtered list of detections
        """
        return [d for d in detections if d["confidence"] >= min_confidence]
    
    def update_thresholds(
        self,
        confidence_threshold: float = None,
        nms_iou_threshold: float = None
    ):
        """Update detection thresholds."""
        if confidence_threshold is not None:
            self.conf_threshold = confidence_threshold
        if nms_iou_threshold is not None:
            self.nms_iou = nms_iou_threshold
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "model_type": str(type(self.model)),
            "confidence_threshold": self.conf_threshold,
            "nms_iou_threshold": self.nms_iou,
            "device": self.device,
            "class_names": self.class_names
        }
