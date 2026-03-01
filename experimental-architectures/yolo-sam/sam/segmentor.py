"""
SAM (Segment Anything Model) Segmentor Wrapper.

Provides high-level interface for SAM-based instance segmentation:
- Automatic model loading
- Grayscale image support (auto-expanded to 3 channels)
- Bounding box prompt-based segmentation
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class SAMSegmentor:
    """
    SAM Segmentor wrapper for defect instance segmentation.
    
    Uses bounding box prompts from YOLO detections to generate
    precise instance masks. Auto-loads settings from config.py.
    
    Attributes:
        model: Loaded SAM model
        predictor: SAM predictor for inference
        device: Computation device
    """
    
    def __init__(
        self,
        model_type: str = None,
        checkpoint_path: Optional[str] = None,
        device: str = None,
        config = None
    ):
        """
        Initialize the SAM segmentor.
        
        Args:
            model_type: SAM model variant ('vit_h', 'vit_l', 'vit_b') (uses config if None)
            checkpoint_path: Path to SAM checkpoint file (uses config if None)
            device: Device for inference (uses config if None)
            config: Optional Config object (auto-loads if None)
        """
        # Load config if not provided
        if config is None:
            from config import get_config
            config = get_config()
        
        self.model_type = model_type if model_type is not None else config.sam.model_type
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else config.sam.checkpoint_path
        self.device = device if device is not None else config.sam.device
        
        self.sam = None
        self.predictor = None
        self._image_set = False
        self._current_image_shape = None
        
        self._load_model()
    
    def _load_model(self):
        """Load SAM model and create predictor."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything package not found. "
                "Install with: pip install segment-anything"
            )
        
        # Get checkpoint path
        if self.checkpoint_path is None:
            # Default checkpoint names
            checkpoint_map = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_b": "sam_vit_b_01ec64.pth"
            }
            self.checkpoint_path = checkpoint_map.get(self.model_type)
        
        if self.checkpoint_path and not Path(self.checkpoint_path).exists():
            print(f"Warning: SAM checkpoint not found at {self.checkpoint_path}")
            print("Please download from: https://github.com/facebookresearch/segment-anything")
            # Continue anyway - will fail later if model is used
        
        try:
            # Build model
            self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.sam.to(self.device)
            
            # Create predictor
            self.predictor = SamPredictor(self.sam)
        except Exception as e:
            print(f"Warning: Failed to load SAM model: {e}")
            self.sam = None
            self.predictor = None
    
    def set_image(self, image: np.ndarray):
        """
        Set the image for segmentation.
        
        Preprocesses the image and computes image embeddings.
        Must be called before predict() for efficiency.
        
        Args:
            image: Input image (H, W) grayscale or (H, W, 3) RGB
        """
        if self.predictor is None:
            raise RuntimeError("SAM model not loaded. Check checkpoint path.")
        
        # Expand grayscale to 3 channels
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Set image
        self.predictor.set_image(image)
        self._image_set = True
        self._current_image_shape = image.shape[:2]
    
    def predict(
        self,
        bbox: Tuple[float, float, float, float],
        multimask_output: bool = False
    ) -> Dict:
        """
        Predict mask for a single bounding box.
        
        Args:
            bbox: Bounding box (x_min, y_min, x_max, y_max) in pixel coordinates
            multimask_output: Whether to return multiple mask options
        
        Returns:
            Dictionary containing:
                - mask: Binary mask of shape (H, W)
                - score: IoU prediction score
                - masks: All predicted masks (if multimask_output=True)
                - scores: All IoU scores (if multimask_output=True)
        """
        if not self._image_set:
            raise RuntimeError("Image not set. Call set_image() first.")
        
        # Convert bbox to numpy array
        input_box = np.array(bbox)
        
        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=multimask_output
        )
        
        # Select best mask
        if multimask_output:
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
        else:
            best_mask = masks[0]
            best_score = scores[0]
        
        result = {
            "mask": best_mask,
            "score": float(best_score)
        }
        
        if multimask_output:
            result["masks"] = masks
            result["scores"] = scores.tolist()
        
        return result
    
    def predict_batch(
        self,
        bboxes: List[Tuple[float, float, float, float]],
        multimask_output: bool = False
    ) -> List[Dict]:
        """
        Predict masks for multiple bounding boxes.
        
        Note: Image must be set before calling this method.
        
        Args:
            bboxes: List of bounding boxes
            multimask_output: Whether to return multiple mask options
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for bbox in bboxes:
            result = self.predict(bbox, multimask_output=multimask_output)
            result["bbox"] = bbox
            results.append(result)
        
        return results
    
    def segment_image(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[float, float, float, float]],
        class_ids: List[int] = None,
        class_names: List[str] = None
    ) -> List[Dict]:
        """
        Complete segmentation pipeline for an image.
        
        Combines set_image and predict_batch for convenience.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes
            class_ids: Optional class IDs for each bbox
            class_names: Optional class names for each bbox
        
        Returns:
            List of segmentation results
        """
        self.set_image(image)
        results = self.predict_batch(bboxes)
        
        # Add class information if provided
        if class_ids:
            for i, result in enumerate(results):
                if i < len(class_ids):
                    result["class_id"] = class_ids[i]
        
        if class_names:
            for i, result in enumerate(results):
                if i < len(class_names):
                    result["class_name"] = class_names[i]
        
        return results
    
    def segment_from_yolo_detections(
        self,
        image: np.ndarray,
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Segment from YOLO detection outputs.
        
        Takes YOLO detector output format and returns refined masks.
        
        Args:
            image: Input image
            detections: List of YOLO detections with 'bbox', 'class_id', 'class_name'
        
        Returns:
            List of enriched detections with SAM masks
        """
        if not detections:
            return []
        
        self.set_image(image)
        
        results = []
        for det in detections:
            sam_result = self.predict(det["bbox"])
            
            # Merge YOLO detection with SAM result
            result = {
                **det,
                "sam_mask": sam_result["mask"],
                "sam_score": sam_result["score"]
            }
            
            results.append(result)
        
        return results
    
    def reset_image(self):
        """Reset the image state."""
        self._image_set = False
        self._current_image_shape = None
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "model_type": self.model_type,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "model_loaded": self.sam is not None
        }
