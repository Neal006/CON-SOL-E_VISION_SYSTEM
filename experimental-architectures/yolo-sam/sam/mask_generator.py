"""
Mask Generation Pipeline.

Handles the complete mask generation workflow:
- Processing YOLO detections through SAM
- Batch mask generation
- Mask post-processing and refinement
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .segmentor import SAMSegmentor
from .prompt_builder import yolo_detections_to_prompts


class MaskGenerator:
    """
    High-level mask generation pipeline.
    
    Combines YOLO detections with SAM segmentation to produce
    refined instance masks for each detected defect.
    
    Features:
    - Automatic grayscale handling
    - Batch processing
    - Mask quality filtering
    - Optional mask refinement
    """
    
    def __init__(
        self,
        sam_model_type: str = "vit_b",
        sam_checkpoint: str = None,
        device: str = "cuda",
        mask_threshold: float = 0.5,
        min_mask_area: int = 100
    ):
        """
        Initialize the mask generator.
        
        Args:
            sam_model_type: SAM model variant
            sam_checkpoint: Path to SAM checkpoint
            device: Computation device
            mask_threshold: Threshold for binary mask
            min_mask_area: Minimum mask area in pixels
        """
        self.mask_threshold = mask_threshold
        self.min_mask_area = min_mask_area
        
        self.segmentor = SAMSegmentor(
            model_type=sam_model_type,
            checkpoint_path=sam_checkpoint,
            device=device
        )
    
    def generate_masks(
        self,
        image: np.ndarray,
        detections: List[Dict],
        refine_masks: bool = True
    ) -> List[Dict]:
        """
        Generate instance masks from YOLO detections.
        
        Args:
            image: Input image (grayscale or RGB)
            detections: List of YOLO detections with 'bbox' key
            refine_masks: Whether to apply post-processing refinement
        
        Returns:
            List of enriched detections with masks
        """
        if not detections:
            return []
        
        # Set image in SAM
        self.segmentor.set_image(image)
        
        results = []
        
        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                continue
            
            # Generate mask
            sam_result = self.segmentor.predict(bbox)
            mask = sam_result["mask"]
            score = sam_result["score"]
            
            # Apply threshold
            binary_mask = (mask > self.mask_threshold).astype(np.uint8)
            
            # Check minimum area
            mask_area = np.sum(binary_mask)
            if mask_area < self.min_mask_area:
                continue
            
            # Refine mask if requested
            if refine_masks:
                binary_mask = self._refine_mask(binary_mask)
            
            # Create result
            result = {
                **det,
                "mask": binary_mask,
                "sam_score": score,
                "mask_area": int(np.sum(binary_mask))
            }
            
            results.append(result)
        
        return results
    
    def _refine_mask(
        self,
        mask: np.ndarray,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Refine binary mask using morphological operations.
        
        Args:
            mask: Binary mask
            kernel_size: Size of morphological kernel
        
        Returns:
            Refined mask
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        
        # Close small holes
        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        
        return refined
    
    def generate_from_directory(
        self,
        image_dir: Union[str, Path],
        detections_per_image: Dict[str, List[Dict]],
        output_dir: Union[str, Path] = None,
        save_masks: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Generate masks for multiple images.
        
        Args:
            image_dir: Directory containing images
            detections_per_image: Dictionary mapping image names to detections
            output_dir: Optional directory to save masks
            save_masks: Whether to save mask images
        
        Returns:
            Dictionary mapping image names to results
        """
        image_dir = Path(image_dir)
        results = {}
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for image_name, detections in detections_per_image.items():
            # Find image file
            image_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = image_dir / f"{Path(image_name).stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path is None:
                print(f"Warning: Image not found: {image_name}")
                continue
            
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            
            # Generate masks
            image_results = self.generate_masks(image, detections)
            results[image_name] = image_results
            
            # Save masks if requested
            if save_masks and output_dir:
                self._save_masks(image_results, image_name, output_dir)
        
        return results
    
    def _save_masks(
        self,
        results: List[Dict],
        image_name: str,
        output_dir: Path
    ):
        """Save individual masks to files."""
        for i, result in enumerate(results):
            mask = result.get("mask")
            if mask is None:
                continue
            
            class_name = result.get("class_name", "unknown")
            mask_name = f"{Path(image_name).stem}_{class_name}_{i}.png"
            mask_path = output_dir / mask_name
            
            # Save as binary image (0 or 255)
            cv2.imwrite(str(mask_path), mask * 255)
    
    def create_combined_mask(
        self,
        results: List[Dict],
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create a single mask combining all instances.
        
        Args:
            results: List of result dictionaries with masks
            image_shape: (height, width) of output mask
        
        Returns:
            Combined mask with instance IDs (0 = background)
        """
        combined = np.zeros(image_shape, dtype=np.int32)
        
        for i, result in enumerate(results, start=1):
            mask = result.get("mask")
            if mask is None:
                continue
            
            # Resize mask if needed
            if mask.shape != image_shape:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Add to combined (later instances override earlier)
            combined[mask > 0] = i
        
        return combined
    
    def create_class_mask(
        self,
        results: List[Dict],
        image_shape: Tuple[int, int],
        class_mapping: Dict[str, int] = None
    ) -> np.ndarray:
        """
        Create a mask with class IDs instead of instance IDs.
        
        Args:
            results: List of result dictionaries with masks and class info
            image_shape: (height, width) of output mask
            class_mapping: Optional class name to ID mapping
        
        Returns:
            Class mask (0 = background, 1+ = class IDs)
        """
        class_mapping = class_mapping or {
            "Scratch": 1,
            "Dust": 2,
            "Rundown": 3
        }
        
        combined = np.zeros(image_shape, dtype=np.uint8)
        
        for result in results:
            mask = result.get("mask")
            class_name = result.get("class_name")
            
            if mask is None or class_name is None:
                continue
            
            class_id = class_mapping.get(class_name, 0)
            
            # Resize mask if needed
            if mask.shape != image_shape:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            
            combined[mask > 0] = class_id
        
        return combined
    
    def visualize_masks(
        self,
        image: np.ndarray,
        results: List[Dict],
        colors: Dict[str, Tuple[int, int, int]] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create visualization of masks overlaid on image.
        
        Args:
            image: Input image
            results: List of result dictionaries with masks
            colors: Class name to BGR color mapping
            alpha: Transparency for overlay
        
        Returns:
            Visualization image
        """
        colors = colors or {
            "Scratch": (0, 255, 0),    # Green
            "Dust": (255, 0, 0),       # Blue
            "Rundown": (0, 0, 255)     # Red
        }
        
        # Ensure 3-channel image
        if image.ndim == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Create overlay
        overlay = vis_image.copy()
        
        for result in results:
            mask = result.get("mask")
            class_name = result.get("class_name", "unknown")
            
            if mask is None:
                continue
            
            color = colors.get(class_name, (128, 128, 128))
            
            # Resize mask if needed
            if mask.shape[:2] != vis_image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (vis_image.shape[1], vis_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Apply color
            overlay[mask > 0] = color
        
        # Blend
        vis_image = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)
        
        # Add bboxes and labels
        for result in results:
            bbox = result.get("bbox")
            class_name = result.get("class_name", "unknown")
            confidence = result.get("confidence", 0)
            
            if bbox:
                x_min, y_min, x_max, y_max = [int(v) for v in bbox]
                color = colors.get(class_name, (128, 128, 128))
                
                cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
                
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(
                    vis_image, label,
                    (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
        
        return vis_image
