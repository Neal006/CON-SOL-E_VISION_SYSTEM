"""
Defect Instance Separator for SPGS-Net
=======================================
Post-processing for individual defect detection.

Section 6 of Architecture: Post-Processing & Defect Instance Separation
- Predicted segmentation mask thresholded using adaptive strategy
- Morphological operations applied to remove spurious noise
- Connected component analysis to separate individual defect instances
- For each instance:
  * Bounding box coordinates computed
  * Pixel-accurate masks retained
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from scipy import ndimage
from skimage import measure

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PostProcessingConfig, OutputConfig


class InstanceSeparator:
    """
    Separates segmentation mask into individual defect instances.
    
    Section 6: Post-processing & defect instance separation.
    Applies thresholding, morphological ops, and connected components.
    """
    
    def __init__(
        self,
        confidence_threshold: float = None,
        morphology_kernel_size: int = None,
        erosion_iterations: int = None,
        dilation_iterations: int = None,
        min_defect_area: int = None,
        connectivity: int = None
    ):
        """
        Initialize instance separator.
        
        Section 6: Configuration for post-processing pipeline.
        
        Args:
            confidence_threshold: Threshold for binary mask
            morphology_kernel_size: Kernel size for morphological ops
            erosion_iterations: Number of erosion iterations
            dilation_iterations: Number of dilation iterations
            min_defect_area: Minimum pixel area for valid defect
            connectivity: 4 or 8 connectivity for connected components
        """
        self.confidence_threshold = confidence_threshold or PostProcessingConfig.CONFIDENCE_THRESHOLD
        self.morphology_kernel_size = morphology_kernel_size or PostProcessingConfig.MORPHOLOGY_KERNEL_SIZE
        self.erosion_iterations = erosion_iterations or PostProcessingConfig.EROSION_ITERATIONS
        self.dilation_iterations = dilation_iterations or PostProcessingConfig.DILATION_ITERATIONS
        self.min_defect_area = min_defect_area or PostProcessingConfig.MIN_DEFECT_AREA_PIXELS
        self.connectivity = connectivity or PostProcessingConfig.CONNECTIVITY
        
        # Section 6: Create morphological kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )
    
    def threshold_mask(
        self,
        probabilities: np.ndarray,
        class_id: int
    ) -> np.ndarray:
        """
        Apply adaptive thresholding to probability map.
        
        Section 6: Threshold predicted mask using adaptive strategy.
        
        Args:
            probabilities: (H, W, num_classes) or (num_classes, H, W)
            class_id: Class to threshold
            
        Returns:
            binary_mask: (H, W) binary mask
        """
        # Handle different input formats
        if len(probabilities.shape) == 3:
            if probabilities.shape[0] < probabilities.shape[2]:
                # (C, H, W) format
                class_prob = probabilities[class_id]
            else:
                # (H, W, C) format
                class_prob = probabilities[:, :, class_id]
        else:
            class_prob = probabilities
        
        # Section 6: Adaptive thresholding
        binary_mask = (class_prob >= self.confidence_threshold).astype(np.uint8)
        
        return binary_mask
    
    def apply_morphology(
        self,
        binary_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply morphological operations to clean mask.
        
        Section 6: Morphological operations to remove spurious noise.
        Uses erosion followed by dilation (opening-like effect).
        
        Args:
            binary_mask: (H, W) binary mask
            
        Returns:
            cleaned_mask: (H, W) morphologically cleaned mask
        """
        # Section 6: Erosion to remove small noise
        if self.erosion_iterations > 0:
            eroded = cv2.erode(
                binary_mask, self.kernel,
                iterations=self.erosion_iterations
            )
        else:
            eroded = binary_mask
        
        # Section 6: Dilation to restore object size
        if self.dilation_iterations > 0:
            dilated = cv2.dilate(
                eroded, self.kernel,
                iterations=self.dilation_iterations
            )
        else:
            dilated = eroded
        
        return dilated
    
    def find_connected_components(
        self,
        binary_mask: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Find connected components in binary mask.
        
        Section 6: Connected component analysis for instance separation.
        
        Args:
            binary_mask: (H, W) binary mask
            
        Returns:
            labels: (H, W) labeled image where each component has unique ID
            num_components: Number of components found
        """
        # Section 6: Connected component analysis
        connectivity = 8 if self.connectivity == 8 else 4
        
        # Use OpenCV for connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=connectivity
        )
        
        # num_labels includes background (0), so actual components = num_labels - 1
        return labels, num_labels - 1
    
    def extract_instance_info(
        self,
        labels: np.ndarray,
        class_id: int,
        probabilities: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Extract information for each instance.
        
        Section 6: For each instance:
        - Bounding box coordinates computed
        - Pixel-accurate masks retained
        
        Args:
            labels: (H, W) labeled component image
            class_id: Original class ID
            probabilities: Optional probability map for confidence
            
        Returns:
            List of instance dictionaries
        """
        instances = []
        unique_labels = np.unique(labels)
        
        for label_id in unique_labels:
            if label_id == 0:  # Skip background
                continue
            
            # Section 6: Extract instance mask
            instance_mask = (labels == label_id).astype(np.uint8)
            
            # Check minimum area
            area_pixels = instance_mask.sum()
            if area_pixels < self.min_defect_area:
                continue
            
            # Section 6: Compute bounding box
            coords = np.where(instance_mask)
            y1, y2 = coords[0].min(), coords[0].max()
            x1, x2 = coords[1].min(), coords[1].max()
            
            # Compute confidence (mean probability in region)
            confidence = 1.0
            if probabilities is not None:
                if len(probabilities.shape) == 3:
                    if probabilities.shape[0] < probabilities.shape[2]:
                        class_prob = probabilities[class_id]
                    else:
                        class_prob = probabilities[:, :, class_id]
                else:
                    class_prob = probabilities
                confidence = float(class_prob[instance_mask == 1].mean())
            
            # Section 6: Store instance info
            instances.append({
                'class_id': class_id,
                'class_name': OutputConfig.CLASS_NAMES.get(class_id, "Unknown"),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'mask': instance_mask,
                'area_pixels': int(area_pixels),
                'confidence': confidence,
                'centroid': (int((x1 + x2) / 2), int((y1 + y2) / 2))
            })
        
        return instances
    
    def process(
        self,
        segmentation_mask: Union[np.ndarray, torch.Tensor],
        probabilities: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> List[Dict]:
        """
        Full instance separation pipeline.
        
        Section 6: Complete post-processing workflow.
        
        Args:
            segmentation_mask: (H, W) class predictions
            probabilities: Optional (C, H, W) or (H, W, C) probabilities
            
        Returns:
            List of all detected instances with metadata
        """
        # Convert tensors to numpy
        if isinstance(segmentation_mask, torch.Tensor):
            segmentation_mask = segmentation_mask.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        all_instances = []
        
        # Process each defect class (skip background = 0)
        for class_id in range(1, 4):  # Classes 1, 2, 3 (Dust, RunDown, Scratch)
            # Section 6: Get binary mask for this class
            class_mask = (segmentation_mask == class_id).astype(np.uint8)
            
            if class_mask.sum() == 0:
                continue
            
            # Section 6: Apply morphological operations
            cleaned_mask = self.apply_morphology(class_mask)
            
            if cleaned_mask.sum() == 0:
                continue
            
            # Section 6: Find connected components
            labels, num_components = self.find_connected_components(cleaned_mask)
            
            if num_components == 0:
                continue
            
            # Section 6: Extract instance information
            instances = self.extract_instance_info(labels, class_id, probabilities)
            all_instances.extend(instances)
        
        return all_instances


def separate_defect_instances(
    segmentation_mask: Union[np.ndarray, torch.Tensor],
    probabilities: Optional[Union[np.ndarray, torch.Tensor]] = None,
    **kwargs
) -> List[Dict]:
    """
    Convenience function for instance separation.
    
    Section 6: Full post-processing pipeline.
    
    Args:
        segmentation_mask: (H, W) or (B, H, W) class predictions
        probabilities: Optional class probabilities
        **kwargs: Additional arguments for InstanceSeparator
        
    Returns:
        List of detected defect instances
    """
    separator = InstanceSeparator(**kwargs)
    
    # Handle batch dimension
    if isinstance(segmentation_mask, torch.Tensor):
        segmentation_mask = segmentation_mask.cpu().numpy()
    
    if len(segmentation_mask.shape) == 3:
        # Batch processing
        all_results = []
        for i in range(segmentation_mask.shape[0]):
            mask_i = segmentation_mask[i]
            prob_i = probabilities[i] if probabilities is not None else None
            instances = separator.process(mask_i, prob_i)
            all_results.append(instances)
        return all_results
    
    return separator.process(segmentation_mask, probabilities)


def create_instance_mask(
    instances: List[Dict],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Create combined instance mask from separated instances.
    
    Args:
        instances: List of instance dictionaries
        image_shape: (H, W) output shape
        
    Returns:
        instance_mask: (H, W) where each instance has unique ID
    """
    instance_mask = np.zeros(image_shape, dtype=np.int32)
    
    for idx, inst in enumerate(instances, start=1):
        mask = inst.get('mask')
        if mask is not None:
            instance_mask[mask == 1] = idx
    
    return instance_mask


if __name__ == "__main__":
    # Test instance separator
    print("Testing Defect Instance Separator...")
    print("=" * 60)
    
    # Create dummy segmentation mask with multiple defects
    height, width = 256, 256
    
    # Create segmentation mask with multiple classes and instances
    seg_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Add some defects
    # Dust patches (class 1)
    seg_mask[50:80, 60:100] = 1
    seg_mask[100:130, 150:190] = 1
    
    # RunDown (class 2) - elongated
    seg_mask[150:180, 40:45] = 2
    seg_mask[150:180, 200:210] = 2
    
    # Scratch (class 3)
    seg_mask[30:35, 120:200] = 3
    
    # Add some noise
    noise_mask = np.random.random((height, width)) < 0.001
    seg_mask[noise_mask] = np.random.randint(1, 4, noise_mask.sum())
    
    print(f"Input mask shape: {seg_mask.shape}")
    print(f"Unique classes: {np.unique(seg_mask)}")
    
    # Create dummy probabilities
    probs = np.zeros((4, height, width), dtype=np.float32)
    for c in range(4):
        probs[c][seg_mask == c] = 0.9
    probs[0] = 1.0 - probs[1:].sum(axis=0)  # Background prob
    
    # Test instance separation
    separator = InstanceSeparator()
    instances = separator.process(seg_mask, probs)
    
    print(f"\n[Section 6] Found {len(instances)} defect instances:")
    for idx, inst in enumerate(instances):
        print(f"  Instance {idx + 1}:")
        print(f"    Class: {inst['class_name']} (ID: {inst['class_id']})")
        print(f"    BBox: {inst['bbox']}")
        print(f"    Area: {inst['area_pixels']} pixels")
        print(f"    Confidence: {inst['confidence']:.3f}")
    
    # Test convenience function
    instances_batch = separate_defect_instances(
        np.stack([seg_mask, seg_mask]),  # Batch of 2
        np.stack([probs, probs])
    )
    print(f"\n[Section 6] Batch processing: {len(instances_batch)} images processed")
    print(f"  Total instances: {sum(len(i) for i in instances_batch)}")
