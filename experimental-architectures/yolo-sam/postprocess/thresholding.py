"""
Mask Thresholding.

Converts probability masks to binary masks.
Supports fixed and adaptive thresholding strategies.
"""

from typing import Dict, Union

import cv2
import numpy as np


def apply_threshold(
    mask: np.ndarray,
    threshold: float = 0.5,
    output_dtype: np.dtype = np.uint8
) -> np.ndarray:
    """
    Apply fixed threshold to convert probability mask to binary.
    
    Args:
        mask: Probability mask with values in [0, 1]
        threshold: Threshold value (default: 0.5)
        output_dtype: Output data type (default: uint8)
    
    Returns:
        Binary mask with values 0 or 1
    
    Example:
        >>> prob_mask = np.array([[0.3, 0.7], [0.8, 0.2]])
        >>> binary = apply_threshold(prob_mask, threshold=0.5)
        >>> binary
        array([[0, 1],
               [1, 0]], dtype=uint8)
    """
    # Ensure mask is in correct range
    if mask.max() > 1.0:
        mask = mask / 255.0
    
    binary = (mask > threshold).astype(output_dtype)
    return binary


def adaptive_threshold(
    mask: np.ndarray,
    class_thresholds: Dict[str, float] = None,
    class_name: str = None,
    default_threshold: float = 0.5
) -> np.ndarray:
    """
    Apply class-specific adaptive threshold.
    
    Different defect types may require different thresholds:
    - Scratch: Fine details, lower threshold
    - Dust: Clear boundaries, standard threshold
    - Rundown: Diffuse regions, higher threshold
    
    Args:
        mask: Probability mask
        class_thresholds: Dictionary of class-specific thresholds
        class_name: Class name for this mask
        default_threshold: Default threshold if class not found
    
    Returns:
        Binary mask
    """
    class_thresholds = class_thresholds or {
        "Scratch": 0.4,
        "Dust": 0.5,
        "Rundown": 0.55
    }
    
    threshold = class_thresholds.get(class_name, default_threshold)
    return apply_threshold(mask, threshold)


def otsu_threshold(mask: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's automatic thresholding.
    
    Useful when optimal threshold varies per image.
    
    Args:
        mask: Probability mask
    
    Returns:
        Binary mask
    """
    # Convert to uint8 for OpenCV
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Apply Otsu's threshold
    _, binary = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return (binary > 0).astype(np.uint8)


def hysteresis_threshold(
    mask: np.ndarray,
    low_threshold: float = 0.3,
    high_threshold: float = 0.6
) -> np.ndarray:
    """
    Apply hysteresis thresholding.
    
    Keeps pixels above low_threshold only if connected
    to pixels above high_threshold.
    
    Args:
        mask: Probability mask
        low_threshold: Low threshold
        high_threshold: High threshold
    
    Returns:
        Binary mask
    """
    # Create high and low masks
    high_mask = (mask > high_threshold).astype(np.uint8)
    low_mask = (mask > low_threshold).astype(np.uint8)
    
    # Find connected components in high mask
    num_labels, labels = cv2.connectedComponents(high_mask)
    
    # Expand high regions using low mask
    result = np.zeros_like(mask, dtype=np.uint8)
    
    for label in range(1, num_labels):
        # Get region in high mask
        region = (labels == label)
        
        # Expand using low mask
        expanded = cv2.dilate(region.astype(np.uint8), None, iterations=10)
        connected = expanded & low_mask
        
        # Add connected region to result
        result[connected > 0] = 1
    
    return result


def apply_morphological_cleanup(
    mask: np.ndarray,
    close_size: int = 5,
    open_size: int = 3
) -> np.ndarray:
    """
    Clean up binary mask using morphological operations.
    
    Args:
        mask: Binary mask
        close_size: Kernel size for closing (fills holes)
        open_size: Kernel size for opening (removes noise)
    
    Returns:
        Cleaned mask
    """
    # Close: fills small holes
    if close_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Open: removes small noise
    if open_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def smooth_mask_boundary(
    mask: np.ndarray,
    blur_size: int = 5,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Smooth mask boundaries using Gaussian blur and re-thresholding.
    
    Args:
        mask: Binary mask
        blur_size: Gaussian blur kernel size
        threshold: Re-thresholding value
    
    Returns:
        Smoothed mask
    """
    # Blur
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
    
    # Re-threshold
    smoothed = (blurred > threshold).astype(np.uint8)
    
    return smoothed
