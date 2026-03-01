"""
Pixel Area Computation and Physical Conversion.

Computes:
- Exact pixel area from masks
- Physical area in mm² using calibration factor

Properties:
- Exact
- Stable
- Fully deterministic
"""

from typing import Dict, List, Tuple

import numpy as np


# Default calibration factor
DEFAULT_MM2_PER_PIXEL = 0.03


def compute_pixel_area(mask: np.ndarray) -> int:
    """
    Compute exact pixel area of defect.
    
    For each defect instance:
    pixel_area = count(mask == 1)
    
    Properties:
    - Exact
    - Stable
    - Fully deterministic
    
    Args:
        mask: Binary mask with values 0 or 1
    
    Returns:
        Pixel count (area in pixels)
    
    Example:
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[10:30, 10:40] = 1  # 20 x 30 = 600 pixels
        >>> area = compute_pixel_area(mask)
        >>> area
        600
    """
    return int(np.sum(mask > 0))


def pixel_to_mm2(
    pixel_area: int,
    mm2_per_pixel: float = DEFAULT_MM2_PER_PIXEL
) -> float:
    """
    Convert pixel area to physical area in mm².
    
    Formula: area_mm2 = pixel_area × mm2_per_pixel
    
    Args:
        pixel_area: Area in pixels
        mm2_per_pixel: Calibration factor (default: 0.03)
    
    Returns:
        Physical area in mm²
    
    Example:
        >>> pixel_area = 1000
        >>> mm2 = pixel_to_mm2(pixel_area, mm2_per_pixel=0.03)
        >>> mm2
        30.0
    """
    return pixel_area * mm2_per_pixel


def mm2_to_pixel(
    mm2_area: float,
    mm2_per_pixel: float = DEFAULT_MM2_PER_PIXEL
) -> float:
    """
    Convert physical area to pixel area.
    
    Args:
        mm2_area: Area in mm²
        mm2_per_pixel: Calibration factor
    
    Returns:
        Area in pixels (float)
    """
    return mm2_area / mm2_per_pixel


def compute_mask_area(
    mask: np.ndarray,
    mm2_per_pixel: float = DEFAULT_MM2_PER_PIXEL
) -> Dict:
    """
    Compute complete area statistics for a mask.
    
    Args:
        mask: Binary mask
        mm2_per_pixel: Calibration factor
    
    Returns:
        Dictionary with area statistics
    """
    pixel_area = compute_pixel_area(mask)
    mm2_area = pixel_to_mm2(pixel_area, mm2_per_pixel)
    
    # Get bounding box for reference
    rows, cols = np.where(mask > 0)
    
    if len(rows) == 0:
        return {
            "pixel_area": 0,
            "mm2_area": 0.0,
            "bbox_area_pixels": 0,
            "bbox_area_mm2": 0.0,
            "fill_ratio": 0.0
        }
    
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
    
    return {
        "pixel_area": pixel_area,
        "mm2_area": mm2_area,
        "bbox_area_pixels": int(bbox_area),
        "bbox_area_mm2": pixel_to_mm2(bbox_area, mm2_per_pixel),
        "fill_ratio": pixel_area / max(bbox_area, 1)
    }


def compute_batch_areas(
    masks: List[np.ndarray],
    mm2_per_pixel: float = DEFAULT_MM2_PER_PIXEL
) -> List[Dict]:
    """
    Compute areas for multiple masks.
    
    Args:
        masks: List of binary masks
        mm2_per_pixel: Calibration factor
    
    Returns:
        List of area dictionaries
    """
    return [compute_mask_area(mask, mm2_per_pixel) for mask in masks]


def get_area_statistics(areas_mm2: List[float]) -> Dict:
    """
    Compute statistics over a collection of areas.
    
    Args:
        areas_mm2: List of areas in mm²
    
    Returns:
        Dictionary with statistical measures
    """
    if not areas_mm2:
        return {
            "count": 0,
            "total_area": 0.0,
            "mean_area": 0.0,
            "median_area": 0.0,
            "std_area": 0.0,
            "min_area": 0.0,
            "max_area": 0.0
        }
    
    areas = np.array(areas_mm2)
    
    return {
        "count": len(areas),
        "total_area": float(np.sum(areas)),
        "mean_area": float(np.mean(areas)),
        "median_area": float(np.median(areas)),
        "std_area": float(np.std(areas)),
        "min_area": float(np.min(areas)),
        "max_area": float(np.max(areas))
    }


def compute_area_distribution(
    areas_mm2: List[float],
    bins: int = 10
) -> Dict:
    """
    Compute area distribution histogram.
    
    Args:
        areas_mm2: List of areas in mm²
        bins: Number of histogram bins
    
    Returns:
        Dictionary with histogram data
    """
    if not areas_mm2:
        return {"bins": [], "counts": []}
    
    counts, bin_edges = np.histogram(areas_mm2, bins=bins)
    
    return {
        "bins": bin_edges.tolist(),
        "counts": counts.tolist(),
        "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
    }


def calibrate_mm2_per_pixel(
    known_width_mm: float,
    known_width_pixels: int
) -> float:
    """
    Calculate mm²/pixel calibration factor from known measurement.
    
    Args:
        known_width_mm: Known width in millimeters
        known_width_pixels: Same width measured in pixels
    
    Returns:
        Calibration factor (mm² per pixel)
    """
    mm_per_pixel = known_width_mm / known_width_pixels
    mm2_per_pixel = mm_per_pixel ** 2
    
    return mm2_per_pixel


def area_error(
    predicted_mm2: float,
    ground_truth_mm2: float
) -> Dict:
    """
    Compute area error between prediction and ground truth.
    
    Args:
        predicted_mm2: Predicted area in mm²
        ground_truth_mm2: Ground truth area in mm²
    
    Returns:
        Dictionary with error metrics
    """
    absolute_error = abs(predicted_mm2 - ground_truth_mm2)
    
    if ground_truth_mm2 > 0:
        percentage_error = (absolute_error / ground_truth_mm2) * 100
    else:
        percentage_error = float('inf') if predicted_mm2 > 0 else 0.0
    
    return {
        "predicted_mm2": predicted_mm2,
        "ground_truth_mm2": ground_truth_mm2,
        "absolute_error_mm2": absolute_error,
        "percentage_error": percentage_error
    }


def batch_area_errors(
    predictions: List[Dict],
    ground_truths: List[Dict],
    mm2_per_pixel: float = DEFAULT_MM2_PER_PIXEL
) -> Dict:
    """
    Compute area errors across multiple predictions.
    
    Args:
        predictions: List of prediction dicts with 'mask' key
        ground_truths: List of ground truth dicts with 'mask' key
        mm2_per_pixel: Calibration factor
    
    Returns:
        Dictionary with error statistics
    """
    errors = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_area = compute_mask_area(pred["mask"], mm2_per_pixel)["mm2_area"]
        gt_area = compute_mask_area(gt["mask"], mm2_per_pixel)["mm2_area"]
        
        error = area_error(pred_area, gt_area)
        errors.append(error)
    
    # Compute summary statistics
    percentage_errors = [e["percentage_error"] for e in errors if e["percentage_error"] != float('inf')]
    
    return {
        "individual_errors": errors,
        "mean_percentage_error": float(np.mean(percentage_errors)) if percentage_errors else 0.0,
        "median_percentage_error": float(np.median(percentage_errors)) if percentage_errors else 0.0,
        "max_percentage_error": float(np.max(percentage_errors)) if percentage_errors else 0.0
    }
