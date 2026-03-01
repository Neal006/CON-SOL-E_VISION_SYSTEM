"""
Physical Metrics.

Metrics for measuring physical defect properties:
- Area error (%)
- Median mm² deviation
- Per-class measurement accuracy
"""

from typing import Dict, List

import numpy as np


def compute_pixel_area(mask: np.ndarray) -> int:
    """Compute pixel area of mask."""
    return int(np.sum(mask > 0))


def compute_area_error(
    pred_masks: List[Dict],
    gt_masks: List[Dict],
    mm2_per_pixel: float = 0.03
) -> Dict:
    """
    Compute area error between predictions and ground truths.
    
    Area Error (%) = |Predicted Area - GT Area| / GT Area × 100
    
    Args:
        pred_masks: List of predictions with 'mask' key
        gt_masks: List of ground truths with 'mask' key
        mm2_per_pixel: Calibration factor
    
    Returns:
        Dictionary with area error statistics
    """
    errors = []
    absolute_errors_mm2 = []
    
    for pred, gt in zip(pred_masks, gt_masks):
        pred_mask = pred.get("mask")
        gt_mask = gt.get("mask")
        
        if pred_mask is None or gt_mask is None:
            continue
        
        pred_area_px = compute_pixel_area(pred_mask)
        gt_area_px = compute_pixel_area(gt_mask)
        
        pred_area_mm2 = pred_area_px * mm2_per_pixel
        gt_area_mm2 = gt_area_px * mm2_per_pixel
        
        if gt_area_mm2 > 0:
            error_pct = abs(pred_area_mm2 - gt_area_mm2) / gt_area_mm2 * 100
            errors.append(error_pct)
        
        absolute_errors_mm2.append(abs(pred_area_mm2 - gt_area_mm2))
    
    if not errors:
        return {
            "mean_error_percent": 0.0,
            "median_error_percent": 0.0,
            "max_error_percent": 0.0,
            "mean_absolute_error_mm2": 0.0,
            "count": 0
        }
    
    return {
        "mean_error_percent": float(np.mean(errors)),
        "median_error_percent": float(np.median(errors)),
        "std_error_percent": float(np.std(errors)),
        "max_error_percent": float(np.max(errors)),
        "min_error_percent": float(np.min(errors)),
        "mean_absolute_error_mm2": float(np.mean(absolute_errors_mm2)),
        "median_absolute_error_mm2": float(np.median(absolute_errors_mm2)),
        "count": len(errors)
    }


def compute_median_deviation(
    pred_masks: List[Dict],
    gt_masks: List[Dict],
    mm2_per_pixel: float = 0.03
) -> Dict:
    """
    Compute median mm² deviation between predictions and ground truths.
    
    Args:
        pred_masks: List of predictions with 'mask' key
        gt_masks: List of ground truths with 'mask' key
        mm2_per_pixel: Calibration factor
    
    Returns:
        Dictionary with deviation statistics
    """
    deviations = []
    
    for pred, gt in zip(pred_masks, gt_masks):
        pred_mask = pred.get("mask")
        gt_mask = gt.get("mask")
        
        if pred_mask is None or gt_mask is None:
            continue
        
        pred_area_px = compute_pixel_area(pred_mask)
        gt_area_px = compute_pixel_area(gt_mask)
        
        pred_mm2 = pred_area_px * mm2_per_pixel
        gt_mm2 = gt_area_px * mm2_per_pixel
        
        deviation = pred_mm2 - gt_mm2  # Signed deviation
        deviations.append(deviation)
    
    if not deviations:
        return {
            "median_deviation_mm2": 0.0,
            "mean_deviation_mm2": 0.0,
            "count": 0
        }
    
    deviations = np.array(deviations)
    
    return {
        "median_deviation_mm2": float(np.median(deviations)),
        "mean_deviation_mm2": float(np.mean(deviations)),
        "std_deviation_mm2": float(np.std(deviations)),
        "median_absolute_deviation_mm2": float(np.median(np.abs(deviations))),
        "positive_bias": float(np.sum(deviations > 0)) / len(deviations),  # Over-estimation ratio
        "count": len(deviations)
    }


def compute_per_class_accuracy(
    pred_masks: List[Dict],
    gt_masks: List[Dict],
    class_names: List[str],
    mm2_per_pixel: float = 0.03,
    tolerance_percent: float = 10.0
) -> Dict:
    """
    Compute per-class measurement accuracy.
    
    A measurement is "accurate" if within tolerance_percent of ground truth.
    
    Args:
        pred_masks: List of predictions with 'mask' and 'class_name'
        gt_masks: List of ground truths with 'mask' and 'class_name'
        class_names: List of class names
        mm2_per_pixel: Calibration factor
        tolerance_percent: Acceptable error tolerance
    
    Returns:
        Dictionary with per-class accuracy
    """
    results = {}
    
    for class_name in class_names:
        class_preds = [p for p in pred_masks if p.get("class_name") == class_name]
        class_gts = [g for g in gt_masks if g.get("class_name") == class_name]
        
        if not class_preds or not class_gts:
            results[class_name] = {
                "accuracy": 0.0,
                "count": 0,
                "mean_error_percent": 0.0
            }
            continue
        
        # Match by index (assuming aligned)
        accurate_count = 0
        errors = []
        
        for pred, gt in zip(class_preds, class_gts):
            pred_mask = pred.get("mask")
            gt_mask = gt.get("mask")
            
            if pred_mask is None or gt_mask is None:
                continue
            
            pred_area = compute_pixel_area(pred_mask) * mm2_per_pixel
            gt_area = compute_pixel_area(gt_mask) * mm2_per_pixel
            
            if gt_area > 0:
                error_pct = abs(pred_area - gt_area) / gt_area * 100
                errors.append(error_pct)
                
                if error_pct <= tolerance_percent:
                    accurate_count += 1
        
        results[class_name] = {
            "accuracy": accurate_count / max(len(errors), 1),
            "count": len(errors),
            "mean_error_percent": float(np.mean(errors)) if errors else 0.0,
            "median_error_percent": float(np.median(errors)) if errors else 0.0
        }
    
    return results


def compute_area_distribution_comparison(
    pred_masks: List[Dict],
    gt_masks: List[Dict],
    mm2_per_pixel: float = 0.03,
    num_bins: int = 10
) -> Dict:
    """
    Compare area distributions between predictions and ground truths.
    
    Args:
        pred_masks: List of predictions
        gt_masks: List of ground truths
        mm2_per_pixel: Calibration factor
        num_bins: Number of histogram bins
    
    Returns:
        Dictionary with distribution comparison
    """
    pred_areas = []
    gt_areas = []
    
    for pred in pred_masks:
        mask = pred.get("mask")
        if mask is not None:
            pred_areas.append(compute_pixel_area(mask) * mm2_per_pixel)
    
    for gt in gt_masks:
        mask = gt.get("mask")
        if mask is not None:
            gt_areas.append(compute_pixel_area(mask) * mm2_per_pixel)
    
    if not pred_areas or not gt_areas:
        return {"error": "No valid masks"}
    
    # Compute histograms with same bins
    all_areas = pred_areas + gt_areas
    bins = np.linspace(min(all_areas), max(all_areas), num_bins + 1)
    
    pred_hist, _ = np.histogram(pred_areas, bins=bins)
    gt_hist, _ = np.histogram(gt_areas, bins=bins)
    
    # Normalize
    pred_hist = pred_hist / max(np.sum(pred_hist), 1)
    gt_hist = gt_hist / max(np.sum(gt_hist), 1)
    
    # Compute distribution distance (KL divergence approximation)
    epsilon = 1e-10
    kl_div = np.sum(gt_hist * np.log((gt_hist + epsilon) / (pred_hist + epsilon)))
    
    return {
        "pred_mean_area_mm2": float(np.mean(pred_areas)),
        "gt_mean_area_mm2": float(np.mean(gt_areas)),
        "pred_std_area_mm2": float(np.std(pred_areas)),
        "gt_std_area_mm2": float(np.std(gt_areas)),
        "kl_divergence": float(kl_div),
        "bins": bins.tolist(),
        "pred_histogram": pred_hist.tolist(),
        "gt_histogram": gt_hist.tolist()
    }


def generate_physical_metrics_summary(
    pred_masks: List[Dict],
    gt_masks: List[Dict],
    class_names: List[str],
    mm2_per_pixel: float = 0.03
) -> Dict:
    """
    Generate comprehensive physical metrics summary.
    
    Args:
        pred_masks: List of predictions
        gt_masks: List of ground truths
        class_names: List of class names
        mm2_per_pixel: Calibration factor
    
    Returns:
        Complete physical metrics summary
    """
    return {
        "overall": {
            "area_error": compute_area_error(pred_masks, gt_masks, mm2_per_pixel),
            "median_deviation": compute_median_deviation(pred_masks, gt_masks, mm2_per_pixel),
            "distribution": compute_area_distribution_comparison(pred_masks, gt_masks, mm2_per_pixel)
        },
        "per_class": compute_per_class_accuracy(pred_masks, gt_masks, class_names, mm2_per_pixel)
    }
