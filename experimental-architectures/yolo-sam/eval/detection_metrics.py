"""
Detection Metrics.

Primary metric: Recall (minimize miss rate for industrial QC)
Additional: Precision, Box IoU, mAP
"""

from typing import Dict, List, Tuple

import numpy as np


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        box1: (x_min, y_min, x_max, y_max)
        box2: (x_min, y_min, x_max, y_max)
    
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_detections(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[List[int], List[int], List[int]]:
    """
    Match predictions to ground truths using IoU.
    
    Args:
        predictions: List of prediction dicts with 'bbox' key
        ground_truths: List of ground truth dicts with 'bbox' key
        iou_threshold: IoU threshold for matching
    
    Returns:
        Tuple of (tp_indices, fp_indices, fn_indices)
        - tp_indices: Indices of true positive predictions
        - fp_indices: Indices of false positive predictions  
        - fn_indices: Indices of unmatched ground truths
    """
    if not predictions or not ground_truths:
        if not predictions:
            return [], [], list(range(len(ground_truths)))
        else:
            return [], list(range(len(predictions))), []
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(predictions), len(ground_truths)))
    
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            pred_bbox = pred["bbox"]
            gt_bbox = gt["bbox"]
            
            # Handle dict format
            if isinstance(pred_bbox, dict):
                pred_bbox = (pred_bbox["x_min"], pred_bbox["y_min"], 
                            pred_bbox["x_max"], pred_bbox["y_max"])
            if isinstance(gt_bbox, dict):
                gt_bbox = (gt_bbox["x_min"], gt_bbox["y_min"],
                          gt_bbox["x_max"], gt_bbox["y_max"])
            
            iou_matrix[i, j] = compute_iou(pred_bbox, gt_bbox)
    
    # Greedy matching
    matched_gts = set()
    tp_indices = []
    fp_indices = []
    
    # Sort predictions by confidence if available
    pred_order = list(range(len(predictions)))
    if "confidence" in predictions[0]:
        pred_order = sorted(pred_order, key=lambda i: -predictions[i]["confidence"])
    
    for pred_idx in pred_order:
        best_iou = 0
        best_gt = -1
        
        for gt_idx in range(len(ground_truths)):
            if gt_idx in matched_gts:
                continue
            
            iou = iou_matrix[pred_idx, gt_idx]
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        
        if best_gt >= 0:
            tp_indices.append(pred_idx)
            matched_gts.add(best_gt)
        else:
            fp_indices.append(pred_idx)
    
    # Find unmatched ground truths (false negatives)
    fn_indices = [i for i in range(len(ground_truths)) if i not in matched_gts]
    
    return tp_indices, fp_indices, fn_indices


def compute_recall(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """
    Compute recall (primary metric for defect detection).
    
    Recall = TP / (TP + FN) = Detected Defects / Total Defects
    
    High recall is critical for industrial QC to minimize missed defects.
    
    Args:
        predictions: List of predictions with 'bbox' key
        ground_truths: List of ground truths with 'bbox' key
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with recall and related metrics
    """
    tp_indices, fp_indices, fn_indices = match_detections(
        predictions, ground_truths, iou_threshold
    )
    
    tp = len(tp_indices)
    fn = len(fn_indices)
    
    recall = tp / max(tp + fn, 1)
    
    return {
        "recall": recall,
        "true_positives": tp,
        "false_negatives": fn,
        "total_ground_truths": len(ground_truths),
        "iou_threshold": iou_threshold
    }


def compute_precision(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """
    Compute precision.
    
    Precision = TP / (TP + FP) = Correct Detections / Total Detections
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with precision and related metrics
    """
    tp_indices, fp_indices, fn_indices = match_detections(
        predictions, ground_truths, iou_threshold
    )
    
    tp = len(tp_indices)
    fp = len(fp_indices)
    
    precision = tp / max(tp + fp, 1)
    
    return {
        "precision": precision,
        "true_positives": tp,
        "false_positives": fp,
        "total_predictions": len(predictions),
        "iou_threshold": iou_threshold
    }


def compute_box_iou(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """
    Compute average box IoU for matched detections.
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with IoU statistics
    """
    tp_indices, _, _ = match_detections(predictions, ground_truths, iou_threshold)
    
    if not tp_indices:
        return {
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "min_iou": 0.0,
            "max_iou": 0.0,
            "matched_count": 0
        }
    
    # Compute IoUs for matched pairs
    ious = []
    matched_gts = set()
    
    for pred_idx in tp_indices:
        pred = predictions[pred_idx]
        pred_bbox = pred["bbox"]
        if isinstance(pred_bbox, dict):
            pred_bbox = (pred_bbox["x_min"], pred_bbox["y_min"],
                        pred_bbox["x_max"], pred_bbox["y_max"])
        
        best_iou = 0
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gts:
                continue
            
            gt_bbox = gt["bbox"]
            if isinstance(gt_bbox, dict):
                gt_bbox = (gt_bbox["x_min"], gt_bbox["y_min"],
                          gt_bbox["x_max"], gt_bbox["y_max"])
            
            iou = compute_iou(pred_bbox, gt_bbox)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
        
        if best_iou > 0:
            ious.append(best_iou)
    
    ious = np.array(ious) if ious else np.array([0.0])
    
    return {
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "min_iou": float(np.min(ious)),
        "max_iou": float(np.max(ious)),
        "matched_count": len(ious)
    }


def compute_miss_rate(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """
    Compute miss rate (1 - recall).
    
    Miss rate = FN / (TP + FN) = Missed Defects / Total Defects
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        iou_threshold: IoU threshold
    
    Returns:
        Dictionary with miss rate
    """
    recall_result = compute_recall(predictions, ground_truths, iou_threshold)
    
    return {
        "miss_rate": 1.0 - recall_result["recall"],
        "false_negatives": recall_result["false_negatives"],
        "total_ground_truths": recall_result["total_ground_truths"]
    }


def compute_f1_score(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """
    Compute F1 score (harmonic mean of precision and recall).
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        iou_threshold: IoU threshold
    
    Returns:
        Dictionary with F1 score and components
    """
    precision_result = compute_precision(predictions, ground_truths, iou_threshold)
    recall_result = compute_recall(predictions, ground_truths, iou_threshold)
    
    precision = precision_result["precision"]
    recall = recall_result["recall"]
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: List[float] = None
) -> Dict:
    """
    Compute mean Average Precision across IoU thresholds.
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        iou_thresholds: List of IoU thresholds (default: 0.5:0.05:0.95)
    
    Returns:
        Dictionary with mAP values
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
    
    aps = []
    
    for iou_thresh in iou_thresholds:
        precision_result = compute_precision(predictions, ground_truths, iou_thresh)
        recall_result = compute_recall(predictions, ground_truths, iou_thresh)
        
        # Simple AP approximation using precision at this threshold
        aps.append(precision_result["precision"])
    
    return {
        "mAP": float(np.mean(aps)),
        "mAP50": aps[0] if len(aps) > 0 else 0.0,
        "mAP75": aps[5] if len(aps) > 5 else 0.0,
        "per_threshold_ap": {f"AP@{t:.2f}": ap for t, ap in zip(iou_thresholds, aps)}
    }


def compute_per_class_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_names: List[str],
    iou_threshold: float = 0.5
) -> Dict:
    """
    Compute detection metrics per class.
    
    Args:
        predictions: List of predictions with 'class_name' key
        ground_truths: List of ground truths with 'class_name' key
        class_names: List of class names
        iou_threshold: IoU threshold
    
    Returns:
        Dictionary with per-class metrics
    """
    results = {}
    
    for class_name in class_names:
        class_preds = [p for p in predictions if p.get("class_name") == class_name]
        class_gts = [g for g in ground_truths if g.get("class_name") == class_name]
        
        recall_result = compute_recall(class_preds, class_gts, iou_threshold)
        precision_result = compute_precision(class_preds, class_gts, iou_threshold)
        f1_result = compute_f1_score(class_preds, class_gts, iou_threshold)
        
        results[class_name] = {
            "recall": recall_result["recall"],
            "precision": precision_result["precision"],
            "f1_score": f1_result["f1_score"],
            "num_predictions": len(class_preds),
            "num_ground_truths": len(class_gts),
            "true_positives": recall_result["true_positives"],
            "false_positives": precision_result["false_positives"],
            "false_negatives": recall_result["false_negatives"]
        }
    
    return results
