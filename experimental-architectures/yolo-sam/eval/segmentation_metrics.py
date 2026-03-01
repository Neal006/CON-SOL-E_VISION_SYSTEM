"""
Segmentation Metrics.

Metrics:
- Mask IoU (Intersection over Union)
- Dice score (F1 for segmentation)
- Boundary accuracy
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np


def compute_mask_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> float:
    """
    Compute IoU between predicted and ground truth masks.
    
    IoU = Intersection / Union
    
    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)
    
    Returns:
        IoU value between 0 and 1
    """
    # Ensure binary
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    # Resize if shapes don't match
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    
    if union == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
    
    return intersection / union


def compute_dice_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> float:
    """
    Compute Dice score (F1 for segmentation).
    
    Dice = 2 * |Intersection| / (|Pred| + |GT|)
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
    
    Returns:
        Dice score between 0 and 1
    """
    # Ensure binary
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    # Resize if shapes don't match
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    intersection = np.sum(pred & gt)
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)
    
    if pred_sum + gt_sum == 0:
        return 1.0
    
    return 2 * intersection / (pred_sum + gt_sum)


def compute_boundary_accuracy(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    dilation_radius: int = 2
) -> float:
    """
    Compute boundary accuracy.
    
    Measures how well the predicted boundary matches ground truth.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        dilation_radius: Radius for boundary tolerance
    
    Returns:
        Boundary accuracy between 0 and 1
    """
    # Ensure binary
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    # Resize if shapes don't match
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    # Extract boundaries
    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = pred - cv2.erode(pred, kernel)
    gt_boundary = gt - cv2.erode(gt, kernel)
    
    if np.sum(gt_boundary) == 0:
        return 1.0 if np.sum(pred_boundary) == 0 else 0.0
    
    # Dilate ground truth boundary for tolerance
    tolerance_kernel = np.ones((dilation_radius * 2 + 1, dilation_radius * 2 + 1), np.uint8)
    gt_boundary_dilated = cv2.dilate(gt_boundary, tolerance_kernel)
    
    # Compute accuracy
    correct = np.sum(pred_boundary & gt_boundary_dilated)
    total = np.sum(pred_boundary)
    
    if total == 0:
        return 1.0
    
    return correct / total


def compute_precision_recall_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> Dict:
    """
    Compute pixel-level precision, recall, and IoU.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
    
    Returns:
        Dictionary with precision, recall, IoU, and Dice
    """
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    
    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn)
    }


def compute_batch_segmentation_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict]
) -> Dict:
    """
    Compute segmentation metrics across batch of masks.
    
    Args:
        predictions: List of dicts with 'mask' key
        ground_truths: List of dicts with 'mask' key
    
    Returns:
        Dictionary with aggregated metrics
    """
    ious = []
    dices = []
    boundary_accs = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_mask = pred.get("mask")
        gt_mask = gt.get("mask")
        
        if pred_mask is None or gt_mask is None:
            continue
        
        ious.append(compute_mask_iou(pred_mask, gt_mask))
        dices.append(compute_dice_score(pred_mask, gt_mask))
        boundary_accs.append(compute_boundary_accuracy(pred_mask, gt_mask))
    
    if not ious:
        return {
            "mean_iou": 0.0,
            "mean_dice": 0.0,
            "mean_boundary_accuracy": 0.0,
            "count": 0
        }
    
    return {
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious)),
        "median_iou": float(np.median(ious)),
        "mean_dice": float(np.mean(dices)),
        "std_dice": float(np.std(dices)),
        "median_dice": float(np.median(dices)),
        "mean_boundary_accuracy": float(np.mean(boundary_accs)),
        "std_boundary_accuracy": float(np.std(boundary_accs)),
        "count": len(ious)
    }


def compute_per_class_segmentation_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_names: List[str]
) -> Dict:
    """
    Compute segmentation metrics per class.
    
    Args:
        predictions: List of predictions with 'mask' and 'class_name'
        ground_truths: List of ground truths with 'mask' and 'class_name'
        class_names: List of class names
    
    Returns:
        Dictionary with per-class metrics
    """
    results = {}
    
    for class_name in class_names:
        class_preds = [p for p in predictions if p.get("class_name") == class_name]
        class_gts = [g for g in ground_truths if g.get("class_name") == class_name]
        
        # Match predictions to ground truths by overlap
        class_metrics = compute_batch_segmentation_metrics(class_preds, class_gts)
        results[class_name] = class_metrics
    
    return results


def compute_confusion_matrix_mask(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int = 2
) -> np.ndarray:
    """
    Compute confusion matrix from masks.
    
    Args:
        pred_mask: Predicted mask with class labels
        gt_mask: Ground truth mask with class labels
        num_classes: Number of classes (including background)
    
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask.astype(np.float32), 
                               (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(np.int32)
    
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for gt_class in range(num_classes):
        for pred_class in range(num_classes):
            confusion[gt_class, pred_class] = np.sum(
                (gt_mask == gt_class) & (pred_mask == pred_class)
            )
    
    return confusion


def compute_mean_iou_from_confusion(confusion: np.ndarray) -> float:
    """
    Compute mean IoU from confusion matrix.
    
    Args:
        confusion: Confusion matrix (num_classes, num_classes)
    
    Returns:
        Mean IoU across classes
    """
    num_classes = confusion.shape[0]
    ious = []
    
    for i in range(num_classes):
        tp = confusion[i, i]
        fn = np.sum(confusion[i, :]) - tp
        fp = np.sum(confusion[:, i]) - tp
        
        if tp + fn + fp > 0:
            ious.append(tp / (tp + fn + fp))
    
    return float(np.mean(ious)) if ious else 0.0
