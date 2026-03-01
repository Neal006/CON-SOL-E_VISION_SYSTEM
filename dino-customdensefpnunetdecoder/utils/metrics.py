"""Segmentation metrics: IoU, Dice, Pixel Accuracy."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def calculate_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100
) -> torch.Tensor:
    """Calculate Intersection over Union per class.
    
    Args:
        pred: Predicted class indices (B, H, W) or (H, W).
        target: Ground truth class indices (B, H, W) or (H, W).
        num_classes: Total number of classes.
        ignore_index: Index to ignore.
        
    Returns:
        IoU per class tensor of shape (num_classes,).
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Create mask for valid pixels
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    
    iou_per_class = torch.zeros(num_classes, device=pred.device)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            iou_per_class[cls] = intersection / union
        else:
            iou_per_class[cls] = float('nan')
    
    return iou_per_class


def calculate_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: int = -100
) -> torch.Tensor:
    """Calculate Dice coefficient per class.
    
    Args:
        pred: Predicted class indices.
        target: Ground truth class indices.
        num_classes: Total number of classes.
        smooth: Smoothing factor.
        ignore_index: Index to ignore.
        
    Returns:
        Dice per class tensor of shape (num_classes,).
    """
    pred = pred.flatten()
    target = target.flatten()
    
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    
    dice_per_class = torch.zeros(num_classes, device=pred.device)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        total = pred_cls.sum() + target_cls.sum()
        
        dice_per_class[cls] = (2 * intersection + smooth) / (total + smooth)
    
    return dice_per_class


def calculate_pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """Calculate pixel-wise accuracy.
    
    Args:
        pred: Predicted class indices.
        target: Ground truth class indices.
        ignore_index: Index to ignore.
        
    Returns:
        Pixel accuracy as float.
    """
    valid = target != ignore_index
    
    if valid.sum() == 0:
        return 0.0
    
    correct = ((pred == target) & valid).sum().float()
    total = valid.sum().float()
    
    return (correct / total).item()


class SegmentationMetrics:
    """Class to track and compute segmentation metrics over batches.
    
    Tracks IoU, Dice, and pixel accuracy across multiple batches,
    then computes mean metrics.
    
    Example:
        metrics = SegmentationMetrics(num_classes=2)
        for pred, target in dataloader:
            metrics.update(pred, target)
        results = metrics.compute()
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -100
    ) -> None:
        """Initialize metrics tracker.
        
        Args:
            num_classes: Number of classes.
            class_names: Optional list of class names for reporting.
            ignore_index: Index to ignore in metrics.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.iou_sum = torch.zeros(self.num_classes)
        self.dice_sum = torch.zeros(self.num_classes)
        self.pixel_correct = 0
        self.pixel_total = 0
        self.batch_count = 0
        self.class_counts = torch.zeros(self.num_classes)
    
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:

        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        
        iou = calculate_iou(pred, target, self.num_classes, self.ignore_index)
        dice = calculate_dice(pred, target, self.num_classes, ignore_index=self.ignore_index)
        
        iou_valid = ~torch.isnan(iou)
        iou = torch.where(iou_valid, iou, torch.zeros_like(iou))
        
        self.iou_sum += iou
        self.dice_sum += dice
        self.class_counts += iou_valid.float()
        
        # Pixel accuracy
        valid = target != self.ignore_index
        self.pixel_correct += ((pred == target) & valid).sum().item()
        self.pixel_total += valid.sum().item()
        
        self.batch_count += 1
    
    def compute(self) -> Dict[str, float]:
        
        class_counts = torch.clamp(self.class_counts, min=1)
        
        iou_per_class = self.iou_sum / class_counts
        dice_per_class = self.dice_sum / self.batch_count
        valid_classes = self.class_counts > 0
        mean_iou = iou_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
        mean_dice = dice_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
        
        pixel_acc = self.pixel_correct / max(self.pixel_total, 1)
        
        results = {
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'pixel_accuracy': pixel_acc
        }
        
        for i, name in enumerate(self.class_names):
            results[f'iou_{name}'] = iou_per_class[i].item()
            results[f'dice_{name}'] = dice_per_class[i].item()
        
        return results
    
    def __str__(self) -> str:
        """String representation of current metrics."""
        results = self.compute()
        lines = [
            f"Mean IoU: {results['mean_iou']:.4f}",
            f"Mean Dice: {results['mean_dice']:.4f}",
            f"Pixel Accuracy: {results['pixel_accuracy']:.4f}"
        ]
        return '\n'.join(lines)


def calculate_boundary_tolerant_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    tolerance: int = 3,
    ignore_index: int = -100
) -> torch.Tensor:
    """Calculate IoU with boundary tolerance to handle polygon vs organic shape mismatch.
    
    Dilates both prediction and target masks before computing IoU,
    allowing slack at boundaries where annotations may be imprecise.
    
    Args:
        pred: Predicted class indices (B, H, W) or (H, W).
        target: Ground truth class indices (B, H, W) or (H, W).
        num_classes: Total number of classes.
        tolerance: Dilation kernel size (pixels of boundary slack allowed).
        ignore_index: Index to ignore.
        
    Returns:
        Boundary-tolerant IoU per class tensor of shape (num_classes,).
    """
    import torch.nn.functional as F
    
    pred = pred.flatten()
    target = target.flatten()
    
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    
    # Reshape for morphological operations
    H = W = int(np.sqrt(len(pred)))
    if H * W != len(pred):
        # Fallback to standard IoU if can't reshape
        return calculate_iou(pred.view(-1), target.view(-1), num_classes, ignore_index)
    
    iou_per_class = torch.zeros(num_classes, device=pred.device)
    kernel_size = 2 * tolerance + 1
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).float().view(1, 1, H, W)
        target_cls = (target == cls).float().view(1, 1, H, W)
        
        # Dilate both masks using max pooling
        pred_dilated = F.max_pool2d(pred_cls, kernel_size, stride=1, padding=tolerance)
        target_dilated = F.max_pool2d(target_cls, kernel_size, stride=1, padding=tolerance)
        
        # Tolerant intersection: pred overlaps with dilated target OR dilated pred overlaps with target
        intersection = ((pred_cls * target_dilated) + (pred_dilated * target_cls)).clamp(0, 1).sum()
        union = ((pred_cls + target_cls) > 0).float().sum()
        
        if union > 0:
            iou_per_class[cls] = intersection / union
        else:
            iou_per_class[cls] = float('nan')
    
    return iou_per_class


def calculate_instance_detection_rate(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    iou_threshold: float = 0.3,
    ignore_index: int = -100
) -> Dict[str, float]:
    """Calculate instance-level detection metrics (object-level, not pixel-level).
    
    Extracts connected components from both prediction and target,
    then matches them based on IoU threshold. This is more forgiving
    of boundary mismatches since it evaluates "did we find the defect?"
    
    Args:
        pred: Predicted class indices (B, H, W) or (H, W).
        target: Ground truth class indices (B, H, W) or (H, W).
        num_classes: Total number of classes.
        iou_threshold: Minimum IoU to consider a detection matched.
        ignore_index: Index to ignore.
        
    Returns:
        Dictionary with precision, recall, f1 for instance detection.
    """
    from scipy import ndimage
    
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    H = W = int(np.sqrt(len(pred)))
    pred = pred.reshape(H, W)
    target = target.reshape(H, W)
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # Skip background class (class 0)
    for cls in range(1, num_classes):
        pred_mask = (pred == cls).astype(np.uint8)
        target_mask = (target == cls).astype(np.uint8)
        
        # Extract connected components (instances)
        pred_labels, num_pred = ndimage.label(pred_mask)
        target_labels, num_target = ndimage.label(target_mask)
        
        matched_targets = set()
        
        # For each predicted instance, find best matching target
        for pred_id in range(1, num_pred + 1):
            pred_instance = (pred_labels == pred_id)
            best_iou = 0
            best_target = None
            
            for target_id in range(1, num_target + 1):
                target_instance = (target_labels == target_id)
                
                intersection = (pred_instance & target_instance).sum()
                union = (pred_instance | target_instance).sum()
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_target = target_id
            
            if best_iou >= iou_threshold and best_target is not None:
                total_tp += 1
                matched_targets.add(best_target)
            else:
                total_fp += 1  # Predicted instance not matched
        
        # Unmatched target instances are false negatives
        total_fn += num_target - len(matched_targets)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'instance_precision': precision,
        'instance_recall': recall,
        'instance_f1': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }


def calculate_boundary_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    tolerance: int = 2,
    ignore_index: int = -100
) -> Dict[str, float]:
    """Calculate boundary-specific metrics using Hausdorff-like distance.
    
    Measures how well prediction boundaries match target boundaries,
    with tolerance for annotation imprecision.
    
    Args:
        pred: Predicted class indices (B, H, W) or (H, W).
        target: Ground truth class indices (B, H, W) or (H, W).
        num_classes: Total number of classes.
        tolerance: Distance threshold in pixels for boundary matching.
        ignore_index: Index to ignore.
        
    Returns:
        Dictionary with boundary precision/recall and average boundary distance.
    """
    from scipy import ndimage
    
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    if pred.ndim == 3:
        pred = pred[0]
        target = target[0]
    
    H, W = pred.shape
    
    boundary_distances = []
    boundary_recalls = []
    
    for cls in range(1, num_classes):  # Skip background
        pred_mask = (pred == cls).astype(np.uint8)
        target_mask = (target == cls).astype(np.uint8)
        
        # Extract boundaries using erosion
        pred_eroded = ndimage.binary_erosion(pred_mask)
        target_eroded = ndimage.binary_erosion(target_mask)
        
        pred_boundary = pred_mask.astype(bool) ^ pred_eroded
        target_boundary = target_mask.astype(bool) ^ target_eroded
        
        if not target_boundary.any() or not pred_boundary.any():
            continue
        
        # Distance transform from target boundary
        target_dist = ndimage.distance_transform_edt(~target_boundary)
        pred_dist = ndimage.distance_transform_edt(~pred_boundary)
        
        # Average distance from predicted boundary to nearest target boundary
        pred_to_target_dist = target_dist[pred_boundary].mean() if pred_boundary.any() else 0
        target_to_pred_dist = pred_dist[target_boundary].mean() if target_boundary.any() else 0
        
        # Symmetric boundary distance (like Hausdorff but average)
        avg_boundary_dist = (pred_to_target_dist + target_to_pred_dist) / 2
        boundary_distances.append(avg_boundary_dist)
        
        # Boundary recall: % of target boundary within tolerance distance of prediction
        within_tolerance = (pred_dist[target_boundary] <= tolerance).sum()
        total_target_boundary = target_boundary.sum()
        boundary_recall = within_tolerance / total_target_boundary if total_target_boundary > 0 else 0
        boundary_recalls.append(boundary_recall)
    
    return {
        'avg_boundary_distance': np.mean(boundary_distances) if boundary_distances else 0.0,
        'boundary_recall': np.mean(boundary_recalls) if boundary_recalls else 0.0,
        'boundary_recall_per_class': boundary_recalls
    }


if __name__ == "__main__":
    # Test metrics
    logging.basicConfig(level=logging.INFO)
    
    num_classes = 2
    batch_size = 4
    H, W = 128, 128
    
    # Create mock predictions
    pred = torch.randint(0, num_classes, (batch_size, H, W))
    target = torch.randint(0, num_classes, (batch_size, H, W))
    
    # Test individual functions
    print("Testing individual metric functions:")
    iou = calculate_iou(pred, target, num_classes)
    dice = calculate_dice(pred, target, num_classes)
    acc = calculate_pixel_accuracy(pred, target)
    
    print(f"  IoU per class: {iou}")
    print(f"  Dice per class: {dice}")
    print(f"  Pixel accuracy: {acc:.4f}")
    
    # Test SegmentationMetrics class
    print("\nTesting SegmentationMetrics class:")
    metrics = SegmentationMetrics(
        num_classes=num_classes,
        class_names=['background', 'defect']
    )
    
    # Simulate multiple batches
    for _ in range(5):
        pred = torch.randint(0, num_classes, (batch_size, H, W))
        target = torch.randint(0, num_classes, (batch_size, H, W))
        metrics.update(pred, target)
    
    results = metrics.compute()
    print(f"  Results: {results}")
