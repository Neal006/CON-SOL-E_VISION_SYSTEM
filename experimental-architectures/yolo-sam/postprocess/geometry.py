"""
Geometry-Derived Bounding Boxes.

Computes tight, pixel-accurate bounding boxes from mask pixels.

Properties:
- Tight boxes
- Pixel-accurate
- Derived from actual defect extent
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np


def mask_to_bbox(
    mask: np.ndarray,
    format: str = "xyxy"
) -> Tuple:
    """
    Derive tight bounding box from mask pixels.
    
    Args:
        mask: Binary mask (H, W)
        format: Output format
            - 'xyxy': (x_min, y_min, x_max, y_max)
            - 'xywh': (x, y, width, height)
            - 'cxcywh': (center_x, center_y, width, height)
    
    Returns:
        Bounding box in specified format
    
    Example:
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[20:40, 30:60] = 1
        >>> bbox = mask_to_bbox(mask, format='xyxy')
        >>> bbox
        (30, 20, 59, 39)
    """
    # Find non-zero pixels
    rows, cols = np.where(mask > 0)
    
    if len(rows) == 0:
        return None
    
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    
    if format == "xyxy":
        return (x_min, y_min, x_max, y_max)
    elif format == "xywh":
        return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    elif format == "cxcywh":
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        cx = x_min + w / 2
        cy = y_min + h / 2
        return (cx, cy, w, h)
    else:
        raise ValueError(f"Unknown format: {format}")


def compute_bbox(
    mask: np.ndarray,
    padding: int = 0,
    image_shape: Tuple[int, int] = None
) -> Dict:
    """
    Compute bounding box with optional padding.
    
    Result:
    - Tight boxes
    - Pixel-accurate
    - Derived from actual defect extent
    
    Args:
        mask: Binary mask
        padding: Pixels to add around bbox
        image_shape: (H, W) for clipping padded bbox
    
    Returns:
        Dictionary with bbox coordinates and dimensions
    """
    bbox = mask_to_bbox(mask, format="xyxy")
    
    if bbox is None:
        return None
    
    x_min, y_min, x_max, y_max = bbox
    
    # Apply padding
    if padding > 0:
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max += padding
        y_max += padding
        
        # Clip to image bounds
        if image_shape:
            h, w = image_shape
            x_max = min(w - 1, x_max)
            y_max = min(h - 1, y_max)
    
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    area = width * height
    
    return {
        "x_min": int(x_min),
        "y_min": int(y_min),
        "x_max": int(x_max),
        "y_max": int(y_max),
        "width": int(width),
        "height": int(height),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "area": int(area),
        "aspect_ratio": float(width) / max(height, 1)
    }


def compute_rotated_bbox(mask: np.ndarray) -> Dict:
    """
    Compute minimum area rotated bounding box.
    
    Useful for elongated defects like scratches.
    
    Args:
        mask: Binary mask
    
    Returns:
        Dictionary with rotated bbox parameters
    """
    # Find contour
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Get minimum area rect
    contour = contours[0]
    rect = cv2.minAreaRect(contour)
    center, (width, height), angle = rect
    
    # Get corner points
    box_points = cv2.boxPoints(rect)
    
    return {
        "center": {"x": float(center[0]), "y": float(center[1])},
        "width": float(max(width, height)),  # Ensure width >= height
        "height": float(min(width, height)),
        "angle": float(angle),
        "corners": box_points.tolist(),
        "area": float(width * height)
    }


def compute_convex_hull_bbox(mask: np.ndarray) -> Dict:
    """
    Compute bounding box of convex hull.
    
    Args:
        mask: Binary mask
    
    Returns:
        Dictionary with convex hull bbox
    """
    # Find contour
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    contour = contours[0]
    hull = cv2.convexHull(contour)
    
    # Bbox of hull
    x, y, w, h = cv2.boundingRect(hull)
    
    return {
        "x_min": int(x),
        "y_min": int(y),
        "x_max": int(x + w - 1),
        "y_max": int(y + h - 1),
        "width": int(w),
        "height": int(h),
        "hull_points": hull.reshape(-1, 2).tolist()
    }


def masks_to_bboxes(masks: List[np.ndarray]) -> List[Dict]:
    """
    Compute bounding boxes for multiple masks.
    
    Args:
        masks: List of binary masks
    
    Returns:
        List of bbox dictionaries
    """
    bboxes = []
    
    for mask in masks:
        bbox = compute_bbox(mask)
        if bbox is not None:
            bboxes.append(bbox)
    
    return bboxes


def bbox_iou(bbox1: Dict, bbox2: Dict) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        bbox1: First bbox dictionary
        bbox2: Second bbox dictionary
    
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(bbox1["x_min"], bbox2["x_min"])
    y1 = max(bbox1["y_min"], bbox2["y_min"])
    x2 = min(bbox1["x_max"], bbox2["x_max"])
    y2 = min(bbox1["y_max"], bbox2["y_max"])
    
    # Intersection
    inter_width = max(0, x2 - x1 + 1)
    inter_height = max(0, y2 - y1 + 1)
    inter_area = inter_width * inter_height
    
    # Union
    area1 = bbox1["area"]
    area2 = bbox2["area"]
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def merge_overlapping_bboxes(
    bboxes: List[Dict],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Merge overlapping bounding boxes.
    
    Args:
        bboxes: List of bbox dictionaries
        iou_threshold: Minimum IoU to merge
    
    Returns:
        Merged list of bboxes
    """
    if not bboxes:
        return []
    
    # Sort by area descending
    sorted_bboxes = sorted(bboxes, key=lambda b: -b["area"])
    
    merged = []
    used = set()
    
    for i, bbox in enumerate(sorted_bboxes):
        if i in used:
            continue
        
        # Find overlapping boxes
        group = [bbox]
        used.add(i)
        
        for j, other in enumerate(sorted_bboxes):
            if j in used:
                continue
            
            if bbox_iou(bbox, other) >= iou_threshold:
                group.append(other)
                used.add(j)
        
        # Merge group
        x_min = min(b["x_min"] for b in group)
        y_min = min(b["y_min"] for b in group)
        x_max = max(b["x_max"] for b in group)
        y_max = max(b["y_max"] for b in group)
        
        merged.append({
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "width": x_max - x_min + 1,
            "height": y_max - y_min + 1,
            "area": (x_max - x_min + 1) * (y_max - y_min + 1),
            "merged_count": len(group)
        })
    
    return merged
