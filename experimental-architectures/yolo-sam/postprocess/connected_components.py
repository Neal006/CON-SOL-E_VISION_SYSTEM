"""
Connected Component Analysis.

Handles separation of:
- Multiple scratches in one region
- Clustered dust particles
- Fragmented rundown regions
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np


def find_components(
    mask: np.ndarray,
    connectivity: int = 8,
    min_area: int = 50
) -> Tuple[int, np.ndarray, List[Dict]]:
    """
    Find connected components in binary mask.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        connectivity: 4 or 8 connectivity
        min_area: Minimum component area in pixels
    
    Returns:
        Tuple of:
            - num_components: Number of valid components (excluding background)
            - labels: Label image where each pixel has component ID
            - stats: List of component statistics
    
    Example:
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[10:30, 10:30] = 1
        >>> mask[50:70, 50:70] = 1
        >>> n, labels, stats = find_components(mask)
        >>> n
        2
    """
    # Ensure binary mask
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats_raw, centroids = cv2.connectedComponentsWithStats(
        mask_binary,
        connectivity=connectivity
    )
    
    # Process components (skip background label 0)
    valid_components = []
    new_labels = np.zeros_like(labels)
    new_id = 1
    
    for i in range(1, num_labels):
        area = stats_raw[i, cv2.CC_STAT_AREA]
        
        if area >= min_area:
            # Compute stats
            component_mask = (labels == i)
            
            stats = {
                "id": new_id,
                "area": area,
                "bbox": {
                    "x": stats_raw[i, cv2.CC_STAT_LEFT],
                    "y": stats_raw[i, cv2.CC_STAT_TOP],
                    "width": stats_raw[i, cv2.CC_STAT_WIDTH],
                    "height": stats_raw[i, cv2.CC_STAT_HEIGHT]
                },
                "centroid": {
                    "x": centroids[i, 0],
                    "y": centroids[i, 1]
                }
            }
            
            valid_components.append(stats)
            new_labels[component_mask] = new_id
            new_id += 1
    
    return len(valid_components), new_labels, valid_components


def separate_instances(
    mask: np.ndarray,
    connectivity: int = 8,
    min_area: int = 50
) -> List[np.ndarray]:
    """
    Separate a combined mask into individual instance masks.
    
    Useful for:
    - Multiple scratches
    - Clustered dust particles
    - Fragmented rundown regions
    
    Args:
        mask: Binary mask potentially containing multiple instances
        connectivity: 4 or 8 connectivity
        min_area: Minimum instance area
    
    Returns:
        List of individual binary masks, one per instance
    """
    num_components, labels, _ = find_components(mask, connectivity, min_area)
    
    instance_masks = []
    for i in range(1, num_components + 1):
        instance_mask = (labels == i).astype(np.uint8)
        instance_masks.append(instance_mask)
    
    return instance_masks


def merge_nearby_components(
    mask: np.ndarray,
    distance_threshold: int = 10,
    connectivity: int = 8
) -> np.ndarray:
    """
    Merge nearby components that likely belong to same defect.
    
    Args:
        mask: Binary mask
        distance_threshold: Maximum distance to merge
        connectivity: Connectivity for component finding
    
    Returns:
        Merged mask
    """
    # Dilate to connect nearby components
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (distance_threshold * 2 + 1, distance_threshold * 2 + 1)
    )
    dilated = cv2.dilate(mask, kernel)
    
    # Find labels in dilated space
    _, dilated_labels, _, _ = cv2.connectedComponentsWithStats(dilated, connectivity)
    
    # Map original pixels to dilated labels
    merged = np.zeros_like(mask)
    for label in range(1, dilated_labels.max() + 1):
        # Get pixels in this dilated region
        region = (dilated_labels == label)
        # Keep only original mask pixels
        merged[region & (mask > 0)] = label
    
    return (merged > 0).astype(np.uint8)


def filter_by_shape(
    labels: np.ndarray,
    stats: List[Dict],
    aspect_ratio_range: Tuple[float, float] = (0.1, 10.0),
    solidity_range: Tuple[float, float] = (0.2, 1.0)
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Filter components by shape properties.
    
    Args:
        labels: Label image from find_components
        stats: Stats list from find_components
        aspect_ratio_range: (min, max) aspect ratio
        solidity_range: (min, max) solidity (area / convex_hull_area)
    
    Returns:
        Filtered labels and stats
    """
    filtered_labels = np.zeros_like(labels)
    filtered_stats = []
    new_id = 1
    
    for stat in stats:
        component_id = stat["id"]
        component_mask = (labels == component_id).astype(np.uint8)
        
        # Compute shape properties
        bbox = stat["bbox"]
        aspect_ratio = bbox["width"] / max(bbox["height"], 1)
        
        # Find contours for solidity
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            continue
        
        contour = contours[0]
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = stat["area"] / max(hull_area, 1)
        
        # Check filters
        if (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and
            solidity_range[0] <= solidity <= solidity_range[1]):
            
            stat["aspect_ratio"] = aspect_ratio
            stat["solidity"] = solidity
            stat["id"] = new_id
            
            filtered_labels[component_mask > 0] = new_id
            filtered_stats.append(stat)
            new_id += 1
    
    return filtered_labels, filtered_stats


def get_component_contours(
    mask: np.ndarray,
    min_area: int = 50
) -> List[np.ndarray]:
    """
    Get contours for all components in mask.
    
    Args:
        mask: Binary mask
        min_area: Minimum contour area
    
    Returns:
        List of contour arrays
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter by area
    valid_contours = [
        c for c in contours
        if cv2.contourArea(c) >= min_area
    ]
    
    return valid_contours


def compute_component_features(mask: np.ndarray) -> Dict:
    """
    Compute comprehensive features for a single component.
    
    Args:
        mask: Binary mask for single component
    
    Returns:
        Dictionary of shape features
    """
    # Find contour
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return {}
    
    contour = contours[0]
    
    # Basic measurements
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Rotated rectangle
    rect = cv2.minAreaRect(contour)
    (rx, ry), (rw, rh), angle = rect
    
    # Convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    # Moments
    moments = cv2.moments(contour)
    
    # Circularity
    circularity = 4 * np.pi * area / max(perimeter ** 2, 1)
    
    # Solidity
    solidity = area / max(hull_area, 1)
    
    return {
        "area": area,
        "perimeter": perimeter,
        "bbox": {"x": x, "y": y, "width": w, "height": h},
        "rotated_bbox": {"center": (rx, ry), "size": (rw, rh), "angle": angle},
        "aspect_ratio": w / max(h, 1),
        "circularity": circularity,
        "solidity": solidity,
        "hull_area": hull_area,
        "centroid": {
            "x": moments["m10"] / max(moments["m00"], 1),
            "y": moments["m01"] / max(moments["m00"], 1)
        }
    }
