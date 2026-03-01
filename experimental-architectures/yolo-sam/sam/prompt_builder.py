"""
SAM Prompt Builder.

Converts YOLO bounding box detections to SAM prompt format.
Handles coordinate transformations and prompt construction.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def bbox_to_prompt(
    bbox: Union[Tuple[float, float, float, float], Dict],
    image_size: Tuple[int, int] = None,
    normalized: bool = False,
    padding: float = 0.0
) -> np.ndarray:
    """
    Convert bounding box to SAM prompt format.
    
    SAM expects bounding box as numpy array: [x_min, y_min, x_max, y_max]
    in pixel coordinates.
    
    Args:
        bbox: Bounding box as tuple (x_min, y_min, x_max, y_max) or dict
        image_size: (height, width) for denormalization if bbox is normalized
        normalized: Whether bbox coordinates are normalized (0-1)
        padding: Optional padding to add around bbox (in pixels or ratio)
    
    Returns:
        Numpy array of shape (4,) with [x_min, y_min, x_max, y_max]
    
    Example:
        >>> bbox = (100, 150, 200, 250)
        >>> prompt = bbox_to_prompt(bbox)
        >>> prompt
        array([100., 150., 200., 250.])
    """
    # Handle dict input
    if isinstance(bbox, dict):
        x_min = bbox.get("x_min", bbox.get("xmin", 0))
        y_min = bbox.get("y_min", bbox.get("ymin", 0))
        x_max = bbox.get("x_max", bbox.get("xmax", 0))
        y_max = bbox.get("y_max", bbox.get("ymax", 0))
    else:
        x_min, y_min, x_max, y_max = bbox
    
    # Denormalize if needed
    if normalized and image_size is not None:
        h, w = image_size
        x_min *= w
        x_max *= w
        y_min *= h
        y_max *= h
    
    # Apply padding
    if padding > 0:
        if normalized and padding < 1:
            # Ratio-based padding
            pad_x = (x_max - x_min) * padding
            pad_y = (y_max - y_min) * padding
        else:
            # Pixel-based padding
            pad_x = padding
            pad_y = padding
        
        x_min -= pad_x
        y_min -= pad_y
        x_max += pad_x
        y_max += pad_y
        
        # Clip to image bounds if known
        if image_size is not None:
            h, w = image_size
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
    
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def build_prompts(
    bboxes: List[Union[Tuple, Dict]],
    image_size: Tuple[int, int] = None,
    normalized: bool = False,
    padding: float = 0.0
) -> List[np.ndarray]:
    """
    Build SAM prompts for multiple bounding boxes.
    
    Args:
        bboxes: List of bounding boxes
        image_size: (height, width) for denormalization
        normalized: Whether coordinates are normalized
        padding: Optional padding for boxes
    
    Returns:
        List of numpy arrays, each of shape (4,)
    """
    prompts = []
    
    for bbox in bboxes:
        prompt = bbox_to_prompt(
            bbox,
            image_size=image_size,
            normalized=normalized,
            padding=padding
        )
        prompts.append(prompt)
    
    return prompts


def yolo_detections_to_prompts(
    detections: List[Dict],
    image_size: Tuple[int, int] = None,
    padding: float = 0.0
) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Convert YOLO detections to SAM prompts with metadata.
    
    Args:
        detections: List of YOLO detection dictionaries
        image_size: (height, width) of image
        padding: Optional bbox padding
    
    Returns:
        Tuple of (prompts, metadata)
        - prompts: List of bbox arrays for SAM
        - metadata: List of detection metadata (class, confidence, etc.)
    """
    prompts = []
    metadata = []
    
    for det in detections:
        # Get bbox
        bbox = det.get("bbox")
        if bbox is None:
            continue
        
        # Convert to prompt
        prompt = bbox_to_prompt(bbox, image_size=image_size, padding=padding)
        prompts.append(prompt)
        
        # Preserve metadata
        meta = {
            "class_id": det.get("class_id"),
            "class_name": det.get("class_name"),
            "confidence": det.get("confidence"),
            "original_bbox": bbox
        }
        metadata.append(meta)
    
    return prompts, metadata


def add_point_prompts(
    bbox: Tuple[float, float, float, float],
    num_points: int = 1,
    sampling: str = "center"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate point prompts within a bounding box.
    
    Optional: SAM can use points as additional prompts for better accuracy.
    
    Args:
        bbox: Bounding box (x_min, y_min, x_max, y_max)
        num_points: Number of points to generate
        sampling: Point sampling strategy ('center', 'random', 'grid')
    
    Returns:
        Tuple of (point_coords, point_labels)
        - point_coords: Array of shape (num_points, 2) with (x, y) coordinates
        - point_labels: Array of shape (num_points,) with labels (1=foreground)
    """
    x_min, y_min, x_max, y_max = bbox
    
    if sampling == "center":
        # Single center point
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        point_coords = np.array([[cx, cy]])
    
    elif sampling == "random":
        # Random points within bbox
        xs = np.random.uniform(x_min, x_max, num_points)
        ys = np.random.uniform(y_min, y_max, num_points)
        point_coords = np.stack([xs, ys], axis=-1)
    
    elif sampling == "grid":
        # Grid of points
        n = int(np.sqrt(num_points))
        xs = np.linspace(x_min, x_max, n + 2)[1:-1]
        ys = np.linspace(y_min, y_max, n + 2)[1:-1]
        xx, yy = np.meshgrid(xs, ys)
        point_coords = np.stack([xx.flatten(), yy.flatten()], axis=-1)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling}")
    
    # All points are foreground (label=1)
    point_labels = np.ones(len(point_coords), dtype=np.int32)
    
    return point_coords, point_labels


def combine_prompts(
    bbox: Tuple[float, float, float, float],
    point_coords: np.ndarray = None,
    point_labels: np.ndarray = None
) -> Dict:
    """
    Combine bbox and point prompts into SAM format.
    
    Args:
        bbox: Bounding box prompt
        point_coords: Optional point coordinates
        point_labels: Optional point labels
    
    Returns:
        Dictionary with SAM-compatible prompt format
    """
    prompt = {
        "box": bbox_to_prompt(bbox) if bbox else None,
        "point_coords": point_coords,
        "point_labels": point_labels
    }
    
    return prompt
