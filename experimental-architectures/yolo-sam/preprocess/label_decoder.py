"""
Label Decoding Pipeline.

Handles:
1. Reading YOLO format label files
2. Decoding polygon annotations
3. Converting polygons to axis-aligned bounding boxes
4. Creating binary masks from polygons
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np


def parse_yolo_polygon(line: str) -> Tuple[int, List[Tuple[float, float]]]:
    """
    Parse a single line of YOLO polygon annotation.
    
    Format: class_id x1 y1 x2 y2 ... xn yn
    All coordinates are normalized (0-1).
    
    Args:
        line: Single line from label file
    
    Returns:
        Tuple of (class_id, list of (x, y) polygon points)
    
    Example:
        >>> line = "2 0.383 0.076 0.186 0.961 0.197 0.972"
        >>> class_id, polygon = parse_yolo_polygon(line)
        >>> class_id
        2
        >>> polygon
        [(0.383, 0.076), (0.186, 0.961), (0.197, 0.972)]
    """
    parts = line.strip().split()
    
    if len(parts) < 5:  # Minimum: class_id + 2 points (4 coords)
        raise ValueError(f"Invalid annotation format: {line}")
    
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    
    # Parse coordinate pairs
    polygon = []
    for i in range(0, len(coords) - 1, 2):
        x = coords[i]
        y = coords[i + 1]
        polygon.append((x, y))
    
    return class_id, polygon


def polygon_to_bbox(
    polygon: List[Tuple[float, float]],
    image_width: int = None,
    image_height: int = None,
    normalized: bool = True
) -> Dict[str, float]:
    """
    Convert polygon coordinates to axis-aligned bounding box.
    
    Args:
        polygon: List of (x, y) normalized coordinates
        image_width: Image width for denormalization
        image_height: Image height for denormalization
        normalized: If True, return normalized coordinates
    
    Returns:
        Dictionary with x_min, y_min, x_max, y_max
    
    Example:
        >>> polygon = [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)]
        >>> bbox = polygon_to_bbox(polygon)
        >>> bbox
        {'x_min': 0.1, 'y_min': 0.2, 'x_max': 0.3, 'y_max': 0.4}
    """
    if not polygon:
        raise ValueError("Empty polygon provided")
    
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    if not normalized and image_width and image_height:
        x_min = int(x_min * image_width)
        x_max = int(x_max * image_width)
        y_min = int(y_min * image_height)
        y_max = int(y_max * image_height)
    
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max
    }


def decode_label_file(
    label_path: Union[str, Path],
    image_width: int = None,
    image_height: int = None,
    class_names: Dict[int, str] = None
) -> List[Dict]:
    """
    Read and decode a complete YOLO label file.
    
    Args:
        label_path: Path to the label file
        image_width: Image width for coordinate denormalization
        image_height: Image height for coordinate denormalization
        class_names: Optional mapping of class ID to name
    
    Returns:
        List of decoded annotations, each containing:
            - class_id: Integer class identifier
            - class_name: String class name (if class_names provided)
            - polygon: List of (x, y) coordinates
            - bbox: Bounding box dictionary
    
    Example output:
        [
            {
                'class_id': 2,
                'class_name': 'Rundown',
                'polygon': [(0.383, 0.076), ...],
                'bbox': {'x_min': 0.186, 'y_min': 0.070, ...}
            }
        ]
    """
    label_path = Path(label_path)
    annotations = []
    
    if not label_path.exists():
        return annotations
    
    class_names = class_names or {
        0: "Dust",
        1: "RunDown",
        2: "Scratch"
    }
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                class_id, polygon = parse_yolo_polygon(line)
                bbox = polygon_to_bbox(
                    polygon,
                    image_width=image_width,
                    image_height=image_height,
                    normalized=(image_width is None or image_height is None)
                )
                
                annotation = {
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, f"class_{class_id}"),
                    "polygon": polygon,
                    "bbox": bbox
                }
                
                annotations.append(annotation)
                
            except ValueError as e:
                print(f"Warning: Skipping invalid annotation in {label_path}: {e}")
                continue
    
    return annotations


def create_binary_mask(
    polygon: List[Tuple[float, float]],
    image_width: int,
    image_height: int
) -> np.ndarray:
    """
    Create a binary mask from polygon coordinates.
    
    Args:
        polygon: List of (x, y) normalized coordinates
        image_width: Target mask width
        image_height: Target mask height
    
    Returns:
        Binary numpy array of shape (H, W) with values 0 or 1
    """
    # Convert normalized coordinates to pixel coordinates
    points = np.array([
        [int(x * image_width), int(y * image_height)]
        for x, y in polygon
    ], dtype=np.int32)
    
    # Create empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # Fill polygon
    cv2.fillPoly(mask, [points], 1)
    
    return mask


def create_instance_masks(
    annotations: List[Dict],
    image_width: int,
    image_height: int
) -> List[Dict]:
    """
    Create binary masks for all annotations.
    
    Args:
        annotations: List of annotation dictionaries with 'polygon' key
        image_width: Target mask width
        image_height: Target mask height
    
    Returns:
        List of annotation dictionaries with added 'mask' key
    """
    for ann in annotations:
        ann["mask"] = create_binary_mask(
            ann["polygon"],
            image_width,
            image_height
        )
    
    return annotations


def bbox_to_xyxy(
    bbox: Dict[str, float],
    image_width: int = None,
    image_height: int = None
) -> Tuple[int, int, int, int]:
    """
    Convert bbox dictionary to (x1, y1, x2, y2) format.
    
    Args:
        bbox: Dictionary with x_min, y_min, x_max, y_max
        image_width: Image width for denormalization
        image_height: Image height for denormalization
    
    Returns:
        Tuple of (x1, y1, x2, y2) pixel coordinates
    """
    if image_width and image_height:
        x1 = int(bbox["x_min"] * image_width)
        y1 = int(bbox["y_min"] * image_height)
        x2 = int(bbox["x_max"] * image_width)
        y2 = int(bbox["y_max"] * image_height)
    else:
        x1 = bbox["x_min"]
        y1 = bbox["y_min"]
        x2 = bbox["x_max"]
        y2 = bbox["y_max"]
    
    return (x1, y1, x2, y2)
