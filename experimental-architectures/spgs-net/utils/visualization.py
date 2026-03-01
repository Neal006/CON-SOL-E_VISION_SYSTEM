"""
Visualization Utilities for SPGS-Net
=====================================
Output generation, overlay visualization, and JSON export.

Section 8 of Architecture: Output Generation & Visualization
- Original image with bounding boxes and segmentation overlays
- Defect type and confidence score
- Defect area in mm²
- Human-readable visual format and machine-readable JSON format
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import OutputConfig, DataConfig, CalibrationConfig


# =============================================================================
# Section 8: Output Generation & Visualization
# Visual overlays and annotations
# =============================================================================

def overlay_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = None
) -> np.ndarray:
    """
    Overlay segmentation mask on image with class-specific colors.
    
    Section 8: Segmentation overlays for visual inspection.
    
    Args:
        image: Original image (H, W, 3) in BGR format
        mask: Segmentation mask (H, W) with class indices
        alpha: Transparency for overlay (default from config)
        
    Returns:
        Image with segmentation overlay
    """
    if alpha is None:
        alpha = OutputConfig.MASK_ALPHA
    
    # Create colored overlay
    overlay = image.copy()
    
    for class_id, color in OutputConfig.CLASS_COLORS.items():
        if class_id == 0:  # Skip background
            continue
        class_mask = (mask == class_id)
        if class_mask.any():
            overlay[class_mask] = color
    
    # Blend with original image
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return result


def draw_bounding_boxes(
    image: np.ndarray,
    instances: List[Dict],
    show_area: bool = True,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes around defect instances.
    
    Section 8: Bounding boxes with defect type and confidence.
    
    Args:
        image: Image to draw on (H, W, 3) BGR format
        instances: List of instance dicts with 'bbox', 'class_id', 'confidence', 'area_mm2'
        show_area: Whether to show area in mm²
        show_confidence: Whether to show confidence score
        
    Returns:
        Image with bounding boxes drawn
    """
    result = image.copy()
    
    for inst in instances:
        class_id = inst.get('class_id', 0)
        if class_id == 0:  # Skip background
            continue
            
        bbox = inst.get('bbox', None)
        if bbox is None:
            continue
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = OutputConfig.CLASS_COLORS.get(class_id, (255, 255, 255))
        class_name = OutputConfig.CLASS_NAMES.get(class_id, "Unknown")
        
        # Section 8: Draw bounding box
        cv2.rectangle(
            result, (x1, y1), (x2, y2),
            color, OutputConfig.BBOX_THICKNESS
        )
        
        # Build label text
        label_parts = [class_name]
        
        if show_confidence and 'confidence' in inst:
            label_parts.append(f"{inst['confidence']:.2f}")
        
        if show_area and 'area_mm2' in inst:
            label_parts.append(f"{inst['area_mm2']:.2f}mm²")
        
        label = " | ".join(label_parts)
        
        # Section 8: Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX,
            OutputConfig.FONT_SCALE, 1
        )
        
        cv2.rectangle(
            result,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w + 5, y1),
            color, -1
        )
        
        # Draw label text
        cv2.putText(
            result, label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            OutputConfig.FONT_SCALE,
            (0, 0, 0),  # Black text on colored background
            1, cv2.LINE_AA
        )
    
    return result


def create_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    instances: List[Dict],
    show_overlay: bool = True,
    show_boxes: bool = True
) -> np.ndarray:
    """
    Create complete visualization with overlay and bounding boxes.
    
    Section 8: Combined visual output.
    
    Args:
        image: Original image (H, W, 3) BGR
        mask: Segmentation mask (H, W)
        instances: List of detected instances
        show_overlay: Whether to show segmentation overlay
        show_boxes: Whether to show bounding boxes
        
    Returns:
        Visualization image
    """
    result = image.copy()
    
    if show_overlay:
        result = overlay_segmentation(result, mask)
    
    if show_boxes:
        result = draw_bounding_boxes(result, instances)
    
    return result


# =============================================================================
# Section 8: JSON Export for Factory Systems
# Machine-readable format for integration
# =============================================================================

def export_results_json(
    image_path: str,
    instances: List[Dict],
    output_path: Optional[Union[str, Path]] = None,
    additional_info: Optional[Dict] = None
) -> Dict:
    """
    Export detection results to JSON format.
    
    Section 8: Machine-readable JSON format for factory systems.
    
    Args:
        image_path: Path to source image
        instances: List of detected defect instances
        output_path: Optional path to save JSON file
        additional_info: Optional additional metadata
        
    Returns:
        Results dictionary
    """
    results = {
        "image_path": str(image_path),
        "timestamp": datetime.now().isoformat(),
        "calibration": {
            "mm_per_pixel": CalibrationConfig.MM_PER_PIXEL,
        },
        "summary": {
            "total_defects": len([i for i in instances if i.get('class_id', 0) != 0]),
            "defects_by_class": {},
        },
        "defects": []
    }
    
    # Count defects by class
    class_counts = {name: 0 for name in DataConfig.INTERNAL_CLASS_MAP.values()}
    total_area = 0.0
    
    for inst in instances:
        class_id = inst.get('class_id', 0)
        if class_id == 0:  # Skip background
            continue
        
        class_name = OutputConfig.CLASS_NAMES.get(class_id, "Unknown")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Section 8: Defect details
        defect_info = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": float(inst.get('confidence', 0.0)),
            "area_pixels": int(inst.get('area_pixels', 0)),
            "area_mm2": float(inst.get('area_mm2', 0.0)),
            "bounding_box": {
                "x1": int(inst['bbox'][0]) if 'bbox' in inst else 0,
                "y1": int(inst['bbox'][1]) if 'bbox' in inst else 0,
                "x2": int(inst['bbox'][2]) if 'bbox' in inst else 0,
                "y2": int(inst['bbox'][3]) if 'bbox' in inst else 0,
            }
        }
        
        results["defects"].append(defect_info)
        total_area += defect_info["area_mm2"]
    
    # Remove zero counts and add summary
    results["summary"]["defects_by_class"] = {
        k: v for k, v in class_counts.items() if v > 0 and k != "Background"
    }
    results["summary"]["total_area_mm2"] = round(total_area, 2)
    
    # Add additional info if provided
    if additional_info:
        results["additional_info"] = additional_info
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=OutputConfig.JSON_INDENT)
    
    return results


def save_visualization(
    visualization: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 95
) -> str:
    """
    Save visualization image to file.
    
    Args:
        visualization: Image to save (BGR format)
        output_path: Output file path
        quality: JPEG quality (1-100)
        
    Returns:
        Saved file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set JPEG quality
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(output_path), visualization, 
                   [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(output_path), visualization)
    
    return str(output_path)


if __name__ == "__main__":
    # Test visualization utilities
    print("Testing visualization utilities...")
    
    # Create dummy data
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_mask = np.zeros((480, 640), dtype=np.uint8)
    dummy_mask[100:200, 150:250] = 1  # Dust
    dummy_mask[250:350, 300:450] = 2  # RunDown
    dummy_mask[50:100, 400:500] = 3   # Scratch
    
    dummy_instances = [
        {'class_id': 1, 'bbox': [150, 100, 250, 200], 'confidence': 0.95, 'area_mm2': 12.5},
        {'class_id': 2, 'bbox': [300, 250, 450, 350], 'confidence': 0.87, 'area_mm2': 45.0},
        {'class_id': 3, 'bbox': [400, 50, 500, 100], 'confidence': 0.92, 'area_mm2': 8.2},
    ]
    
    # Test visualization
    vis = create_visualization(dummy_image, dummy_mask, dummy_instances)
    print(f"Visualization shape: {vis.shape}")
    
    # Test JSON export
    results = export_results_json("test_image.jpg", dummy_instances)
    print(f"JSON results: {json.dumps(results['summary'], indent=2)}")
