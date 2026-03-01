"""
YOLO Inference Pipeline.

Handles:
- Single image inference
- Batch inference
- Directory processing
- Result aggregation
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from .detector import YOLODetector


def run_inference(
    model: Union[YOLODetector, str],
    image: Union[np.ndarray, str, Path],
    confidence_threshold: float = 0.35,
    nms_iou_threshold: float = 0.7,
    return_visualization: bool = False
) -> Dict:
    """
    Run inference on a single image.
    
    Args:
        model: YOLODetector instance or path to weights
        image: Input image (array or path)
        confidence_threshold: Detection confidence threshold
        nms_iou_threshold: NMS IoU threshold
        return_visualization: Whether to return annotated image
    
    Returns:
        Dictionary containing:
            - detections: List of detection dictionaries
            - image_size: (height, width) of processed image
            - visualization: Annotated image (if requested)
    """
    # Load model if path provided
    if isinstance(model, (str, Path)):
        detector = YOLODetector(
            model_path=str(model),
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold
        )
    else:
        detector = model
        detector.update_thresholds(confidence_threshold, nms_iou_threshold)
    
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image_array = cv2.imread(str(image))
        if image_array is None:
            raise ValueError(f"Failed to load image: {image}")
    else:
        image_array = image
    
    # Run detection
    detections = detector.detect(image_array, return_masks=True)
    
    result = {
        "detections": detections,
        "image_size": image_array.shape[:2],
        "num_detections": len(detections)
    }
    
    # Create visualization if requested
    if return_visualization:
        vis_image = visualize_detections(image_array, detections)
        result["visualization"] = vis_image
    
    return result


def batch_inference(
    model: Union[YOLODetector, str],
    images: List[Union[np.ndarray, str, Path]],
    confidence_threshold: float = 0.35,
    nms_iou_threshold: float = 0.7,
    progress_callback: callable = None
) -> List[Dict]:
    """
    Run inference on a batch of images.
    
    Args:
        model: YOLODetector instance or path to weights
        images: List of input images
        confidence_threshold: Detection confidence threshold
        nms_iou_threshold: NMS IoU threshold
        progress_callback: Optional callback for progress updates
    
    Returns:
        List of result dictionaries
    """
    # Load model if path provided
    if isinstance(model, (str, Path)):
        detector = YOLODetector(
            model_path=str(model),
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold
        )
    else:
        detector = model
        detector.update_thresholds(confidence_threshold, nms_iou_threshold)
    
    results = []
    total = len(images)
    
    for i, image in enumerate(images):
        result = run_inference(
            detector,
            image,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold
        )
        result["image_index"] = i
        
        if isinstance(image, (str, Path)):
            result["image_path"] = str(image)
        
        results.append(result)
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    return results


def inference_on_directory(
    model: Union[YOLODetector, str],
    image_dir: Union[str, Path],
    output_dir: Union[str, Path] = None,
    confidence_threshold: float = 0.35,
    nms_iou_threshold: float = 0.7,
    save_visualizations: bool = True,
    image_extensions: List[str] = None
) -> Dict:
    """
    Run inference on all images in a directory.
    
    Args:
        model: YOLODetector instance or path to weights
        image_dir: Directory containing images
        output_dir: Directory to save results
        confidence_threshold: Detection confidence threshold
        nms_iou_threshold: NMS IoU threshold
        save_visualizations: Whether to save annotated images
        image_extensions: List of valid image extensions
    
    Returns:
        Summary dictionary with all results
    """
    image_dir = Path(image_dir)
    image_extensions = image_extensions or [".jpg", ".jpeg", ".png", ".bmp"]
    
    # Find all images
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
    
    image_paths = sorted(image_paths)
    
    if not image_paths:
        return {"error": f"No images found in {image_dir}"}
    
    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    results = batch_inference(
        model=model,
        images=image_paths,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold
    )
    
    # Add paths and save visualizations
    for i, result in enumerate(results):
        result["image_path"] = str(image_paths[i])
        result["image_name"] = image_paths[i].name
        
        if save_visualizations and output_dir:
            # Load image and create visualization
            image = cv2.imread(str(image_paths[i]))
            vis_image = visualize_detections(image, result["detections"])
            
            # Save
            vis_path = output_dir / f"vis_{image_paths[i].name}"
            cv2.imwrite(str(vis_path), vis_image)
            result["visualization_path"] = str(vis_path)
    
    # Compute summary statistics
    total_detections = sum(r["num_detections"] for r in results)
    class_counts = {}
    for r in results:
        for det in r["detections"]:
            class_name = det["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    summary = {
        "total_images": len(image_paths),
        "total_detections": total_detections,
        "detections_per_class": class_counts,
        "avg_detections_per_image": total_detections / len(image_paths),
        "results": results
    }
    
    return summary


def visualize_detections(
    image: np.ndarray,
    detections: List[Dict],
    colors: Dict[str, tuple] = None,
    line_thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw detection boxes and masks on image.
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        colors: Class name to BGR color mapping
        line_thickness: Box line thickness
        font_scale: Font scale for labels
    
    Returns:
        Annotated image
    """
    colors = colors or {
        "Scratch": (0, 255, 0),    # Green
        "Dust": (255, 0, 0),       # Blue
        "Rundown": (0, 0, 255)     # Red
    }
    
    vis_image = image.copy()
    
    for det in detections:
        x_min, y_min, x_max, y_max = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]
        
        color = colors.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(
            vis_image,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            color,
            line_thickness
        )
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        
        cv2.rectangle(
            vis_image,
            (int(x_min), int(y_min) - label_size[1] - 10),
            (int(x_min) + label_size[0] + 10, int(y_min)),
            color,
            -1
        )
        
        cv2.putText(
            vis_image,
            label,
            (int(x_min) + 5, int(y_min) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1
        )
        
        # Draw mask if available
        if "mask" in det:
            mask = det["mask"]
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            mask_overlay = vis_image.copy()
            mask_overlay[mask > 0.5] = color
            vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)
    
    return vis_image
