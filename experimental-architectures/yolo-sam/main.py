"""
YOLO + SAM Defect Detection Pipeline

Main orchestration for:
1. Loading configuration
2. Initializing YOLO detector
3. Initializing SAM segmentor
4. Processing images
5. Running evaluation
6. Exporting results

Usage:
    python main.py --mode train --data_dir data/train
    python main.py --mode inference --image_dir data/test/images
    python main.py --mode eval --output results/evaluation.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from config import Config, get_config
from data import DefectDataset
from preprocess import expand_grayscale, decode_label_file
from yolo import YOLODetector, YOLOTrainer
from sam import MaskGenerator
from postprocess import (
    apply_threshold,
    find_components,
    compute_bbox,
    compute_pixel_area,
    pixel_to_mm2
)
from eval import Evaluator


class DefectDetectionPipeline:
    """
    Complete YOLO + SAM defect detection pipeline.
    
    Workflow:
    1. Load grayscale image
    2. Expand to 3 channels
    3. Run YOLO detection
    4. Run SAM segmentation on each detection
    5. Post-process masks
    6. Compute physical measurements
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        
        self.yolo_detector = None
        self.mask_generator = None
        self.evaluator = None
        
        self._initialized = False
    
    def initialize(
        self,
        yolo_weights: str = None,
        sam_checkpoint: str = None
    ):
        """
        Initialize all components.
        
        Args:
            yolo_weights: Path to YOLO weights
            sam_checkpoint: Path to SAM checkpoint
        """
        print("Initializing YOLO detector...")
        self.yolo_detector = YOLODetector(
            model_path=yolo_weights,
            model_variant=self.config.yolo.model_variant,
            confidence_threshold=self.config.yolo.confidence_threshold,
            nms_iou_threshold=self.config.yolo.nms_iou_threshold,
            device=self.config.yolo.device,
            class_names=self.config.data.class_names
        )
        
        print("Initializing SAM segmentor...")
        self.mask_generator = MaskGenerator(
            sam_model_type=self.config.sam.model_type,
            sam_checkpoint=sam_checkpoint or self.config.sam.checkpoint_path,
            device=self.config.sam.device,
            mask_threshold=self.config.sam.mask_threshold
        )
        
        print("Initializing evaluator...")
        self.evaluator = Evaluator(
            class_names=list(self.config.data.class_names.values()),
            mm2_per_pixel=self.config.postprocess.mm2_per_pixel,
            iou_threshold=self.config.eval.iou_threshold
        )
        
        self._initialized = True
        print("Pipeline initialized!")
    
    def process_image(
        self,
        image: Union[np.ndarray, str, Path],
        return_visualization: bool = True
    ) -> Dict:
        """
        Process a single image through the entire pipeline.
        
        Args:
            image: Input image (grayscale) or path
            return_visualization: Whether to return visualization
        
        Returns:
            Dictionary with results
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        # Load image if path
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
        
        original_size = image.shape[:2]
        
        # Expand grayscale to 3 channels
        image_3ch = expand_grayscale(image)
        
        # Run YOLO detection
        detections = self.yolo_detector.detect(image_3ch, return_masks=False)
        
        if not detections:
            return {
                "detections": [],
                "masks": [],
                "measurements": [],
                "visualization": image_3ch if return_visualization else None
            }
        
        # Run SAM segmentation
        sam_results = self.mask_generator.generate_masks(image_3ch, detections)
        
        # Post-process and compute measurements
        results = []
        for result in sam_results:
            mask = result.get("mask")
            if mask is None:
                continue
            
            # Connected component analysis
            num_components, labels, component_stats = find_components(
                mask,
                min_area=self.config.postprocess.min_component_size
            )
            
            # For each component
            for stat in component_stats:
                component_mask = (labels == stat["id"]).astype(np.uint8)
                
                # Compute geometry-derived bbox
                bbox = compute_bbox(component_mask, image_shape=original_size)
                if bbox is None:
                    continue
                
                # Compute area
                pixel_area = compute_pixel_area(component_mask)
                mm2_area = pixel_to_mm2(pixel_area, self.config.postprocess.mm2_per_pixel)
                
                results.append({
                    "class_id": result.get("class_id"),
                    "class_name": result.get("class_name"),
                    "confidence": result.get("confidence"),
                    "bbox": (bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]),
                    "mask": component_mask,
                    "pixel_area": pixel_area,
                    "mm2_area": mm2_area
                })
        
        output = {
            "detections": results,
            "num_instances": len(results),
            "original_size": original_size
        }
        
        # Create visualization
        if return_visualization:
            output["visualization"] = self._create_visualization(image_3ch, results)
        
        return output
    
    def process_directory(
        self,
        image_dir: Union[str, Path],
        output_dir: Union[str, Path] = None,
        save_visualizations: bool = True
    ) -> Dict:
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory with images
            output_dir: Output directory for results
            save_visualizations: Whether to save visualization images
        
        Returns:
            Summary dictionary
        """
        image_dir = Path(image_dir)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find images
        extensions = self.config.data.image_extensions
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        image_paths = sorted(image_paths)
        
        if not image_paths:
            return {"error": f"No images found in {image_dir}"}
        
        print(f"Processing {len(image_paths)} images...")
        
        all_results = []
        total_detections = 0
        class_counts = {}
        
        for i, image_path in enumerate(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {image_path.name}")
            
            try:
                result = self.process_image(
                    image_path,
                    return_visualization=save_visualizations
                )
                result["image_path"] = str(image_path)
                result["image_name"] = image_path.name
                
                # Count detections
                for det in result["detections"]:
                    class_name = det.get("class_name", "unknown")
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    total_detections += 1
                
                # Save visualization
                if save_visualizations and output_dir and "visualization" in result:
                    vis_path = output_dir / f"result_{image_path.name}"
                    cv2.imwrite(str(vis_path), result["visualization"])
                    result["visualization_path"] = str(vis_path)
                    del result["visualization"]  # Don't keep in memory
                
                # Don't keep masks in summary
                for det in result["detections"]:
                    if "mask" in det:
                        del det["mask"]
                
                all_results.append(result)
                
            except Exception as e:
                print(f"    Error: {e}")
                all_results.append({
                    "image_path": str(image_path),
                    "error": str(e)
                })
        
        summary = {
            "total_images": len(image_paths),
            "total_detections": total_detections,
            "detections_per_class": class_counts,
            "results": all_results
        }
        
        # Save summary
        if output_dir:
            import json
            summary_path = output_dir / "results_summary.json"
            
            # Make serializable copy
            summary_copy = {k: v for k, v in summary.items() if k != "results"}
            with open(summary_path, 'w') as f:
                json.dump(summary_copy, f, indent=2)
        
        return summary
    
    def evaluate(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        output_dir: Union[str, Path] = None
    ) -> Dict:
        """
        Run evaluation.
        
        Args:
            predictions: List of prediction dictionaries
            ground_truths: List of ground truth dictionaries
            output_dir: Optional output directory for reports
        
        Returns:
            Evaluation results
        """
        results = self.evaluator.evaluate(predictions, ground_truths)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.evaluator.generate_csv(output_dir / "evaluation.csv")
            self.evaluator.generate_detailed_report(output_dir / "report.txt")
        
        return results
    
    def _create_visualization(
        self,
        image: np.ndarray,
        results: List[Dict]
    ) -> np.ndarray:
        """Create visualization with detections."""
        colors = {
            "Scratch": (0, 255, 0),    # Green
            "Dust": (255, 0, 0),       # Blue
            "Rundown": (0, 0, 255)     # Red
        }
        
        vis_image = image.copy()
        
        for result in results:
            bbox = result.get("bbox")
            class_name = result.get("class_name", "unknown")
            confidence = result.get("confidence", 0)
            mm2_area = result.get("mm2_area", 0)
            
            color = colors.get(class_name, (128, 128, 128))
            
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name}: {confidence:.2f} ({mm2_area:.2f}mm²)"
                cv2.putText(
                    vis_image, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
            
            # Draw mask overlay
            mask = result.get("mask")
            if mask is not None:
                if mask.shape[:2] != vis_image.shape[:2]:
                    mask = cv2.resize(mask, (vis_image.shape[1], vis_image.shape[0]))
                
                overlay = vis_image.copy()
                overlay[mask > 0] = color
                vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
        
        return vis_image


def train(config: Config, data_dir: str, output_dir: str):
    """Train YOLO model."""
    print("Starting YOLO training...")
    
    # Check for existing data.yaml in data directory
    data_yaml_path = Path(data_dir) / "data.yaml"
    
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml_path}. "
            f"Please create a data.yaml file in the data directory."
        )
    
    print(f"Using existing data.yaml: {data_yaml_path}")
    
    trainer = YOLOTrainer(
        model_variant=config.yolo.model_variant,
        data_yaml_path=str(data_yaml_path),
        output_dir=output_dir,
        epochs=config.yolo.epochs,
        batch_size=config.yolo.batch_size,
        image_size=config.yolo.image_size,
        learning_rate=config.yolo.learning_rate,
        device=config.yolo.device
    )
    
    results = trainer.train()
    
    print(f"Training complete! Model saved to: {results['model_path']}")
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="YOLO + SAM Defect Detection Pipeline")
    
    parser.add_argument(
        "--mode",
        choices=["train", "inference", "eval"],
        default="inference",
        help="Pipeline mode"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Image directory for inference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default=None,
        help="Path to YOLO weights"
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    if args.mode == "train":
        train(config, args.data_dir, args.output_dir)
        
    elif args.mode == "inference":
        pipeline = DefectDetectionPipeline(config)
        pipeline.initialize(
            yolo_weights=args.yolo_weights,
            sam_checkpoint=args.sam_checkpoint
        )
        
        image_dir = args.image_dir or str(config.data.test_images)
        results = pipeline.process_directory(
            image_dir,
            output_dir=args.output_dir,
            save_visualizations=True
        )
        
        print(f"\nProcessed {results['total_images']} images")
        print(f"Total detections: {results['total_detections']}")
        print(f"Per-class counts: {results['detections_per_class']}")
        
    elif args.mode == "eval":
        print("Evaluation mode - requires predictions and ground truths")
        print("Use the Evaluator class directly for evaluation")


if __name__ == "__main__":
    main()
