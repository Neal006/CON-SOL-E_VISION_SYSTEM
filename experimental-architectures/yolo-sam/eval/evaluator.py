"""
Main Evaluator Orchestrator.

Combines all metrics and generates comprehensive evaluation reports.
Outputs CSV in the specified format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .detection_metrics import (
    compute_recall,
    compute_precision,
    compute_box_iou,
    compute_miss_rate,
    compute_per_class_metrics
)
from .segmentation_metrics import (
    compute_mask_iou,
    compute_dice_score,
    compute_boundary_accuracy,
    compute_batch_segmentation_metrics,
    compute_per_class_segmentation_metrics
)
from .physical_metrics import (
    compute_area_error,
    compute_median_deviation,
    compute_per_class_accuracy,
    generate_physical_metrics_summary
)


class Evaluator:
    """
    Main evaluation orchestrator.
    
    Aggregates all metrics:
    - Detection: Recall, Precision, Box IoU, Miss rate
    - Segmentation: Mask IoU, Dice score, Boundary accuracy
    - Physical: Area error (%), Median mm² deviation
    
    Generates CSV output in specified format:
    | Class | IoU | Dice | Area Error (%) | Recall |
    """
    
    def __init__(
        self,
        class_names: List[str] = None,
        mm2_per_pixel: float = 0.03,
        iou_threshold: float = 0.5,
        target_recall: float = 0.95
    ):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names
            mm2_per_pixel: Physical calibration factor
            iou_threshold: IoU threshold for detection matching
            target_recall: Target recall threshold
        """
        self.class_names = class_names or ["Scratch", "Dust", "Rundown"]
        self.mm2_per_pixel = mm2_per_pixel
        self.iou_threshold = iou_threshold
        self.target_recall = target_recall
        
        self.results = {}
    
    def evaluate(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> Dict:
        """
        Run complete evaluation.
        
        Args:
            predictions: List of predictions with 'bbox', 'mask', 'class_name'
            ground_truths: List of ground truths with same keys
        
        Returns:
            Complete evaluation results dictionary
        """
        results = {}
        
        # Detection metrics
        results["detection"] = {
            "overall": {
                "recall": compute_recall(predictions, ground_truths, self.iou_threshold),
                "precision": compute_precision(predictions, ground_truths, self.iou_threshold),
                "box_iou": compute_box_iou(predictions, ground_truths, self.iou_threshold),
                "miss_rate": compute_miss_rate(predictions, ground_truths, self.iou_threshold)
            },
            "per_class": compute_per_class_metrics(
                predictions, ground_truths, self.class_names, self.iou_threshold
            )
        }
        
        # Segmentation metrics
        results["segmentation"] = {
            "overall": compute_batch_segmentation_metrics(predictions, ground_truths),
            "per_class": compute_per_class_segmentation_metrics(
                predictions, ground_truths, self.class_names
            )
        }
        
        # Physical metrics
        results["physical"] = generate_physical_metrics_summary(
            predictions, ground_truths, self.class_names, self.mm2_per_pixel
        )
        
        # Summary
        results["summary"] = self._generate_summary(results)
        
        self.results = results
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate high-level summary."""
        detection = results["detection"]["overall"]
        segmentation = results["segmentation"]["overall"]
        physical = results["physical"]["overall"]
        
        return {
            "mean_iou": segmentation.get("mean_iou", 0.0),
            "mean_dice": segmentation.get("mean_dice", 0.0),
            "area_error_percent": physical["area_error"].get("mean_error_percent", 0.0),
            "instance_recall": detection["recall"]["recall"],
            "recall_target_met": detection["recall"]["recall"] >= self.target_recall,
            "box_iou": detection["box_iou"].get("mean_iou", 0.0),
            "boundary_accuracy": segmentation.get("mean_boundary_accuracy", 0.0)
        }
    
    def generate_csv(
        self,
        output_path: Union[str, Path] = None
    ) -> pd.DataFrame:
        """
        Generate CSV evaluation report in specified format.
        
        Format:
        | Class   | IoU | Dice | Area Error (%) | Recall |
        | ------- | --- | ---- | -------------- | ------ |
        | Scratch |     |      |                |        |
        | Dust    |     |      |                |        |
        | Rundown |     |      |                |        |
        
        Args:
            output_path: Optional path to save CSV
        
        Returns:
            DataFrame with evaluation results
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        
        rows = []
        
        for class_name in self.class_names:
            # Get per-class metrics
            seg_metrics = self.results["segmentation"]["per_class"].get(class_name, {})
            det_metrics = self.results["detection"]["per_class"].get(class_name, {})
            phys_metrics = self.results["physical"]["per_class"].get(class_name, {})
            
            row = {
                "Class": class_name,
                "IoU": round(seg_metrics.get("mean_iou", 0.0), 4),
                "Dice": round(seg_metrics.get("mean_dice", 0.0), 4),
                "Area Error (%)": round(phys_metrics.get("mean_error_percent", 0.0), 2),
                "Recall": round(det_metrics.get("recall", 0.0), 4)
            }
            rows.append(row)
        
        # Add overall row
        summary = self.results["summary"]
        rows.append({
            "Class": "Overall",
            "IoU": round(summary["mean_iou"], 4),
            "Dice": round(summary["mean_dice"], 4),
            "Area Error (%)": round(summary["area_error_percent"], 2),
            "Recall": round(summary["instance_recall"], 4)
        })
        
        df = pd.DataFrame(rows)
        
        if output_path:
            df.to_csv(output_path, index=False)
        
        return df
    
    def generate_detailed_report(
        self,
        output_path: Union[str, Path] = None
    ) -> str:
        """
        Generate detailed text report.
        
        Args:
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        
        lines = []
        lines.append("=" * 60)
        lines.append("YOLO + SAM DEFECT DETECTION EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Summary
        summary = self.results["summary"]
        lines.append("EVALUATION SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Mean IoU (per class + overall): {summary['mean_iou']:.4f}")
        lines.append(f"Dice score: {summary['mean_dice']:.4f}")
        lines.append(f"Area error in mm²: {summary['area_error_percent']:.2f}%")
        lines.append(f"Instance recall: {summary['instance_recall']:.4f} (target: >{self.target_recall})")
        lines.append(f"Box IoU: {summary['box_iou']:.4f}")
        lines.append(f"Boundary accuracy: {summary['boundary_accuracy']:.4f}")
        lines.append("")
        
        # Target check
        if summary["recall_target_met"]:
            lines.append("✓ RECALL TARGET MET")
        else:
            lines.append("✗ RECALL TARGET NOT MET")
        lines.append("")
        
        # Per-class results
        lines.append("PER-CLASS RESULTS")
        lines.append("-" * 40)
        
        df = self.generate_csv()
        lines.append(df.to_string(index=False))
        lines.append("")
        
        # Detection details
        lines.append("DETECTION METRICS")
        lines.append("-" * 40)
        det = self.results["detection"]["overall"]
        lines.append(f"True Positives: {det['recall']['true_positives']}")
        lines.append(f"False Positives: {det['precision']['false_positives']}")
        lines.append(f"False Negatives: {det['recall']['false_negatives']}")
        lines.append(f"Miss Rate: {det['miss_rate']['miss_rate']:.4f}")
        lines.append("")
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report
    
    def get_metric(self, metric_path: str):
        """
        Get specific metric by path.
        
        Args:
            metric_path: Dot-separated path, e.g., 'detection.overall.recall.recall'
        
        Returns:
            Metric value
        """
        parts = metric_path.split('.')
        value = self.results
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    def meets_target_recall(self) -> bool:
        """Check if recall meets target."""
        if not self.results:
            return False
        return self.results["summary"]["recall_target_met"]


def evaluate_predictions(
    predictions: List[Dict],
    ground_truths: List[Dict],
    output_dir: Union[str, Path] = None,
    class_names: List[str] = None,
    mm2_per_pixel: float = 0.03
) -> Dict:
    """
    Convenience function for quick evaluation.
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        output_dir: Directory to save outputs
        class_names: List of class names
        mm2_per_pixel: Calibration factor
    
    Returns:
        Evaluation results dictionary
    """
    evaluator = Evaluator(
        class_names=class_names,
        mm2_per_pixel=mm2_per_pixel
    )
    
    results = evaluator.evaluate(predictions, ground_truths)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator.generate_csv(output_dir / "evaluation.csv")
        evaluator.generate_detailed_report(output_dir / "evaluation_report.txt")
    
    return results
