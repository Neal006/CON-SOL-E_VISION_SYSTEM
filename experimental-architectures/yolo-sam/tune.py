"""
Hyperparameter Tuning Entry Point

Automated hyperparameter tuning for YOLO + SAM defect detection pipeline.

Features:
- Optuna-based hyperparameter search
- MLflow experiment tracking
- Best model auto-deployment to registry

Usage:
    python tune.py --n_trials 50 --experiment "yolo_sam_tuning"
    python tune.py --n_trials 20 --quick --experiment "quick_test"
"""

import argparse
from pathlib import Path
from typing import Dict

from config import Config, get_config
from yolo import YOLOTrainer
from eval import Evaluator
from mlflow_integration import (
    ExperimentTracker,
    HyperparameterTuner,
    ModelRegistry,
    MLflowCallback
)


def create_training_objective(
    config: Config,
    data_yaml: str,
    val_images: str,
    val_labels: str
):
    """
    Create objective function for hyperparameter optimization.
    
    Args:
        config: Configuration object
        data_yaml: Path to YOLO data.yaml
        val_images: Path to validation images
        val_labels: Path to validation labels
    
    Returns:
        Tuple of (train_fn, eval_fn)
    """
    
    def train_fn(params: Dict):
        """Train model with given parameters."""
        trainer = YOLOTrainer(
            model_variant=config.yolo.model_variant,
            data_yaml_path=data_yaml,
            epochs=min(params.get("epochs", 50), config.yolo.epochs),
            batch_size=params.get("batch", 16),
            learning_rate=params.get("lr0", 0.01),
            device=config.yolo.device
        )
        
        # Override augmentation settings
        aug_params = {
            "mosaic": params.get("mosaic", 1.0),
            "mixup": params.get("mixup", 0.0),
            "hsv_v": params.get("hsv_v", 0.4),
            "degrees": params.get("degrees", 10),
            "translate": params.get("translate", 0.1),
            "scale": params.get("scale", 0.5),
            "flipud": params.get("flipud", 0.5),
            "fliplr": params.get("fliplr", 0.5),
        }
        
        results = trainer.train(**aug_params)
        
        return {
            "model_path": results["model_path"],
            "conf_threshold": params.get("conf", 0.35),
            "nms_iou": params.get("nms_iou", 0.7)
        }
    
    def eval_fn(train_result: Dict) -> Dict:
        """Evaluate trained model."""
        from yolo import YOLODetector
        from data import DefectDataset
        
        # Load detector with tuned thresholds
        detector = YOLODetector(
            model_path=train_result["model_path"],
            confidence_threshold=train_result["conf_threshold"],
            nms_iou_threshold=train_result["nms_iou"],
            class_names=config.data.class_names
        )
        
        # Load validation dataset
        dataset = DefectDataset(
            images_dir=val_images,
            labels_dir=val_labels,
            class_names=config.data.class_names,
            return_info=True
        )
        
        # Run inference and compute metrics
        predictions = []
        ground_truths = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            image = sample["image"].numpy().transpose(1, 2, 0)
            image = (image * 255).astype("uint8")
            
            # Get detections
            detections = detector.detect(image, return_masks=False)
            predictions.extend(detections)
            
            # Get ground truths
            for label in sample["labels"]:
                gt = {
                    "bbox": (
                        label["bbox"]["x_min"],
                        label["bbox"]["y_min"],
                        label["bbox"]["x_max"],
                        label["bbox"]["y_max"]
                    ),
                    "class_id": label["class_id"],
                    "class_name": label["class_name"]
                }
                ground_truths.append(gt)
        
        # Compute metrics
        evaluator = Evaluator(
            class_names=list(config.data.class_names.values()),
            mm2_per_pixel=config.postprocess.mm2_per_pixel
        )
        
        # Simple metrics computation
        from eval.detection_metrics import compute_recall, compute_precision, compute_f1_score
        
        recall_result = compute_recall(predictions, ground_truths)
        precision_result = compute_precision(predictions, ground_truths)
        f1_result = compute_f1_score(predictions, ground_truths)
        
        return {
            "recall": recall_result["recall"],
            "precision": precision_result["precision"],
            "f1_score": f1_result["f1_score"]
        }
    
    return train_fn, eval_fn


def run_tuning(
    config: Config,
    data_dir: str,
    n_trials: int = 50,
    experiment_name: str = "yolo_sam_optimization",
    output_dir: str = "tuning_results"
):
    """
    Run hyperparameter tuning.
    
    Args:
        config: Configuration object
        data_dir: Data directory with train/test splits
        n_trials: Number of optimization trials
        experiment_name: MLflow experiment name
        output_dir: Output directory for results
    """
    print(f"Starting hyperparameter tuning with {n_trials} trials...")
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup paths
    train_images = data_dir / "train" / "images"
    train_labels = data_dir / "train" / "labels"
    val_images = data_dir / "test" / "images"
    val_labels = data_dir / "test" / "labels"
    
    # Create data.yaml
    data_yaml = output_dir / "data.yaml"
    trainer = YOLOTrainer()
    trainer.create_data_yaml(
        train_images=str(train_images),
        val_images=str(val_images),
        class_names=config.data.class_names,
        output_path=str(data_yaml)
    )
    
    # Create objective functions
    train_fn, eval_fn = create_training_objective(
        config, str(data_yaml), str(val_images), str(val_labels)
    )
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        experiment_name=experiment_name,
        n_trials=n_trials,
        timeout=config.tuning.timeout,
        pruning=config.tuning.pruning
    )
    
    # Create and run optimization
    tuner.create_objective(
        train_fn=train_fn,
        eval_fn=eval_fn,
        metric="recall",  # Primary metric
        search_space_fn=tuner.define_yolo_search_space
    )
    
    results = tuner.run_optimization(direction="maximize")
    
    # Print results
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 60)
    print(f"Best recall: {results['best_value']:.4f}")
    print(f"Trials completed: {results['n_completed']}/{n_trials}")
    print(f"Trials pruned: {results['n_pruned']}")
    print("\nBest parameters:")
    for key, value in results['best_params'].items():
        print(f"  {key}: {value}")
    
    # Export results
    tuner.export_study_results(str(output_dir / "study_results.csv"))
    
    # Visualizations (if available)
    try:
        tuner.visualize_optimization(str(output_dir / "plots"))
        print(f"\nVisualizations saved to: {output_dir / 'plots'}")
    except Exception as e:
        print(f"\nCould not create visualizations: {e}")
    
    # Print parameter importance
    importance = tuner.get_importance()
    if importance:
        print("\nParameter importance:")
        for param, score in sorted(importance.items(), key=lambda x: -x[1])[:5]:
            print(f"  {param}: {score:.3f}")
    
    # Save best config
    best_config_path = output_dir / "best_config.yaml"
    print(f"\nBest configuration saved to: {best_config_path}")
    
    return results


def main():
    """Main entry point for tuning."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for YOLO + SAM defect detection"
    )
    
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="yolo_sam_optimization",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory with train/test splits"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tuning_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with reduced epochs per trial"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum tuning time in seconds"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    # Apply quick mode settings
    if args.quick:
        config.yolo.epochs = 10
        config.tuning.timeout = 3600  # 1 hour
    
    if args.timeout:
        config.tuning.timeout = args.timeout
    
    # Run tuning
    results = run_tuning(
        config=config,
        data_dir=args.data_dir,
        n_trials=args.n_trials,
        experiment_name=args.experiment,
        output_dir=args.output_dir
    )
    
    print("\n✓ Tuning complete!")
    print(f"View results with: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
