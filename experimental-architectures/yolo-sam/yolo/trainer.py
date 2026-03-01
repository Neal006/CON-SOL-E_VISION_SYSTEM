"""
YOLOv8 Training Configuration and Trainer.

Handles:
- Training configuration for grayscale defect detection
- Data augmentation optimized for grayscale images
- Checkpoint management
- MLflow integration hooks
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml


class YOLOTrainer:
    """
    YOLOv8 Trainer for defect detection.
    
    Features:
    - Grayscale-optimized augmentation
    - Configurable training parameters
    - Checkpoint saving and resumption
    - MLflow callback integration
    - Auto-loads settings from config.py
    """
    
    def __init__(
        self,
        model_variant: str = None,
        data_yaml_path: str = None,
        output_dir: str = None,
        epochs: int = None,
        batch_size: int = None,
        image_size: int = None,
        learning_rate: float = None,
        device: str = None,
        config = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model_variant: YOLOv8 model variant (uses config if None)
            data_yaml_path: Path to data.yaml file
            output_dir: Output directory for training artifacts
            epochs: Number of training epochs (uses config if None)
            batch_size: Training batch size (uses config if None)
            image_size: Input image size (uses config if None)
            learning_rate: Initial learning rate (uses config if None)
            device: Training device (uses config if None)
            config: Optional Config object (auto-loads if None)
        """
        # Load config if not provided
        if config is None:
            from config import get_config
            config = get_config()
        
        self.config = config
        self.model_variant = model_variant if model_variant is not None else config.yolo.model_variant
        self.data_yaml_path = data_yaml_path
        self.output_dir = Path(output_dir if output_dir is not None else "runs/train")
        self.epochs = epochs if epochs is not None else config.yolo.epochs
        self.batch_size = batch_size if batch_size is not None else config.yolo.batch_size
        self.image_size = image_size if image_size is not None else config.yolo.image_size
        self.learning_rate = learning_rate if learning_rate is not None else config.yolo.learning_rate
        self.device = device if device is not None else config.yolo.device
        
        self.model = None
        self.callbacks = []
    
    def create_data_yaml(
        self,
        train_images: str,
        val_images: str,
        class_names: Dict[int, str],
        output_path: str = "data.yaml"
    ) -> str:
        """
        Create YOLO data.yaml configuration file.
        
        Args:
            train_images: Path to training images directory
            val_images: Path to validation images directory
            class_names: Mapping of class ID to name
            output_path: Path to save data.yaml
        
        Returns:
            Path to created data.yaml file
        """
        data_config = {
            "path": str(Path(train_images).parent.parent),
            "train": str(Path(train_images).name),
            "val": str(Path(val_images).name),
            "names": class_names,
            "nc": len(class_names)
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        self.data_yaml_path = output_path
        return output_path
    
    def get_augmentation_config(self) -> Dict:
        """
        Get augmentation configuration optimized for grayscale images.
        
        Returns:
            Dictionary of augmentation parameters
        """
        return {
            # Disable color augmentations (no effect on grayscale)
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.4,  # Brightness/value augmentation still useful
            
            # Geometric augmentations
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            
            # Flip augmentations
            "flipud": 0.5,
            "fliplr": 0.5,
            
            # Mosaic and mixup
            "mosaic": 1.0,
            "mixup": 0.0,
            
            # Other
            "copy_paste": 0.0,
            "erasing": 0.0
        }
    
    def add_callback(self, callback: Callable):
        """Add a training callback."""
        self.callbacks.append(callback)
    
    def train(
        self,
        resume: bool = False,
        pretrained_weights: str = None,
        **kwargs
    ) -> Dict:
        """
        Start training.
        
        Args:
            resume: Whether to resume from last checkpoint
            pretrained_weights: Path to pretrained weights
            **kwargs: Additional training arguments
        
        Returns:
            Training results dictionary
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not found. "
                "Install with: pip install ultralytics"
            )
        
        # Load model
        if pretrained_weights and Path(pretrained_weights).exists():
            self.model = YOLO(pretrained_weights)
        else:
            self.model = YOLO(self.model_variant)
        
        # Get augmentation config
        aug_config = self.get_augmentation_config()
        
        # Merge with kwargs
        train_args = {
            "data": self.data_yaml_path,
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.image_size,
            "lr0": self.learning_rate,
            "device": self.device,
            "project": str(self.output_dir.parent),
            "name": self.output_dir.name,
            "exist_ok": True,
            "resume": resume,
            "workers": 0,  # Disable multiprocessing to avoid Windows MemoryError
            **aug_config,
            **kwargs
        }
        
        # Train
        results = self.model.train(**train_args)
        
        return {
            "model_path": str(self.output_dir / "weights" / "best.pt"),
            "results": results
        }
    
    def validate(self, weights_path: str = None) -> Dict:
        """
        Run validation.
        
        Args:
            weights_path: Path to model weights
        
        Returns:
            Validation metrics
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package not found.")
        
        if weights_path:
            model = YOLO(weights_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model loaded. Train first or provide weights path.")
        
        results = model.val(data=self.data_yaml_path)
        
        return {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.p.mean() if hasattr(results.box, 'p') else None,
            "recall": results.box.r.mean() if hasattr(results.box, 'r') else None
        }
    
    def export(
        self,
        weights_path: str,
        format: str = "onnx",
        **kwargs
    ) -> str:
        """
        Export model to different formats.
        
        Args:
            weights_path: Path to model weights
            format: Export format ('onnx', 'torchscript', etc.)
        
        Returns:
            Path to exported model
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package not found.")
        
        model = YOLO(weights_path)
        export_path = model.export(format=format, **kwargs)
        
        return export_path
    
    def get_training_config(self) -> Dict:
        """Get current training configuration."""
        return {
            "model_variant": self.model_variant,
            "data_yaml_path": self.data_yaml_path,
            "output_dir": str(self.output_dir),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "augmentation": self.get_augmentation_config()
        }


def create_training_config(
    train_images: str,
    train_labels: str,
    val_images: str,
    val_labels: str,
    class_names: Dict[int, str],
    output_dir: str = "runs/train",
    **kwargs
) -> Dict:
    """
    Create a complete training configuration.
    
    Returns dictionary ready for YOLOTrainer.
    """
    config = {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "class_names": class_names,
        "output_dir": output_dir,
        **kwargs
    }
    
    return config
