# -*- coding: utf-8 -*-
"""
Configuration module for YOLO + SAM Defect Detection Pipeline.
Compatible with Google Colab and local environments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import os


def is_colab():
    """Check if running in Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_device():
    """Get appropriate device based on availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_base_path():
    """Get base path depending on environment."""
    if is_colab():
        return Path("/content/drive/MyDrive/me_cv")
    return Path(".")


@dataclass
class DataConfig:
    """Configuration for data loading and paths."""
    
    data_root: Path = field(default_factory=lambda: get_base_path() / "data")
    train_images: Path = field(default_factory=lambda: get_base_path() / "data/train/images")
    train_labels: Path = field(default_factory=lambda: get_base_path() / "data/train/labels")
    test_images: Path = field(default_factory=lambda: get_base_path() / "data/test/images")
    test_labels: Path = field(default_factory=lambda: get_base_path() / "data/test/labels")
    train_ratio: float = 0.7
    test_ratio: float = 0.3
    image_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp"])
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Dust",
        1: "RunDown",
        2: "Scratch"
    })
    num_classes: int = 3


@dataclass
class YOLOConfig:
    """Configuration for YOLOv8 detector."""
    
    model_variant: str = "yolov8n-seg"
    pretrained_weights: Optional[str] = None
    
    # Detection thresholds - recall-first strategy
    confidence_threshold: float = 0.20
    nms_iou_threshold: float = 0.7
    
    # Training settings - reduced batch size for Colab compatibility
    epochs: int = 100
    batch_size: int = 8  # Reduced from 16 for Colab T4 GPU memory
    image_size: int = 640
    learning_rate: float = 0.01
    optimizer: str = "SGD"
    momentum: float = 0.937
    nesterov: bool = True
    
    # Augmentation settings
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.0
    hsv_h: float = 0.0
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    flipud: float = 0.5
    fliplr: float = 0.5
    
    # Device - auto-detect
    device: str = field(default_factory=get_device)


@dataclass
class SAMConfig:
    """Configuration for Segment Anything Model."""
    model_type: str = "vit_b"
    checkpoint_path: Optional[str] = field(default_factory=lambda: 
        str(get_base_path() / "sam_vit_b.pth") if is_colab() else None
    )
    mask_threshold: float = 0.5
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    device: str = field(default_factory=get_device)


@dataclass
class PostProcessConfig:
    """Configuration for post-processing operations."""
    mask_binary_threshold: float = 0.2
    min_component_size: int = 50
    connectivity: int = 8
    apply_morphology: bool = True
    morph_kernel_size: int = 3
    mm2_per_pixel: float = 0.03


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics."""
    iou_threshold: float = 0.2
    iou_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    target_recall: float = 0.95
    output_dir: Path = field(default_factory=lambda: get_base_path() / "results")
    save_visualizations: bool = True
    save_csv: bool = True


@dataclass
class MLflowConfig:
    """Configuration for MLflow experiment tracking."""
    tracking_uri: str = field(default_factory=lambda: str(get_base_path() / "mlruns"))
    experiment_name: str = "yolo_sam_defect_detection"
    registry_uri: str = field(default_factory=lambda: str(get_base_path() / "mlruns"))
    model_name: str = "yolo_defect_detector"
    log_artifacts: bool = True
    log_models: bool = True
    log_confusion_matrix: bool = True
    log_sample_predictions: int = 10


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    n_trials: int = 50
    timeout: int = 43200  # 12 hours in seconds
    pruning: bool = True
    parallel_trials: int = 1
    study_name: str = "yolo_sam_optimization"
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    conf_min: float = 0.15
    conf_max: float = 0.5
    nms_min: float = 0.4
    nms_max: float = 0.75
    batch_sizes: List[int] = field(default_factory=lambda: [4, 8, 16])  # Removed 32 for Colab
    sam_mask_thresh_min: float = 0.15
    sam_mask_thresh_max: float = 0.7
    optimization_metric: str = "recall"
    optimization_direction: str = "maximize"


@dataclass
class ColabConfig:
    """Colab-specific configuration."""
    mount_drive: bool = True
    drive_path: str = "/content/drive/MyDrive/me_cv"
    auto_save_checkpoints: bool = True
    checkpoint_interval: int = 10  # Save every N epochs
    use_mixed_precision: bool = True  # FP16 for faster training
    workers: int = 2  # Reduced workers for Colab


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    colab: ColabConfig = field(default_factory=ColabConfig)
    seed: int = 42
    verbose: bool = True
    is_colab: bool = field(default_factory=is_colab)


def get_config() -> Config:
    """Get default configuration with environment detection."""
    config = Config()
    
    # Print environment info
    env_type = "Google Colab" if config.is_colab else "Local"
    device = get_device()
    print("Configuration loaded:")
    print("  Environment: " + env_type)
    print("  Device: " + device)
    print("  Batch size: " + str(config.yolo.batch_size))
    
    return config


def get_colab_config() -> Config:
    """Get configuration optimized for Google Colab."""
    config = Config()
    
    # Ensure Colab-optimized settings
    config.yolo.batch_size = 8
    config.yolo.device = "cuda"
    config.sam.device = "cuda"
    config.tuning.batch_sizes = [4, 8, 16]
    
    return config


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Create config with loaded values
    config = Config()
    # TODO: Apply config_dict values to config
    return config


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    import yaml
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    def convert_paths(obj):
        if isinstance(obj, dict):
            return dict((k, convert_paths(v)) for k, v in obj.items())
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj
    
    config_dict = convert_paths(config_dict)
    
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


# Environment check on import
if __name__ == "__main__":
    cfg = get_config()
    print("Config test passed!")
