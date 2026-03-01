import os
from pathlib import Path
class PathConfig:
    SPGS_NET_ROOT = Path("/content/drive/MyDrive/spgs_net")
    DATA_ROOT = Path("/content/drive/MyDrive/data")
    TRAIN_IMAGES = DATA_ROOT / "train" / "images"
    TRAIN_LABELS = DATA_ROOT / "train" / "labels"
    VALID_IMAGES = DATA_ROOT / "valid" / "images"
    VALID_LABELS = DATA_ROOT / "valid" / "labels"
    TEST_IMAGES = DATA_ROOT / "test" / "images"
    TEST_LABELS = DATA_ROOT / "test" / "labels"
    CHECKPOINT_DIR = SPGS_NET_ROOT / "checkpoints"
    BEST_MODEL = CHECKPOINT_DIR / "best_model.pth"
    OUTPUT_DIR = SPGS_NET_ROOT / "outputs"
    VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
    JSON_DIR = OUTPUT_DIR / "json_results"

class DINOv2Config:
    MODEL_NAME = "dinov2_vits14"
    PATCH_SIZE = 14
    FEATURE_DIM = 384
    FEATURE_LAYERS = [3, 6, 11]  
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    FREEZE_BACKBONE = True

class MLConfig:
    N_ESTIMATORS = 250
    MAX_DEPTH = 8
    LEARNING_RATE = 0.1
    MIN_CHILD_WEIGHT = 1
    SUBSAMPLE = 0.8
    COLSAMPLE_BYTREE = 0.8
    OBJECTIVE = "multi:softprob"
    NUM_CLASSES = 4  
    SCORE_MIN = 0.0
    SCORE_MAX = 1.0
    MODEL_PATH = PathConfig.CHECKPOINT_DIR / "xgboost_patch_classifier.json"

class UpsamplingConfig:
    UPSAMPLE_MODE = "bilinear"
    ALIGN_CORNERS = True
    APPLY_SMOOTHING = True
    SMOOTHING_KERNEL_SIZE = 5
    SMOOTHING_SIGMA = 1.0

class UNetConfig:
    IN_CHANNELS = 3  
    OUT_CHANNELS = 4 
    ENCODER_CHANNELS = [64, 128, 256, 512, 1024]
    USE_ATTENTION = True
    ATTENTION_REDUCTION = 2  
    PRIOR_INJECTION = "attention"
    DROPOUT_RATE = 0.3
    USE_BATCH_NORM = True


class PostProcessingConfig:
    CONFIDENCE_THRESHOLD = 0.5
    MORPHOLOGY_KERNEL_SIZE = 3
    EROSION_ITERATIONS = 1
    DILATION_ITERATIONS = 2
    MIN_DEFECT_AREA_PIXELS = 50 
    CONNECTIVITY = 8  

class CalibrationConfig:
    MM_PER_PIXEL = 0.1  
    EXPECTED_IMAGE_WIDTH = None
    EXPECTED_IMAGE_HEIGHT = None


class OutputConfig:
    BBOX_THICKNESS = 2
    MASK_ALPHA = 0.4  
    CLASS_COLORS = {
        0: (128, 128, 128), 
        1: (0, 255, 255),    
        2: (0, 165, 255),    
        3: (0, 0, 255),      
    }
    
    CLASS_NAMES = {
        0: "Background",
        1: "Dust",
        2: "RunDown",
        3: "Scratch",
    }
    
    EXPORT_JSON = True
    JSON_INDENT = 2


class TrainingConfig:
    EPOCHS = 100
    BATCH_SIZE = 4
    INITIAL_LR = 1e-4
    WEIGHT_DECAY = 1e-5
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "cosine"  
    SCHEDULER_PATIENCE = 10  
    DICE_WEIGHT = 0.5
    FOCAL_WEIGHT = 0.5
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0  
    INITIAL_LR = 1e-4
    WEIGHT_DECAY = 1e-5
    
    
    USE_PRIOR_REWEIGHT = True
    PRIOR_REWEIGHT_FACTOR = 2.0  
    
    EARLY_STOPPING = True
    PATIENCE = 8
    
    SAVE_BEST_ONLY = True
    SAVE_EVERY_N_EPOCHS = 5
    
    DEVICE = "cuda"  
    SEED = 42


class DataConfig:
    YOLO_CLASS_MAP = {
        0: "Dust",
        1: "RunDown", 
        2: "Scratch",
    }
    
    INTERNAL_CLASS_MAP = {
        0: "Background",
        1: "Dust",
        2: "RunDown",
        3: "Scratch",
    }
    
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
    
    TRAIN_IMAGE_SIZE = (560, 560)
    
    USE_AUGMENTATION = True
    AUGMENTATION_PROB = 0.5


def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        PathConfig.CHECKPOINT_DIR,
        PathConfig.OUTPUT_DIR,
        PathConfig.VISUALIZATION_DIR,
        PathConfig.JSON_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test configuration
    print("SPGS-Net Configuration")
    print("=" * 50)
    print(f"Project Root: {PathConfig.PROJECT_ROOT}")
    print(f"Data Root: {PathConfig.DATA_ROOT}")
    print(f"DINOv2 Model: {DINOv2Config.MODEL_NAME}")
    print(f"Patch Size: {DINOv2Config.PATCH_SIZE}")
    print(f"Feature Dim: {DINOv2Config.FEATURE_DIM}")
    print(f"Number of Classes: {MLConfig.NUM_CLASSES}")
    print(f"mm per pixel: {CalibrationConfig.MM_PER_PIXEL}")
    
    # Create directories
    create_directories()
    print("\nDirectories created successfully!")
