import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
DATASET_PATH = PROJECT_ROOT / "data"
TRAIN_IMAGES_PATH = DATASET_PATH / "train" / "images"
TRAIN_LABELS_PATH = DATASET_PATH / "train" / "labels"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FEATURES_DIR = OUTPUTS_DIR / "features"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
MLRUNS_DIR = OUTPUTS_DIR / "mlruns"
DATA_DIR = PROJECT_ROOT / "data"
MASKS_DIR = DATA_DIR / "masks"
SPLITS_DIR = DATA_DIR / "splits"

for dir_path in [FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR, EVALUATION_DIR, 
                 MLRUNS_DIR, MASKS_DIR, SPLITS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Dust", "RunDown", "Scratch"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

PATCH_CLASS_NAMES = ["Dust", "RunDown", "Scratch", "Normal"]
PATCH_CLASS_TO_ID = {name: idx for idx, name in enumerate(PATCH_CLASS_NAMES)}
PATCH_ID_TO_CLASS = {idx: name for idx, name in enumerate(PATCH_CLASS_NAMES)}
PATCH_NUM_CLASSES = len(PATCH_CLASS_NAMES)
NORMAL_CLASS_ID = 3  

DINO_MODEL_NAME = "dinov2_vits14"
DINO_EMBED_DIM = 384 
DINO_PATCH_SIZE = 14
DINO_INPUT_SIZE = 518  

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

XGBOOST_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "reg_alpha": 0.01,
    "reg_lambda": 0.5,
    "gamma": 0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
    "objective": "multi:softprob",  
    "num_class": NUM_CLASSES,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 20,
}

NEG_RATIO = 3.5
DECISION_THRESHOLD = 0.2
MIN_PATCH_AREA = 1
CONFIDENCE_THRESHOLD = 0.4

MLFLOW_EXPERIMENT_NAME = "multiclass_anomaly_detection"
MLFLOW_TRACKING_URI = f"file:///{MLRUNS_DIR.as_posix()}"

SAM_MODEL_TYPE = "vit_b"  
SAM_CHECKPOINT_PATH = MODELS_DIR / "sam_vit_b_01ec64.pth"
SAM_POINTS_PER_SIDE = 32  
SAM_PRED_IOU_THRESH = 0.65
SAM_STABILITY_SCORE_THRESH = 0.80

OPTUNA_N_TRIALS = 50  
OPTUNA_TIMEOUT = 3600  
OPTUNA_CV_FOLDS = 5  
OPTUNA_STUDY_NAME = "xgboost_optimization"
OPTUNA_DIRECTION = "maximize"  

PIXELS_PER_MM = None  

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
