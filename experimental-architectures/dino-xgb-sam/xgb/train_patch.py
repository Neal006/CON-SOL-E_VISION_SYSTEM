"""
XGBoost Training on Patch-Level Features
Trains a 4-class classifier: Dust, RunDown, Scratch, Normal
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import mlflow
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURES_DIR, MODELS_DIR, EVALUATION_DIR, MLRUNS_DIR,
    PATCH_CLASS_NAMES, PATCH_ID_TO_CLASS, PATCH_NUM_CLASSES, NORMAL_CLASS_ID,
    XGBOOST_PARAMS, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RANDOM_SEED
)
from dino.patch_feature_extractor import load_patch_features_from_parquet


def load_all_patch_features():
    """Load patch features for all splits."""
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        parquet_path = FEATURES_DIR / f"patch_{split_name}_features.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Patch features not found: {parquet_path}. "
                f"Run python dino/patch_feature_extractor.py first."
            )
        
        features, labels, image_refs = load_patch_features_from_parquet(parquet_path)
        splits[split_name] = {
            'features': features,
            'labels': labels,
            'image_refs': image_refs
        }
        print(f"Loaded {split_name}: {features.shape[0]:,} patches")
    
    return splits


def balance_dataset(features, labels, max_normal_ratio=2.0):
    """
    Balance the dataset by undersampling Normal class.
    Normal class typically dominates (>90% of patches).
    
    Args:
        max_normal_ratio: Maximum ratio of normal to defect patches
    """
    # Count classes
    counter = Counter(labels)
    defect_count = sum(count for cls, count in counter.items() if cls != NORMAL_CLASS_ID)
    normal_count = counter.get(NORMAL_CLASS_ID, 0)
    
    if normal_count == 0:
        return features, labels
    
    # Calculate max allowed normal samples
    max_normal = int(defect_count * max_normal_ratio)
    
    if normal_count <= max_normal:
        return features, labels
    
    print(f"\nBalancing dataset:")
    print(f"  Original: {defect_count:,} defect, {normal_count:,} normal")
    
    # Get indices
    normal_indices = np.where(labels == NORMAL_CLASS_ID)[0]
    defect_indices = np.where(labels != NORMAL_CLASS_ID)[0]
    
    # Random subsample normal patches
    np.random.seed(RANDOM_SEED)
    sampled_normal_indices = np.random.choice(normal_indices, size=max_normal, replace=False)
    
    # Combine indices
    all_indices = np.concatenate([defect_indices, sampled_normal_indices])
    np.random.shuffle(all_indices)
    
    balanced_features = features[all_indices]
    balanced_labels = labels[all_indices]
    
    print(f"  Balanced: {len(defect_indices):,} defect, {len(sampled_normal_indices):,} normal")
    
    return balanced_features, balanced_labels


def compute_metrics(y_true, y_pred, split_name=""):
    """Compute metrics for multi-class classification."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


def print_metrics_table(train_metrics, val_metrics, test_metrics):
    """Print formatted metrics table."""
    print(f"\n{'='*70}")
    print("METRICS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Train':>12} {'Val':>12} {'Test':>12}")
    print(f"{'-'*70}")
    
    for key in train_metrics.keys():
        train_val = train_metrics[key] * 100
        val_val = val_metrics[key] * 100
        test_val = test_metrics[key] * 100
        print(f"{key:<25} {train_val:>11.2f}% {val_val:>11.2f}% {test_val:>11.2f}%")
    
    print(f"{'='*70}")


def train_patch_model():
    """Main training function."""
    print(f"\n{'='*60}")
    print("PATCH-LEVEL XGBOOST TRAINING")
    print(f"Classes: {PATCH_CLASS_NAMES}")
    print(f"{'='*60}")
    
    # Load features
    print("\nLoading patch features...")
    splits = load_all_patch_features()
    
    X_train = splits['train']['features']
    y_train = splits['train']['labels']
    X_val = splits['val']['features']
    y_val = splits['val']['labels']
    X_test = splits['test']['features']
    y_test = splits['test']['labels']
    
    # Print class distribution
    print("\nOriginal class distribution (train):")
    for cls_id, cls_name in PATCH_ID_TO_CLASS.items():
        count = np.sum(y_train == cls_id)
        pct = 100 * count / len(y_train)
        print(f"  - {cls_name}: {count:,} patches ({pct:.1f}%)")
    
    # Balance training data
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, max_normal_ratio=2.0)
    
    # Fit scaler on balanced training data
    print("\nFitting scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute sample weights for remaining imbalance
    sample_weights = compute_sample_weight('balanced', y_train_balanced)
    
    # Update XGBoost params for 4 classes
    params = XGBOOST_PARAMS.copy()
    params['num_class'] = PATCH_NUM_CLASSES
    params.pop('early_stopping_rounds', None)
    
    # Train model
    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(**params)
    
    model.fit(
        X_train_scaled, y_train_balanced,
        sample_weight=sample_weights,
        eval_set=[(X_val_scaled, y_val)],
        verbose=True
    )
    
    # Predictions
    print("\nGenerating predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Compute metrics
    train_metrics = compute_metrics(y_train_balanced, y_train_pred, "train")
    val_metrics = compute_metrics(y_val, y_val_pred, "val")
    test_metrics = compute_metrics(y_test, y_test_pred, "test")
    
    print_metrics_table(train_metrics, val_metrics, test_metrics)
    
    # Classification report
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=PATCH_CLASS_NAMES, 
                                zero_division=0))
    
    # Save model and scaler
    model_path = MODELS_DIR / "xgb_patch_multiclass.joblib"
    scaler_path = MODELS_DIR / "patch_scaler.joblib"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save report
    report_path = EVALUATION_DIR / "patch_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("PATCH-LEVEL CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(classification_report(y_test, y_test_pred,
                                      target_names=PATCH_CLASS_NAMES,
                                      zero_division=0))
    
    print(f"✓ Report saved to: {report_path}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    
    return model, scaler, test_metrics


if __name__ == "__main__":
    model, scaler, metrics = train_patch_model()
