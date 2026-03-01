"""
XGBoost Multi-Class Training with MLflow Experiment Tracking
Trains a classifier for Scratch, Dust, and RunDown defect detection.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURES_DIR, MODELS_DIR, EVALUATION_DIR, MLRUNS_DIR,
    CLASS_NAMES, ID_TO_CLASS, NUM_CLASSES,
    XGBOOST_PARAMS, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
)
from dino.feature_extractor import load_features_from_parquet


def load_all_features():
    """Load features for all splits."""
    print(f"\n{'='*60}")
    print("LOADING FEATURES FROM PARQUET")
    print(f"{'='*60}")
    
    splits = {}
    for split in ['train', 'val', 'test']:
        parquet_path = FEATURES_DIR / f"{split}_features.parquet"
        features, labels, paths = load_features_from_parquet(parquet_path)
        splits[split] = {
            'features': features,
            'labels': labels,
            'paths': paths
        }
        print(f"✓ Loaded {split}: {features.shape[0]} samples, {features.shape[1]} features")
    
    return splits


def compute_class_weights(labels):
    """Compute balanced class weights for imbalanced dataset."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = {int(c): total / (len(unique) * count) for c, count in zip(unique, counts)}
    return weights


def compute_metrics(y_true, y_pred, split_name=""):
    """Compute comprehensive metrics for multi-class classification."""
    metrics = {
        f'{split_name}_accuracy': accuracy_score(y_true, y_pred) * 100,
        f'{split_name}_recall_macro': recall_score(y_true, y_pred, average='macro') * 100,
        f'{split_name}_precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        f'{split_name}_f1_macro': f1_score(y_true, y_pred, average='macro') * 100,
    }
    
    # Per-class recall
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(recall_per_class):
            metrics[f'{split_name}_recall_{class_name.lower()}'] = recall_per_class[i] * 100
    
    return metrics


def print_metrics_table(train_metrics, val_metrics, test_metrics):
    """Print a formatted metrics comparison table."""
    print(f"\n{'='*70}")
    print("EVALUATION METRICS")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Training':>12} {'Validation':>12} {'Test':>12}")
    print("-" * 70)
    
    # Overall metrics
    for metric in ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']:
        train_val = train_metrics.get(f'train_{metric}', 0)
        val_val = val_metrics.get(f'val_{metric}', 0)
        test_val = test_metrics.get(f'test_{metric}', 0)
        print(f"{metric.replace('_', ' ').title():<25} {train_val:>11.1f}% {val_val:>11.1f}% {test_val:>11.1f}%")
    
    print("-" * 70)
    print("Per-Class Recall:")
    for class_name in CLASS_NAMES:
        metric_key = f'recall_{class_name.lower()}'
        train_val = train_metrics.get(f'train_{metric_key}', 0)
        val_val = val_metrics.get(f'val_{metric_key}', 0)
        test_val = test_metrics.get(f'test_{metric_key}', 0)
        print(f"  {class_name:<23} {train_val:>11.1f}% {val_val:>11.1f}% {test_val:>11.1f}%")
    
    print("=" * 70)


def save_classification_report(y_true, y_pred, split_name, output_dir):
    """Save classification report to file."""
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = output_dir / f"{split_name}_classification_report.csv"
    report_df.to_csv(report_path)
    return report_path


def train_model():
    """Main training function with MLflow tracking."""
    
    print(f"\n{'='*60}")
    print("XGBOOST MULTI-CLASS TRAINING WITH MLFLOW")
    print(f"{'='*60}")
    
    # Load features
    splits = load_all_features()
    
    X_train = splits['train']['features']
    y_train = splits['train']['labels']
    X_val = splits['val']['features']
    y_val = splits['val']['labels']
    X_test = splits['test']['features']
    y_test = splits['test']['labels']
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(y_train)
    sample_weights = np.array([class_weights[int(y)] for y in y_train])
    
    print(f"\nClass weights (for imbalanced data):")
    for class_id, weight in class_weights.items():
        print(f"  - {CLASS_NAMES[class_id]}: {weight:.3f}")
    
    # Scale features
    print(f"\nScaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    print(f"\n{'='*60}")
    print("TRAINING XGBOOST CLASSIFIER")
    print(f"{'='*60}")
    
    with mlflow.start_run(run_name=f"xgb_multiclass_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_params(XGBOOST_PARAMS)
        mlflow.log_param("num_train_samples", len(X_train))
        mlflow.log_param("num_val_samples", len(X_val))
        mlflow.log_param("num_test_samples", len(X_test))
        mlflow.log_param("feature_dim", X_train.shape[1])
        
        # Initialize model
        model_params = XGBOOST_PARAMS.copy()
        early_stopping = model_params.pop('early_stopping_rounds', 20)
        
        model = XGBClassifier(**model_params)
        
        # Train with early stopping
        print(f"\nTraining with early stopping (patience={early_stopping})...")
        model.fit(
            X_train_scaled, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val_scaled, y_val)],
            verbose=True
        )
        
        # Get predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Compute metrics
        train_metrics = compute_metrics(y_train, y_train_pred, "train")
        val_metrics = compute_metrics(y_val, y_val_pred, "val")
        test_metrics = compute_metrics(y_test, y_test_pred, "test")
        
        # Log metrics to MLflow
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)
        
        # Print metrics table
        print_metrics_table(train_metrics, val_metrics, test_metrics)
        
        # Save confusion matrices
        for split_name, y_true, y_pred in [
            ('train', y_train, y_train_pred),
            ('val', y_val, y_val_pred),
            ('test', y_test, y_test_pred)
        ]:
            cm = confusion_matrix(y_true, y_pred)
            cm_path = EVALUATION_DIR / f"{split_name}_confusion_matrix.csv"
            pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(cm_path)
            mlflow.log_artifact(str(cm_path))
            
            # Save classification report
            report_path = save_classification_report(y_true, y_pred, split_name, EVALUATION_DIR)
            mlflow.log_artifact(str(report_path))
        
        # Save model
        model_path = MODELS_DIR / "xgb_multiclass.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))
        
        # Log model using MLflow's XGBoost integration
        mlflow.xgboost.log_model(model, "xgb_model")
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ MLflow run logged to: {MLRUNS_DIR}")
        
        # Save predictions for analysis
        predictions = {
            'train': {'y_true': y_train.tolist(), 'y_pred': y_train_pred.tolist(), 'paths': splits['train']['paths']},
            'val': {'y_true': y_val.tolist(), 'y_pred': y_val_pred.tolist(), 'paths': splits['val']['paths']},
            'test': {'y_true': y_test.tolist(), 'y_pred': y_test_pred.tolist(), 'paths': splits['test']['paths']}
        }
        predictions_path = EVALUATION_DIR / "predictions.json"
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        mlflow.log_artifact(str(predictions_path))
        
        print(f"✓ Predictions saved to: {predictions_path}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    
    return model, scaler, test_metrics


if __name__ == "__main__":
    model, scaler, metrics = train_model()
