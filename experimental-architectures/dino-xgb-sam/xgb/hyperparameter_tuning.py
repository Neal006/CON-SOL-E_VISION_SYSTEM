"""
XGBoost Hyperparameter Tuning with Optuna and MLflow
Performs Bayesian optimization to find optimal XGBoost parameters.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
import optuna
from optuna.integration.mlflow import MLflowCallback
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    balanced_accuracy_score, make_scorer
)
from xgboost import XGBClassifier
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURES_DIR, MODELS_DIR, EVALUATION_DIR, MLRUNS_DIR,
    CLASS_NAMES, NUM_CLASSES, RANDOM_SEED,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, OPTUNA_CV_FOLDS,
    OPTUNA_STUDY_NAME, OPTUNA_DIRECTION,
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
)
from dino.feature_extractor import load_features_from_parquet


def load_all_features() -> Dict[str, Dict]:
    """Load features for all splits."""
    print(f"\n{'='*60}")
    print("LOADING FEATURES")
    print(f"{'='*60}")
    
    splits = {}
    for split in ['train', 'val', 'test']:
        parquet_path = FEATURES_DIR / f"{split}_features.parquet"
        if not parquet_path.exists():
            # Try patch features
            parquet_path = FEATURES_DIR / f"patch_{split}_features.parquet"
        
        if parquet_path.exists():
            features, labels, paths = load_features_from_parquet(parquet_path)
            splits[split] = {
                'features': features,
                'labels': labels,
                'paths': paths
            }
            print(f"✓ Loaded {split}: {features.shape[0]} samples, {features.shape[1]} features")
        else:
            print(f"⚠ Features not found for {split}")
    
    return splits


def create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler: StandardScaler,
    cv_folds: int = OPTUNA_CV_FOLDS
):
    """
    Create Optuna objective function for XGBoost optimization.
    
    Uses validation F1 score as the optimization target.
    """
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbosity': 0,
            'objective': 'multi:softprob',
            'num_class': NUM_CLASSES,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
        }
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model
        model = XGBClassifier(**params)
        
        # Use early stopping with validation set
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        # Predict on validation set
        y_val_pred = model.predict(X_val_scaled)
        
        # Calculate F1 score (macro average)
        f1 = f1_score(y_val, y_val_pred, average='macro')
        
        # Log additional metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred, average='macro')
        precision = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
        
        # Store in trial user attributes
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('balanced_accuracy', balanced_acc)
        trial.set_user_attr('recall_macro', recall)
        trial.set_user_attr('precision_macro', precision)
        trial.set_user_attr('f1_macro', f1)
        
        return f1
    
    return objective


def run_hyperparameter_tuning(
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
    cv_folds: int = OPTUNA_CV_FOLDS,
    study_name: str = OPTUNA_STUDY_NAME,
    use_cv: bool = False
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """
    Run hyperparameter tuning with Optuna and MLflow tracking.
    
    Args:
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds
        cv_folds: Number of cross-validation folds
        study_name: Name for the Optuna study
        use_cv: Use cross-validation (slower but more robust)
    
    Returns:
        study: Optuna study object
        best_params: Best hyperparameters found
    """
    print(f"\n{'='*60}")
    print("XGBOOST HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    print(f"Trials: {n_trials}")
    print(f"Timeout: {timeout}s")
    print(f"Cross-validation folds: {cv_folds}")
    print(f"Study name: {study_name}")
    print(f"{'='*60}\n")
    
    # Load data
    splits = load_all_features()
    
    X_train = splits['train']['features']
    y_train = splits['train']['labels']
    X_val = splits['val']['features']
    y_val = splits['val']['labels']
    X_test = splits['test']['features']
    y_test = splits['test']['labels']
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = f"{MLFLOW_EXPERIMENT_NAME}_tuning"
    mlflow.set_experiment(experiment_name)
    
    # Create MLflow callback for Optuna
    mlflow_callback = MLflowCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        metric_name="recall_macro",
        create_experiment=False,
        mlflow_kwargs={"experiment_id": mlflow.get_experiment_by_name(experiment_name).experiment_id}
    )
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize F1 score
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    # Create objective function
    objective = create_objective(X_train, y_train, X_val, y_val, scaler, cv_folds)
    
    # Run optimization
    print("Starting optimization...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[mlflow_callback],
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best F1 Score: {best_value:.4f}")
    print(f"\nBest Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return study, best_params


def train_with_best_params(
    best_params: Dict[str, Any],
    save_model: bool = True
) -> Tuple[XGBClassifier, StandardScaler, Dict[str, float]]:
    """
    Train final model with best hyperparameters.
    
    Args:
        best_params: Best hyperparameters from tuning
        save_model: Whether to save the model
    
    Returns:
        model: Trained XGBClassifier
        scaler: Fitted StandardScaler
        metrics: Evaluation metrics
    """
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print(f"{'='*60}")
    
    # Load data
    splits = load_all_features()
    
    X_train = splits['train']['features']
    y_train = splits['train']['labels']
    X_val = splits['val']['features']
    y_val = splits['val']['labels']
    X_test = splits['test']['features']
    y_test = splits['test']['labels']
    
    # Prepare model parameters
    model_params = best_params.copy()
    model_params.update({
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbosity': 1,
        'objective': 'multi:softprob',
        'num_class': NUM_CLASSES,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
    })
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining model...")
    model = XGBClassifier(**model_params)
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=True
    )
    
    # Evaluate
    y_test_pred = model.predict(X_test_scaled)
    
    metrics = {
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'test_f1_macro': f1_score(y_test, y_test_pred, average='macro'),
        'test_recall_macro': recall_score(y_test, y_test_pred, average='macro'),
        'test_precision_macro': precision_score(y_test, y_test_pred, average='macro', zero_division=0),
    }
    
    # Per-class metrics
    recall_per_class = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(recall_per_class):
            metrics[f'test_recall_{class_name.lower()}'] = recall_per_class[i]
    
    print(f"\n{'='*60}")
    print("TEST SET METRICS")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    if save_model:
        # Save model and scaler
        model_path = MODELS_DIR / "xgb_tuned.joblib"
        scaler_path = MODELS_DIR / "scaler_tuned.joblib"
        params_path = MODELS_DIR / "best_params.json"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Scaler saved to: {scaler_path}")
        print(f"✓ Best params saved to: {params_path}")
        
        # Log to MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name=f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(scaler_path))
            mlflow.log_artifact(str(params_path))
            mlflow.xgboost.log_model(model, "xgb_tuned_model")
    
    return model, scaler, metrics


def plot_optimization_history(study: optuna.Study, save_path: Optional[Path] = None):
    """Plot optimization history."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Optimization history
        trials = [t.number for t in study.trials if t.value is not None]
        values = [t.value for t in study.trials if t.value is not None]
        best_values = np.maximum.accumulate(values)
        
        axes[0].plot(trials, values, 'o-', alpha=0.5, label='Trial Value')
        axes[0].plot(trials, best_values, 'r-', linewidth=2, label='Best Value')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('F1 Score (Macro)')
        axes[0].set_title('Optimization History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Parameter importance
        try:
            importances = optuna.importance.get_param_importances(study)
            params = list(importances.keys())[:10]  # Top 10
            values = [importances[p] for p in params]
            
            axes[1].barh(params, values, color='steelblue')
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Hyperparameter Importance')
            axes[1].grid(True, alpha=0.3, axis='x')
        except Exception as e:
            axes[1].text(0.5, 0.5, f"Could not compute importance:\n{e}",
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Optimization plot saved to: {save_path}")
        
        plt.close()
        
    except ImportError:
        print("⚠ matplotlib not available for plotting")


def main():
    """Main entry point for hyperparameter tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='XGBoost Hyperparameter Tuning')
    parser.add_argument('--n-trials', type=int, default=OPTUNA_N_TRIALS,
                       help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=OPTUNA_TIMEOUT,
                       help='Maximum time in seconds')
    parser.add_argument('--cv-folds', type=int, default=OPTUNA_CV_FOLDS,
                       help='Cross-validation folds')
    parser.add_argument('--study-name', type=str, default=OPTUNA_STUDY_NAME,
                       help='Optuna study name')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the final model')
    
    args = parser.parse_args()
    
    # Run hyperparameter tuning
    study, best_params = run_hyperparameter_tuning(
        n_trials=args.n_trials,
        timeout=args.timeout,
        cv_folds=args.cv_folds,
        study_name=args.study_name
    )
    
    # Plot optimization history
    plot_path = EVALUATION_DIR / "optimization_history.png"
    plot_optimization_history(study, save_path=plot_path)
    
    # Train final model with best parameters
    model, scaler, metrics = train_with_best_params(
        best_params,
        save_model=not args.no_save
    )
    
    print(f"\n{'='*60}")
    print("HYPERPARAMETER TUNING COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
