import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import optuna
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
try:
    _current_dir = Path(__file__).parent.parent
except NameError:
    _current_dir = Path(os.getcwd())
sys.path.insert(0, str(_current_dir))
from config import (
    FEATURES_DIR, MODELS_DIR, EVALUATION_DIR,
    PATCH_CLASS_NAMES, PATCH_NUM_CLASSES, RANDOM_SEED,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, OPTUNA_CV_FOLDS,
    OPTUNA_STUDY_NAME, OPTUNA_DIRECTION
)
from dino.patch_feature_extractor import load_patch_features_from_parquet

def load_all_features() -> Dict[str, Dict]:
    print("LOADING FEATURES")
    splits = {}
    for split in ['train', 'val', 'test']:
        parquet_path = FEATURES_DIR / f"patch_{split}_features.parquet"        
        if parquet_path.exists():
            features, labels, refs = load_patch_features_from_parquet(parquet_path)
            splits[split] = {'features': features, 'labels': labels, 'refs': refs}
            print(f"Loaded {split}: {features.shape[0]:,} patches, {features.shape[1]} features")
        else:
            print(f"Features not found: {parquet_path}")
    return splits

def create_objective(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, scaler: StandardScaler, cv_folds: int = OPTUNA_CV_FOLDS):
    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbosity': 0,
            'objective': 'multi:softprob',
            'num_class': PATCH_NUM_CLASSES,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
        }
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model = XGBClassifier(**params)
        model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
        y_val_pred = model.predict(X_val_scaled)
        f1 = f1_score(y_val, y_val_pred, average='macro')
        accuracy = accuracy_score(y_val, y_val_pred)
        balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred, average='macro')
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('balanced_accuracy', balanced_acc)
        trial.set_user_attr('recall_macro', recall)
        trial.set_user_attr('f1_macro', f1)
        return recall
    return objective

def run_hyperparameter_tuning(n_trials: int = OPTUNA_N_TRIALS, timeout: int = OPTUNA_TIMEOUT, cv_folds: int = OPTUNA_CV_FOLDS, study_name: str = OPTUNA_STUDY_NAME, use_cv: bool = False, max_samples: int = 100000) -> Tuple[optuna.Study, Dict[str, Any]]:
    print("XGBOOST HYPERPARAMETER TUNING")
    print(f"Trials: {n_trials}, Timeout: {timeout}s, CV folds: {cv_folds}")
    splits = load_all_features()
    X_train = splits['train']['features']
    y_train = splits['train']['labels']
    X_val = splits['val']['features']
    y_val = splits['val']['labels']
    X_test = splits['test']['features']
    y_test = splits['test']['labels']
    # Subsample for faster tuning
    if len(X_train) > max_samples:
        print(f"Subsampling training data: {len(X_train):,} -> {max_samples:,} samples for tuning")
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(X_train), size=max_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
    if len(X_val) > max_samples // 4:
        val_size = max_samples // 4
        print(f"Subsampling validation data: {len(X_val):,} -> {val_size:,} samples for tuning")
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(X_val), size=val_size, replace=False)
        X_val = X_val[indices]
        y_val = y_val[indices]
    scaler = StandardScaler()
    study = optuna.create_study(study_name=study_name, direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED), pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    objective = create_objective(X_train, y_train, X_val, y_val, scaler, cv_folds)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    best_params = study.best_params
    best_value = study.best_value
    print("OPTIMIZATION COMPLETE")
    print(f"Best Recall Score: {best_value:.4f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    return study, best_params

def train_with_best_params(best_params: Dict[str, Any], save_model: bool = True) -> Tuple[XGBClassifier, StandardScaler, Dict[str, float]]:
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    splits = load_all_features()
    X_train = splits['train']['features']
    y_train = splits['train']['labels']
    X_val = splits['val']['features']
    y_val = splits['val']['labels']
    X_test = splits['test']['features']
    y_test = splits['test']['labels']
    model_params = best_params.copy()
    model_params.update({'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbosity': 1, 'objective': 'multi:softprob', 'num_class': PATCH_NUM_CLASSES, 'eval_metric': 'mlogloss', 'use_label_encoder': False})
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    model = XGBClassifier(**model_params)
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=True)
    y_test_pred = model.predict(X_test_scaled)
    metrics = {
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'test_recall_macro': recall_score(y_test, y_test_pred, average='macro', zero_division=0),
        'test_precision_macro': precision_score(y_test, y_test_pred, average='macro', zero_division=0),
    }
    recall_per_class = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    for i, class_name in enumerate(PATCH_CLASS_NAMES):
        if i < len(recall_per_class):
            metrics[f'recall_{class_name}'] = recall_per_class[i]
    print("TEST SET METRICS")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    if save_model:
        model_path = MODELS_DIR / "xgb_tuned.joblib"
        scaler_path = MODELS_DIR / "scaler_tuned.joblib"
        params_path = MODELS_DIR / "best_params.json"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Best params saved to: {params_path}")
    return model, scaler, metrics

def plot_optimization_history(study: optuna.Study, save_path: Optional[Path] = None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        trials = [t.number for t in study.trials if t.value is not None]
        values = [t.value for t in study.trials if t.value is not None]
        best_values = np.maximum.accumulate(values)
        axes[0].plot(trials, values, 'o-', alpha=0.5, label='Trial value')
        axes[0].plot(trials, best_values, 'r-', linewidth=2, label='Best value')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Recall (macro)')
        axes[0].set_title('Optimization History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        try:
            importances = optuna.importance.get_param_importances(study)
            params = list(importances.keys())[:10]
            values = [importances[p] for p in params]
            axes[1].barh(params, values)
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Parameter Importance')
        except:
            axes[1].text(0.5, 0.5, 'Not enough trials for importance', ha='center', va='center')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        plt.close()
    except Exception as e:
        print(f"Could not create plot: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='XGBoost Hyperparameter Tuning')
    parser.add_argument('--n-trials', type=int, default=OPTUNA_N_TRIALS, help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=OPTUNA_TIMEOUT, help='Timeout in seconds')
    parser.add_argument('--cv-folds', type=int, default=OPTUNA_CV_FOLDS, help='Cross-validation folds')
    parser.add_argument('--study-name', type=str, default=OPTUNA_STUDY_NAME, help='Optuna study name')
    parser.add_argument('--no-save', action='store_true', help='Do not save the final model')
    args = parser.parse_args()
    study, best_params = run_hyperparameter_tuning(n_trials=args.n_trials, timeout=args.timeout, cv_folds=args.cv_folds, study_name=args.study_name)
    plot_path = EVALUATION_DIR / "optimization_history.png"
    plot_optimization_history(study, save_path=plot_path)
    model, scaler, metrics = train_with_best_params(best_params, save_model=not args.no_save)    
    print("HYPERPARAMETER TUNING COMPLETE")

if __name__ == "__main__":
    main()
