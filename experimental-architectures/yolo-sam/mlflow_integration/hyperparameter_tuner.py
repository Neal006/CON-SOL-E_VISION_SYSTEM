"""
Hyperparameter Tuner using Optuna + MLflow.

Features:
- Automated YOLO hyperparameter search
- Multi-objective optimization (Recall + Precision)
- Pruning for early stopping of bad trials
- Best model auto-registration
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class HyperparameterTuner:
    """
    Hyperparameter tuner using Optuna with MLflow integration.
    
    Supports:
    - YOLO detection tuning (Maximize Recall >95%)
    - SAM threshold tuning (Maximize Dice)
    - Post-processing tuning (Minimize Area Error)
    """
    
    def __init__(
        self,
        experiment_name: str = "yolo_sam_optimization",
        n_trials: int = 50,
        timeout: int = 43200,  # 12 hours
        tracking_uri: str = "mlruns",
        pruning: bool = True
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            experiment_name: Name for the optimization study
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
            tracking_uri: MLflow tracking URI
            pruning: Whether to enable trial pruning
        """
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.timeout = timeout
        self.tracking_uri = tracking_uri
        self.pruning = pruning
        
        self._optuna = None
        self._mlflow = None
        self._study = None
        self._objective_fn = None
        
        self._setup()
    
    def _setup(self):
        """Setup Optuna and MLflow."""
        try:
            import optuna
            self._optuna = optuna
            
            # Suppress Optuna logs
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
        except ImportError:
            print("Warning: Optuna not installed. Install with: pip install optuna")
            self._optuna = None
        
        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
        except ImportError:
            print("Warning: MLflow not installed.")
            self._mlflow = None
    
    def define_yolo_search_space(self, trial) -> Dict:
        """
        Define search space for YOLO hyperparameters.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Dictionary of hyperparameters for this trial
        """
        params = {
            # Learning rate (log uniform for better sampling)
            "lr0": trial.suggest_float("lr0", 1e-5, 1e-2, log=True),
            
            # Confidence threshold (recall-first, lower values)
            "conf": trial.suggest_float("conf", 0.2, 0.5),
            
            # NMS IoU threshold
            "nms_iou": trial.suggest_float("nms_iou", 0.5, 0.8),
            
            # Batch size
            "batch": trial.suggest_categorical("batch", [4, 8, 16, 32]),
            
            # Optimizer
            "optimizer": trial.suggest_categorical(
                "optimizer",
                ["Adam", "AdamW", "SGD"]
            ),
            
            # Augmentation settings
            "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
            "mixup": trial.suggest_float("mixup", 0.0, 0.5),
            "hsv_v": trial.suggest_float("hsv_v", 0.0, 0.5),  # Value augmentation
            
            # Geometric augmentations
            "degrees": trial.suggest_float("degrees", 0.0, 20.0),
            "translate": trial.suggest_float("translate", 0.0, 0.3),
            "scale": trial.suggest_float("scale", 0.0, 0.9),
            "flipud": trial.suggest_float("flipud", 0.0, 0.5),
            "fliplr": trial.suggest_float("fliplr", 0.0, 0.5),
        }
        
        return params
    
    def define_sam_search_space(self, trial) -> Dict:
        """
        Define search space for SAM parameters.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Dictionary of SAM hyperparameters
        """
        params = {
            "mask_threshold": trial.suggest_float("mask_threshold", 0.3, 0.7),
            "pred_iou_thresh": trial.suggest_float("pred_iou_thresh", 0.7, 0.95),
            "stability_score_thresh": trial.suggest_float("stability_score_thresh", 0.8, 0.99),
        }
        
        return params
    
    def define_postprocess_search_space(self, trial) -> Dict:
        """
        Define search space for post-processing parameters.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Dictionary of post-processing hyperparameters
        """
        params = {
            "min_component_size": trial.suggest_int("min_component_size", 10, 200),
            "morph_kernel_size": trial.suggest_int("morph_kernel_size", 1, 7, step=2),
            "binary_threshold": trial.suggest_float("binary_threshold", 0.3, 0.7),
        }
        
        return params
    
    def create_objective(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        metric: str = "recall",
        search_space_fn: Callable = None
    ) -> Callable:
        """
        Create objective function for optimization.
        
        Args:
            train_fn: Function to train model, signature: train_fn(params) -> model
            eval_fn: Function to evaluate model, signature: eval_fn(model) -> metrics_dict
            metric: Metric to optimize (default: recall)
            search_space_fn: Function to define search space
        
        Returns:
            Objective function for Optuna
        """
        search_space_fn = search_space_fn or self.define_yolo_search_space
        
        def objective(trial):
            # Get hyperparameters
            params = search_space_fn(trial)
            
            # Log to MLflow if available
            if self._mlflow is not None:
                with self._mlflow.start_run(nested=True):
                    self._mlflow.log_params(params)
                    
                    try:
                        # Train
                        model = train_fn(params)
                        
                        # Evaluate
                        metrics = eval_fn(model)
                        
                        # Log metrics
                        self._mlflow.log_metrics(metrics)
                        
                        # Get target metric
                        value = metrics.get(metric, 0.0)
                        
                        # Report for pruning
                        trial.report(value, step=0)
                        
                        if trial.should_prune():
                            raise self._optuna.TrialPruned()
                        
                        return value
                        
                    except Exception as e:
                        self._mlflow.log_param("error", str(e))
                        raise
            else:
                # Without MLflow
                model = train_fn(params)
                metrics = eval_fn(model)
                return metrics.get(metric, 0.0)
        
        self._objective_fn = objective
        return objective
    
    def run_optimization(
        self,
        objective: Callable = None,
        direction: str = "maximize",
        study_name: str = None
    ) -> Dict:
        """
        Run hyperparameter optimization.
        
        Args:
            objective: Objective function (uses stored if None)
            direction: "maximize" or "minimize"
            study_name: Optional study name
        
        Returns:
            Dictionary with best parameters and results
        """
        if self._optuna is None:
            raise RuntimeError("Optuna not installed")
        
        objective = objective or self._objective_fn
        if objective is None:
            raise ValueError("No objective function defined")
        
        study_name = study_name or self.experiment_name
        
        # Create sampler
        sampler = self._optuna.samplers.TPESampler(seed=42)
        
        # Create pruner if enabled
        if self.pruning:
            pruner = self._optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=0
            )
        else:
            pruner = self._optuna.pruners.NopPruner()
        
        # Create study
        self._study = self._optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        # Run optimization
        self._study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get results
        best_trial = self._study.best_trial
        
        return {
            "best_params": best_trial.params,
            "best_value": best_trial.value,
            "n_trials": len(self._study.trials),
            "n_completed": len([t for t in self._study.trials 
                               if t.state == self._optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in self._study.trials 
                            if t.state == self._optuna.trial.TrialState.PRUNED])
        }
    
    def get_best_params(self) -> Dict:
        """Get best parameters from completed study."""
        if self._study is None:
            raise ValueError("No study available. Run optimization first.")
        return self._study.best_params
    
    def get_best_value(self) -> float:
        """Get best metric value from completed study."""
        if self._study is None:
            raise ValueError("No study available. Run optimization first.")
        return self._study.best_value
    
    def export_study_results(self, output_path: str):
        """
        Export study results to CSV.
        
        Args:
            output_path: Path for CSV file
        """
        if self._study is None:
            raise ValueError("No study available")
        
        df = self._study.trials_dataframe()
        df.to_csv(output_path, index=False)
    
    def get_importance(self) -> Dict[str, float]:
        """
        Get hyperparameter importance.
        
        Returns:
            Dictionary of parameter name to importance score
        """
        if self._study is None or self._optuna is None:
            return {}
        
        try:
            importance = self._optuna.importance.get_param_importances(self._study)
            return importance
        except Exception:
            return {}
    
    def visualize_optimization(self, output_dir: str = None):
        """
        Create optimization visualizations.
        
        Args:
            output_dir: Directory to save plots
        """
        if self._study is None or self._optuna is None:
            return
        
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate
            )
            
            output_dir = Path(output_dir) if output_dir else Path(".")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Optimization history
            fig = plot_optimization_history(self._study)
            fig.write_image(str(output_dir / "optimization_history.png"))
            
            # Parameter importance
            fig = plot_param_importances(self._study)
            fig.write_image(str(output_dir / "param_importance.png"))
            
            # Parallel coordinate
            fig = plot_parallel_coordinate(self._study)
            fig.write_image(str(output_dir / "parallel_coordinate.png"))
            
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
    
    def get_study(self):
        """Get the Optuna study object."""
        return self._study


def quick_tune(
    train_fn: Callable,
    eval_fn: Callable,
    n_trials: int = 20,
    metric: str = "recall",
    direction: str = "maximize"
) -> Dict:
    """
    Quick hyperparameter tuning with minimal setup.
    
    Args:
        train_fn: Training function
        eval_fn: Evaluation function
        n_trials: Number of trials
        metric: Metric to optimize
        direction: Optimization direction
    
    Returns:
        Best parameters dictionary
    """
    tuner = HyperparameterTuner(n_trials=n_trials)
    tuner.create_objective(train_fn, eval_fn, metric)
    results = tuner.run_optimization(direction=direction)
    
    return results
