"""
Training Callbacks for MLflow Integration.

Provides callbacks for:
- Ultralytics YOLO training
- Custom training loops
- Epoch-level metric logging
"""

from typing import Any, Dict, Optional


class MLflowCallback:
    """
    Callback for integrating MLflow with training loops.
    
    Can be used with:
    - Ultralytics YOLO training
    - Custom PyTorch training loops
    """
    
    def __init__(
        self,
        experiment_name: str = "yolo_sam_defect_detection",
        tracking_uri: str = "mlruns",
        run_name: str = None,
        log_every_n_epochs: int = 1
    ):
        """
        Initialize callback.
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
            run_name: Optional run name
            log_every_n_epochs: Log frequency
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.log_every_n_epochs = log_every_n_epochs
        
        self._tracker = None
        self._epoch = 0
        
    def _ensure_tracker(self):
        """Ensure tracker is initialized."""
        if self._tracker is None:
            from .experiment_tracker import ExperimentTracker
            
            self._tracker = ExperimentTracker(
                experiment_name=self.experiment_name,
                tracking_uri=self.tracking_uri
            )
    
    def on_train_start(self, trainer=None, params: Dict = None):
        """Called at training start."""
        self._ensure_tracker()
        self._tracker.start_run(run_name=self.run_name)
        
        if params:
            self._tracker.log_params(params)
        
        if trainer and hasattr(trainer, 'args'):
            # Log Ultralytics trainer args
            trainer_params = {
                k: v for k, v in vars(trainer.args).items()
                if isinstance(v, (int, float, str, bool))
            }
            self._tracker.log_params(trainer_params)
    
    def on_train_epoch_end(
        self,
        trainer=None,
        metrics: Dict = None,
        epoch: int = None
    ):
        """Called at end of each training epoch."""
        self._ensure_tracker()
        
        self._epoch = epoch if epoch is not None else self._epoch + 1
        
        if self._epoch % self.log_every_n_epochs != 0:
            return
        
        # Extract metrics
        log_metrics = {}
        
        if trainer and hasattr(trainer, 'metrics'):
            # Ultralytics trainer metrics
            for key, value in trainer.metrics.items():
                if isinstance(value, (int, float)):
                    log_metrics[f"train_{key}"] = value
        
        if metrics:
            log_metrics.update(metrics)
        
        if trainer and hasattr(trainer, 'loss'):
            log_metrics["train_loss"] = float(trainer.loss)
        
        self._tracker.log_metrics(log_metrics, step=self._epoch)
    
    def on_val_end(
        self,
        validator=None,
        metrics: Dict = None
    ):
        """Called at end of validation."""
        self._ensure_tracker()
        
        log_metrics = {}
        
        if validator and hasattr(validator, 'metrics'):
            # Ultralytics validator metrics
            for key, value in validator.metrics.items():
                if isinstance(value, (int, float)):
                    log_metrics[f"val_{key}"] = value
        
        if metrics:
            log_metrics.update(metrics)
        
        self._tracker.log_metrics(log_metrics, step=self._epoch)
    
    def on_train_end(
        self,
        trainer=None,
        model_path: str = None,
        final_metrics: Dict = None
    ):
        """Called at end of training."""
        self._ensure_tracker()
        
        # Log final metrics
        if final_metrics:
            self._tracker.log_metrics(final_metrics)
        
        # Log model artifact
        if model_path:
            self._tracker.log_artifact(model_path, "models")
        
        # Log best model from Ultralytics
        if trainer and hasattr(trainer, 'best'):
            self._tracker.log_artifact(str(trainer.best), "models")
        
        self._tracker.end_run()
    
    def on_batch_end(self, batch: int, logs: Dict = None):
        """Called at end of each batch (optional)."""
        pass  # Too frequent for MLflow, skip by default
    
    def log_confusion_matrix(
        self,
        confusion_matrix,
        class_names,
        artifact_name: str = "confusion_matrix.png"
    ):
        """Log confusion matrix."""
        self._ensure_tracker()
        self._tracker.log_confusion_matrix(
            confusion_matrix, class_names, artifact_name
        )
    
    def log_predictions(
        self,
        images,
        predictions,
        ground_truths=None
    ):
        """Log sample predictions."""
        self._ensure_tracker()
        self._tracker.log_sample_predictions(images, predictions, ground_truths)


def create_yolo_callback(
    experiment_name: str = "yolo_defect_detection",
    tracking_uri: str = "mlruns"
) -> Dict:
    """
    Create callbacks dictionary for Ultralytics YOLO.
    
    Usage:
        from ultralytics import YOLO
        
        callbacks = create_yolo_callback()
        model = YOLO('yolov8n.pt')
        model.train(data='data.yaml', callbacks=callbacks)
    
    Args:
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking URI
    
    Returns:
        Dictionary of callback functions
    """
    callback = MLflowCallback(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri
    )
    
    return {
        "on_train_start": callback.on_train_start,
        "on_train_epoch_end": lambda trainer: callback.on_train_epoch_end(trainer=trainer),
        "on_val_end": lambda validator: callback.on_val_end(validator=validator),
        "on_train_end": lambda trainer: callback.on_train_end(trainer=trainer)
    }


class ProgressCallback:
    """Simple progress callback for tracking training."""
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.history = []
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Record epoch metrics."""
        self.current_epoch = epoch
        self.history.append({
            "epoch": epoch,
            **metrics
        })
    
    def get_best_epoch(self, metric: str = "val_loss", mode: str = "min"):
        """Get best epoch for given metric."""
        if not self.history:
            return None
        
        if mode == "min":
            best = min(self.history, key=lambda x: x.get(metric, float('inf')))
        else:
            best = max(self.history, key=lambda x: x.get(metric, float('-inf')))
        
        return best["epoch"]
    
    def get_metric_history(self, metric: str):
        """Get history of specific metric."""
        return [h.get(metric) for h in self.history]
