"""
MLflow Experiment Tracker.

Handles:
- Experiment creation and management
- Parameter, metric, and artifact logging
- Model logging and versioning
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class ExperimentTracker:
    """
    MLflow experiment tracking wrapper.
    
    Features:
    - Auto-log training parameters
    - Log metrics per epoch
    - Save confusion matrices as artifacts
    - Log sample prediction images
    """
    
    def __init__(
        self,
        experiment_name: str = "yolo_sam_defect_detection",
        tracking_uri: str = "mlruns",
        artifact_location: str = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server
            artifact_location: Optional custom artifact location
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        
        self._mlflow = None
        self._run = None
        self._active_run_id = None
        
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Initialize MLflow."""
        try:
            import mlflow
            self._mlflow = mlflow
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location
                )
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.experiment_name)
            
        except ImportError:
            print("Warning: MLflow not installed. Tracking disabled.")
            print("Install with: pip install mlflow")
            self._mlflow = None
    
    def start_run(
        self,
        run_name: str = None,
        tags: Dict[str, str] = None,
        nested: bool = False
    ) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags dictionary
            nested: Whether this is a nested run
        
        Returns:
            Run ID
        """
        if self._mlflow is None:
            return None
        
        self._run = self._mlflow.start_run(run_name=run_name, nested=nested)
        self._active_run_id = self._run.info.run_id
        
        if tags:
            self._mlflow.set_tags(tags)
        
        return self._active_run_id
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if self._mlflow is None:
            return
        
        self._mlflow.end_run(status=status)
        self._run = None
        self._active_run_id = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.
        
        Args:
            params: Dictionary of parameters
        """
        if self._mlflow is None:
            return
        
        # Handle nested dicts by flattening
        flat_params = self._flatten_dict(params)
        
        # MLflow has limits on param value length
        for key, value in flat_params.items():
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:497] + "..."
            self._mlflow.log_param(key, str_value)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int = None
    ):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Optional step number
        """
        if self._mlflow is None:
            return
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self._mlflow.log_metric(key, value, step=step)
    
    def log_artifact(
        self,
        local_path: str,
        artifact_path: str = None
    ):
        """
        Log a local file as an artifact.
        
        Args:
            local_path: Path to local file
            artifact_path: Optional artifact subdirectory
        """
        if self._mlflow is None:
            return
        
        self._mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: str = None
    ):
        """
        Log all files in a directory as artifacts.
        
        Args:
            local_dir: Path to local directory
            artifact_path: Optional artifact subdirectory
        """
        if self._mlflow is None:
            return
        
        self._mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: str = None
    ):
        """
        Log a model.
        
        Args:
            model: Model object
            artifact_path: Artifact path for the model
            registered_model_name: Optional name to register in model registry
        """
        if self._mlflow is None:
            return
        
        # Try different model flavors
        try:
            self._mlflow.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name
            )
        except Exception:
            # Fallback to generic artifact logging
            import tempfile
            import torch
            
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                torch.save(model, f.name)
                self._mlflow.log_artifact(f.name, artifact_path)
    
    def log_figure(
        self,
        figure,
        artifact_file: str
    ):
        """
        Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure
            artifact_file: Filename for the artifact
        """
        if self._mlflow is None:
            return
        
        self._mlflow.log_figure(figure, artifact_file)
    
    def log_image(
        self,
        image: np.ndarray,
        artifact_file: str
    ):
        """
        Log an image array.
        
        Args:
            image: Numpy array image
            artifact_file: Filename for the artifact
        """
        if self._mlflow is None:
            return
        
        self._mlflow.log_image(image, artifact_file)
    
    def log_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        artifact_name: str = "confusion_matrix.png"
    ):
        """
        Log confusion matrix as image artifact.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            artifact_name: Name for the artifact file
        """
        if self._mlflow is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            
            self.log_figure(fig, artifact_name)
            plt.close(fig)
            
        except ImportError:
            print("Warning: matplotlib/seaborn not available for confusion matrix plotting")
    
    def log_sample_predictions(
        self,
        images: List[np.ndarray],
        predictions: List[Dict],
        ground_truths: List[Dict] = None,
        artifact_dir: str = "sample_predictions"
    ):
        """
        Log sample prediction images.
        
        Args:
            images: List of images
            predictions: List of prediction dictionaries
            ground_truths: Optional list of ground truth dictionaries
            artifact_dir: Artifact subdirectory
        """
        if self._mlflow is None:
            return
        
        import tempfile
        import cv2
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (image, preds) in enumerate(zip(images, predictions)):
                # Create visualization
                vis_image = image.copy()
                
                # Draw predictions
                for pred in preds if isinstance(preds, list) else [preds]:
                    if "bbox" in pred:
                        x1, y1, x2, y2 = [int(x) for x in pred["bbox"]]
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = pred.get("class_name", "")
                        cv2.putText(
                            vis_image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                        )
                
                # Save
                save_path = Path(tmpdir) / f"sample_{i}.png"
                cv2.imwrite(str(save_path), vis_image)
            
            self.log_artifacts(tmpdir, artifact_dir)
    
    def set_tag(self, key: str, value: str):
        """Set a single tag."""
        if self._mlflow is None:
            return
        self._mlflow.set_tag(key, value)
    
    def get_run_id(self) -> Optional[str]:
        """Get current run ID."""
        return self._active_run_id
    
    def get_experiment_id(self) -> Optional[str]:
        """Get current experiment ID."""
        if self._mlflow is None:
            return None
        
        experiment = self._mlflow.get_experiment_by_name(self.experiment_name)
        return experiment.experiment_id if experiment else None
    
    def _flatten_dict(
        self,
        d: Dict,
        parent_key: str = "",
        sep: str = "."
    ) -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    @property
    def is_active(self) -> bool:
        """Check if a run is active."""
        return self._run is not None


def create_tracker(
    experiment_name: str = "yolo_sam_defect_detection",
    tracking_uri: str = "mlruns"
) -> ExperimentTracker:
    """
    Convenience function to create tracker.
    
    Args:
        experiment_name: Experiment name
        tracking_uri: Tracking URI
    
    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri
    )
