"""
MLflow Model Registry.

Handles:
- Model versioning
- Stage transitions (Staging -> Production)
- Model loading from registry
- Model comparison
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class ModelRegistry:
    """
    MLflow Model Registry wrapper.
    
    Model Lifecycle:
        None → Staging → Production → Archived
    """
    
    def __init__(
        self,
        registry_uri: str = "mlruns",
        tracking_uri: str = None
    ):
        """
        Initialize model registry.
        
        Args:
            registry_uri: URI for model registry
            tracking_uri: URI for MLflow tracking (defaults to registry_uri)
        """
        self.registry_uri = registry_uri
        self.tracking_uri = tracking_uri or registry_uri
        
        self._mlflow = None
        self._client = None
        
        self._setup()
    
    def _setup(self):
        """Setup MLflow client."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            self._mlflow = mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_registry_uri(self.registry_uri)
            
            self._client = MlflowClient()
            
        except ImportError:
            print("Warning: MLflow not installed")
            self._mlflow = None
            self._client = None
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        stage: str = None,
        description: str = None,
        tags: Dict[str, str] = None
    ) -> str:
        """
        Register a model from a run.
        
        Args:
            run_id: MLflow run ID containing the model
            model_name: Name for the registered model
            artifact_path: Path to model artifact within run
            stage: Optional initial stage (Staging, Production)
            description: Optional model description
            tags: Optional tags
        
        Returns:
            Model version string
        """
        if self._mlflow is None:
            return None
        
        # Create model URI
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        # Register
        result = self._mlflow.register_model(model_uri, model_name)
        version = result.version
        
        # Set stage if provided
        if stage:
            self.transition_stage(model_name, version, stage)
        
        # Set description if provided
        if description:
            self._client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
        
        # Set tags if provided
        if tags:
            for key, value in tags.items():
                self._client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key=key,
                    value=value
                )
        
        return str(version)
    
    def transition_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
        archive_existing: bool = True
    ):
        """
        Transition model version to new stage.
        
        Args:
            model_name: Name of registered model
            version: Model version number
            stage: Target stage (Staging, Production, Archived, None)
            archive_existing: Whether to archive existing models in target stage
        """
        if self._client is None:
            return
        
        self._client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
    
    def load_model(
        self,
        model_name: str,
        stage: str = "Production",
        version: int = None
    ) -> Any:
        """
        Load model from registry.
        
        Args:
            model_name: Name of registered model
            stage: Stage to load from (if version not specified)
            version: Specific version to load
        
        Returns:
            Loaded model object
        """
        if self._mlflow is None:
            return None
        
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage}"
        
        try:
            model = self._mlflow.pyfunc.load_model(model_uri)
            return model
        except Exception:
            # Try loading as PyTorch model
            try:
                model = self._mlflow.pytorch.load_model(model_uri)
                return model
            except Exception:
                return None
    
    def get_latest_version(
        self,
        model_name: str,
        stage: str = None
    ) -> Optional[int]:
        """
        Get latest version of a model.
        
        Args:
            model_name: Name of registered model
            stage: Optional stage filter
        
        Returns:
            Latest version number
        """
        if self._client is None:
            return None
        
        stages = [stage] if stage else None
        versions = self._client.get_latest_versions(model_name, stages=stages)
        
        if not versions:
            return None
        
        return max(int(v.version) for v in versions)
    
    def get_model_versions(self, model_name: str) -> List[Dict]:
        """
        Get all versions of a model.
        
        Args:
            model_name: Name of registered model
        
        Returns:
            List of version dictionaries
        """
        if self._client is None:
            return []
        
        versions = self._client.search_model_versions(f"name='{model_name}'")
        
        return [
            {
                "version": v.version,
                "stage": v.current_stage,
                "status": v.status,
                "creation_time": v.creation_timestamp,
                "description": v.description,
                "run_id": v.run_id
            }
            for v in versions
        ]
    
    def compare_models(
        self,
        model_names: List[str],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple registered models.
        
        Args:
            model_names: List of model names to compare
            metrics: List of metrics to compare
        
        Returns:
            DataFrame with comparison
        """
        if self._client is None:
            return pd.DataFrame()
        
        metrics = metrics or ["recall", "precision", "mAP", "dice", "area_error"]
        
        rows = []
        
        for model_name in model_names:
            versions = self._client.search_model_versions(f"name='{model_name}'")
            
            for v in versions:
                if v.current_stage not in ["Production", "Staging"]:
                    continue
                
                # Get run metrics
                run = self._client.get_run(v.run_id)
                run_metrics = run.data.metrics
                
                row = {
                    "model_name": model_name,
                    "version": v.version,
                    "stage": v.current_stage
                }
                
                for metric in metrics:
                    row[metric] = run_metrics.get(metric, None)
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def delete_version(self, model_name: str, version: int):
        """
        Delete a model version.
        
        Args:
            model_name: Name of registered model
            version: Version to delete
        """
        if self._client is None:
            return
        
        self._client.delete_model_version(
            name=model_name,
            version=str(version)
        )
    
    def delete_model(self, model_name: str):
        """
        Delete registered model and all versions.
        
        Args:
            model_name: Name of model to delete
        """
        if self._client is None:
            return
        
        self._client.delete_registered_model(name=model_name)
    
    def list_models(self) -> List[Dict]:
        """
        List all registered models.
        
        Returns:
            List of model dictionaries
        """
        if self._client is None:
            return []
        
        models = self._client.search_registered_models()
        
        return [
            {
                "name": m.name,
                "description": m.description,
                "creation_time": m.creation_timestamp,
                "last_updated": m.last_updated_timestamp,
                "latest_versions": [
                    {"version": v.version, "stage": v.current_stage}
                    for v in m.latest_versions
                ]
            }
            for m in models
        ]
    
    def get_production_model_uri(self, model_name: str) -> Optional[str]:
        """
        Get URI for production model.
        
        Args:
            model_name: Name of registered model
        
        Returns:
            Model URI string
        """
        version = self.get_latest_version(model_name, stage="Production")
        
        if version is None:
            return None
        
        return f"models:/{model_name}/Production"
    
    def promote_to_production(
        self,
        model_name: str,
        version: int,
        archive_previous: bool = True
    ):
        """
        Promote model version to production.
        
        Args:
            model_name: Name of registered model
            version: Version to promote
            archive_previous: Whether to archive current production model
        """
        self.transition_stage(
            model_name=model_name,
            version=version,
            stage="Production",
            archive_existing=archive_previous
        )
