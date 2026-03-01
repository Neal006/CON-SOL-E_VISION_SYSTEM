"""
XGBoost Patch Classifier for SPGS-Net
======================================
Classical ML for patch-level anomaly scoring.

Section 3 of Architecture: Patch-Level Defect Awareness via Classical ML
- Patch embeddings from DINOv2 reshaped into 2D feature set
- XGBoost classifier trained using normal and defect patches
- ML model outputs anomaly score for each patch
- Scores re-projected to spatial locations forming patch-level anomaly heatmap
- Heatmap normalized to range [0, 1]

Rationale: Classical ML models provide strong out-of-distribution sensitivity
and superior generalization to unseen defect types, critical in industrial environments.
"""

import numpy as np
import torch
import xgboost as xgb
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MLConfig, DINOv2Config, PathConfig


class PatchClassifier:
    """
    XGBoost classifier for patch-level defect classification.
    
    Section 3: Patch-level defect awareness via classical ML.
    Predicts anomaly scores for each patch extracted by DINOv2.
    """
    
    def __init__(
        self,
        num_classes: int = None,
        model_path: Optional[Path] = None
    ):
        """
        Initialize patch classifier.
        
        Section 3: Classical ML model for anomaly scoring.
        
        Args:
            num_classes: Number of classes (default from config)
            model_path: Path to load existing model (optional)
        """
        self.num_classes = num_classes or MLConfig.NUM_CLASSES
        self.model_path = model_path or MLConfig.MODEL_PATH
        self.model = None
        
        # Section 3: XGBoost hyperparameters
        self.xgb_params = {
            'n_estimators': MLConfig.N_ESTIMATORS,
            'max_depth': MLConfig.MAX_DEPTH,
            'learning_rate': MLConfig.LEARNING_RATE,
            'min_child_weight': MLConfig.MIN_CHILD_WEIGHT,
            'subsample': MLConfig.SUBSAMPLE,
            'colsample_bytree': MLConfig.COLSAMPLE_BYTREE,
            'objective': MLConfig.OBJECTIVE,
            'num_class': self.num_classes,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1,
        }
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def _prepare_patch_data(
        self,
        patch_features: np.ndarray,
        patch_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flatten patch features for XGBoost training.
        
        Section 3: Patch embeddings reshaped into 2D feature set.
        
        Args:
            patch_features: (N, num_patches, feature_dim) or (N, feature_dim, h, w)
            patch_labels: (N, num_patches) or (N, h, w) patch-level labels
            
        Returns:
            X: (total_patches, feature_dim) flattened features
            y: (total_patches,) class labels
        """
        # Reshape features to 2D
        if len(patch_features.shape) == 4:
            # (N, feature_dim, h, w) -> (N, h*w, feature_dim)
            N, F, H, W = patch_features.shape
            patch_features = patch_features.transpose(0, 2, 3, 1).reshape(-1, F)
            patch_labels = patch_labels.reshape(-1)
        elif len(patch_features.shape) == 3:
            # (N, num_patches, feature_dim) -> (N*num_patches, feature_dim)
            patch_features = patch_features.reshape(-1, patch_features.shape[-1])
            patch_labels = patch_labels.reshape(-1)
        
        return patch_features, patch_labels
    
    def train(
        self,
        patch_features: np.ndarray,
        patch_labels: np.ndarray,
        validation_split: float = 0.2,
        early_stopping_rounds: int = 20,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train XGBoost classifier on patch features.
        
        Section 3: Train ML model using normal and defect patches.
        
        Args:
            patch_features: Patch embeddings from DINOv2
            patch_labels: Patch-level ground truth labels
            validation_split: Fraction for validation
            early_stopping_rounds: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training metrics
        """
        # Section 3: Prepare patch data for training
        X, y = self._prepare_patch_data(patch_features, patch_labels)
        
        if verbose:
            print(f"[Section 3] Training XGBoost patch classifier")
            print(f"  Total patches: {len(X)}")
            print(f"  Feature dimension: {X.shape[1]}")
            print(f"  Class distribution: {np.bincount(y)}")
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        xgb_params_with_early_stop = {
            **self.xgb_params,
            'early_stopping_rounds': early_stopping_rounds
        }
        self.model = xgb.XGBClassifier(**xgb_params_with_early_stop)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose
        )
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        
        if verbose:
            print("\n[Section 3] Validation Results:")
            print(classification_report(y_val, y_pred, 
                                       target_names=['Background', 'Dust', 'RunDown', 'Scratch']))
        
        metrics = {
            'accuracy': float(np.mean(y_pred == y_val)),
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else 0,
        }
        
        return metrics
    
    def predict_proba(
        self,
        patch_features: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Predict class probabilities for patches.
        
        Section 3: ML model outputs anomaly score for each patch.
        
        Args:
            patch_features: (num_patches, feature_dim) or batched
            
        Returns:
            probabilities: (num_patches, num_classes) class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if isinstance(patch_features, torch.Tensor):
            patch_features = patch_features.cpu().numpy()
        
        # Flatten if needed
        original_shape = patch_features.shape
        if len(original_shape) > 2:
            patch_features = patch_features.reshape(-1, original_shape[-1])
        
        probabilities = self.model.predict_proba(patch_features)
        
        return probabilities
    
    def predict_anomaly_scores(
        self,
        patch_features: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Compute anomaly scores (probability of any defect class).
        
        Section 3: Anomaly score = 1 - P(background).
        Scores range [0, 1].
        
        Args:
            patch_features: Patch embeddings
            
        Returns:
            anomaly_scores: (num_patches,) scores in [0, 1]
        """
        proba = self.predict_proba(patch_features)
        
        # Section 3: Anomaly score = 1 - background probability
        # Background is class 0
        anomaly_scores = 1.0 - proba[:, 0]
        
        # Section 3: Normalize to [0, 1] range
        anomaly_scores = np.clip(anomaly_scores, MLConfig.SCORE_MIN, MLConfig.SCORE_MAX)
        
        return anomaly_scores
    
    def save(self, path: Optional[Path] = None):
        """Save model to file."""
        path = path or self.model_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        print(f"[Section 3] Model saved to {path}")
    
    def load(self, path: Optional[Path] = None):
        """Load model from file."""
        path = path or self.model_path
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        print(f"[Section 3] Model loaded from {path}")


def create_anomaly_heatmap(
    patch_features: Union[np.ndarray, torch.Tensor],
    patch_grid: Tuple[int, int],
    classifier: PatchClassifier
) -> np.ndarray:
    """
    Create patch-level anomaly heatmap from features.
    
    Section 3: Scores re-projected to spatial locations
    forming patch-level anomaly heatmap.
    
    Args:
        patch_features: (B, num_patches, feature_dim) or (B, F, H, W)
        patch_grid: (patch_h, patch_w) grid dimensions
        classifier: Trained PatchClassifier
        
    Returns:
        heatmap: (B, patch_h, patch_w) anomaly heatmap in [0, 1]
    """
    if isinstance(patch_features, torch.Tensor):
        patch_features = patch_features.cpu().numpy()
    
    patch_h, patch_w = patch_grid
    
    # Handle different input shapes
    if len(patch_features.shape) == 4:
        # (B, F, H, W) -> (B, H*W, F)
        B, F, H, W = patch_features.shape
        features_flat = patch_features.transpose(0, 2, 3, 1).reshape(B, -1, F)
    elif len(patch_features.shape) == 3:
        # (B, num_patches, F)
        B = patch_features.shape[0]
        features_flat = patch_features
    else:
        # (num_patches, F) - single image
        B = 1
        features_flat = patch_features[np.newaxis, ...]
    
    # Section 3: Get anomaly scores for all patches
    heatmaps = []
    for b in range(B):
        scores = classifier.predict_anomaly_scores(features_flat[b])
        # Reshape to spatial grid
        heatmap = scores.reshape(patch_h, patch_w)
        heatmaps.append(heatmap)
    
    heatmaps = np.stack(heatmaps, axis=0)
    
    # Section 3: Normalize heatmap to [0, 1]
    heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min() + 1e-8)
    
    return heatmaps


def get_patch_labels_from_mask(
    segmentation_mask: np.ndarray,
    patch_size: int = DINOv2Config.PATCH_SIZE
) -> np.ndarray:
    """
    Extract patch-level labels from dense segmentation mask.
    Uses per-class thresholds for better recall on thin defects.
    
    Args:
        segmentation_mask: (H, W) dense mask with class indices
        patch_size: Size of each patch (14 for DINOv2)
        
    Returns:
        patch_labels: (patch_h, patch_w) patch-level labels
    """
    H, W = segmentation_mask.shape
    patch_h = H // patch_size
    patch_w = W // patch_size
    patch_area = patch_size * patch_size
    
    # Per-class coverage thresholds (fraction of patch that must contain class)
    # Lower threshold = more sensitive detection = higher recall
    CLASS_THRESHOLDS = {
        1: 0.10,   # Dust: 10% coverage (blob-like, easier to detect)
        2: 0.10,   # RunDown: 10% coverage
        3: 0.01,   # Scratch: 1% coverage (thin linear defects need low threshold)
    }
    
    patch_labels = np.zeros((patch_h, patch_w), dtype=np.int64)
    
    for i in range(patch_h):
        for j in range(patch_w):
            # Extract patch region
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            patch = segmentation_mask[y1:y2, x1:x2]
            
            unique, counts = np.unique(patch, return_counts=True)
            class_counts = dict(zip(unique, counts))
            
            # Check each defect class with its specific threshold
            # Priority: Scratch > RunDown > Dust (thin defects matter more)
            for class_id in [3, 2, 1]:  # Check Scratch first
                if class_id in class_counts:
                    threshold = CLASS_THRESHOLDS.get(class_id, 0.10)
                    if class_counts[class_id] > patch_area * threshold:
                        patch_labels[i, j] = class_id
                        break
            else:
                # No defect found above threshold, use background
                patch_labels[i, j] = 0
    
    return patch_labels


if __name__ == "__main__":
    # Test XGBoost classifier
    print("Testing XGBoost Patch Classifier...")
    print("=" * 60)
    
    # Create dummy patch features and labels
    num_images = 10
    num_patches = 37 * 37  # For 518x518 image with 14x14 patches
    feature_dim = DINOv2Config.FEATURE_DIM
    
    # Simulate patch features (random)
    patch_features = np.random.randn(num_images, num_patches, feature_dim).astype(np.float32)
    
    # Simulate labels (mostly background, some defects)
    patch_labels = np.zeros((num_images, num_patches), dtype=np.int64)
    # Add some random defects
    for i in range(num_images):
        defect_indices = np.random.choice(num_patches, size=50, replace=False)
        patch_labels[i, defect_indices] = np.random.randint(1, 4, size=50)
    
    print(f"Patch features shape: {patch_features.shape}")
    print(f"Patch labels shape: {patch_labels.shape}")
    
    # Train classifier
    classifier = PatchClassifier()
    metrics = classifier.train(patch_features, patch_labels, verbose=True)
    
    print(f"\n[Section 3] Training metrics: {metrics}")
    
    # Test anomaly heatmap
    test_features = np.random.randn(2, 37, 37, feature_dim).astype(np.float32)
    test_features = test_features.transpose(0, 3, 1, 2)  # (B, F, H, W)
    
    heatmap = create_anomaly_heatmap(test_features, (37, 37), classifier)
    print(f"\n[Section 3] Anomaly heatmap shape: {heatmap.shape}")
    print(f"  Min: {heatmap.min():.4f}, Max: {heatmap.max():.4f}")
