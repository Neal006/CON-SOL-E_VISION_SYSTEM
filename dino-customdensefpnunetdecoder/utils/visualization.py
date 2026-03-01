"""Visualization utilities for segmentation results."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Color palette for classes
DEFAULT_COLORS = np.array([
    [0, 0, 0],        # Background - Black
    [255, 0, 0],      # Class 1 - Red
    [0, 255, 0],      # Class 2 - Green
    [0, 0, 255],      # Class 3 - Blue
    [255, 255, 0],    # Class 4 - Yellow
    [255, 0, 255],    # Class 5 - Magenta
    [0, 255, 255],    # Class 6 - Cyan
], dtype=np.uint8)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize_image(image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Denormalize image from ImageNet normalization.
    
    Args:
        image: Normalized image (C, H, W) or (H, W, C).
        
    Returns:
        Denormalized image in [0, 255] as uint8.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if image.shape[0] == 3:  # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


def mask_to_rgb(
    mask: Union[torch.Tensor, np.ndarray],
    colors: Optional[np.ndarray] = None
) -> np.ndarray:
    """Convert segmentation mask to RGB image.
    
    Args:
        mask: Class indices of shape (H, W).
        colors: Color palette array of shape (num_classes, 3).
        
    Returns:
        RGB image of shape (H, W, 3).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if colors is None:
        colors = DEFAULT_COLORS
    
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    for cls_idx in range(len(colors)):
        rgb[mask == cls_idx] = colors[cls_idx]
    
    return rgb


def visualize_predictions(
    image: Union[torch.Tensor, np.ndarray],
    true_mask: Union[torch.Tensor, np.ndarray],
    pred_mask: Union[torch.Tensor, np.ndarray],
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    alpha: float = 0.5
) -> Optional[plt.Figure]:
    """Visualize image with ground truth and predicted masks.
    
    Args:
        image: Input image (C, H, W) or (H, W, C).
        true_mask: Ground truth mask (H, W).
        pred_mask: Predicted mask (H, W).
        save_path: Optional path to save figure.
        class_names: Optional list of class names.
        alpha: Overlay transparency.
        
    Returns:
        Figure object if save_path is None.
    """
    # Prepare image
    image = denormalize_image(image)
    
    # Prepare masks
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    
    true_rgb = mask_to_rgb(true_mask)
    pred_rgb = mask_to_rgb(pred_mask)
    
    # Create overlay
    true_overlay = (image * (1 - alpha) + true_rgb * alpha).astype(np.uint8)
    pred_overlay = (image * (1 - alpha) + pred_rgb * alpha).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(true_overlay)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_overlay)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Difference map
    diff = (true_mask != pred_mask).astype(np.uint8) * 255
    axes[3].imshow(diff, cmap='Reds')
    axes[3].set_title('Errors')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved visualization to {save_path}")
        return None
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_ious: List[float],
    val_ious: List[float],
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """Plot training and validation curves.
    
    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        train_ious: Training IoU per epoch.
        val_ious: Validation IoU per epoch.
        save_path: Optional path to save figure.
        
    Returns:
        Figure object if save_path is None.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU curves
    axes[1].plot(epochs, train_ious, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_ious, 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean IoU')
    axes[1].set_title('Training and Validation IoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training curves to {save_path}")
        return None
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True
) -> Optional[plt.Figure]:
    """Plot confusion matrix.
    
    Args:
        cm: Confusion matrix of shape (num_classes, num_classes).
        class_names: List of class names.
        save_path: Optional path to save figure.
        normalize: Whether to normalize the matrix.
        
    Returns:
        Figure object if save_path is None.
    """
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, format(cm[i, j], fmt),
                   ha='center', va='center', color=color)
    
    plt.colorbar(im)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix to {save_path}")
        return None
    
    return fig


def visualize_feature_maps(
    features: torch.Tensor,
    save_path: Optional[str] = None,
    num_features: int = 16
) -> Optional[plt.Figure]:
    """Visualize feature maps from encoder.
    
    Args:
        features: Feature tensor of shape (C, H, W) or (B, C, H, W).
        save_path: Optional path to save figure.
        num_features: Number of feature channels to visualize.
        
    Returns:
        Figure object if save_path is None.
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    
    if features.ndim == 4:
        features = features[0]  # Take first batch
    
    C, H, W = features.shape
    num_features = min(num_features, C)
    
    # Calculate grid size
    cols = int(np.ceil(np.sqrt(num_features)))
    rows = int(np.ceil(num_features / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if num_features > 1 else [axes]
    
    for i in range(num_features):
        feat = features[i]
        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)
        axes[i].imshow(feat, cmap='viridis')
        axes[i].set_title(f'Ch {i}')
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(num_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved feature maps to {save_path}")
        return None
    
    return fig


if __name__ == "__main__":
    # Test visualization functions
    logging.basicConfig(level=logging.INFO)
    
    # Create mock data
    H, W = 256, 256
    image = torch.randn(3, H, W)
    true_mask = torch.randint(0, 2, (H, W))
    pred_mask = torch.randint(0, 2, (H, W))
    
    print("Testing visualization functions...")
    
    # Test visualize_predictions
    fig = visualize_predictions(image, true_mask, pred_mask)
    plt.close(fig)
    print("  visualize_predictions: OK")
    
    # Test plot_training_curves
    train_losses = [1.0, 0.8, 0.6, 0.4, 0.3]
    val_losses = [1.1, 0.9, 0.7, 0.5, 0.4]
    train_ious = [0.3, 0.5, 0.6, 0.7, 0.75]
    val_ious = [0.25, 0.45, 0.55, 0.65, 0.7]
    fig = plot_training_curves(train_losses, val_losses, train_ious, val_ious)
    plt.close(fig)
    print("  plot_training_curves: OK")
    
    # Test plot_confusion_matrix
    cm = np.array([[90, 10], [20, 80]])
    fig = plot_confusion_matrix(cm, ['background', 'defect'])
    plt.close(fig)
    print("  plot_confusion_matrix: OK")
    
    print("\nAll visualization tests passed!")
