"""Loss functions for segmentation: Dice, Focal, and Combined losses."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice loss for segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1e-6, ignore_index: int = -100) -> None:
        """Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero.
            ignore_index: Class index to ignore in loss computation.
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.
        
        Args:
            pred: Predicted logits of shape (B, C, H, W).
            target: Ground truth of shape (B, H, W) with class indices.
            
        Returns:
            Dice loss scalar.
        """
        num_classes = pred.shape[1]
        
        # Apply softmax to get probabilities
        pred_prob = F.softmax(pred, dim=1)
        
        # Create mask for valid pixels
        valid_mask = (target != self.ignore_index)
        target_masked = target.clone().long()  # Ensure long type for one_hot
        target_masked[~valid_mask] = 0
        
        # One-hot encode target
        target_one_hot = F.one_hot(target_masked, num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).expand_as(pred_prob)
        pred_prob = pred_prob * valid_mask
        target_one_hot = target_one_hot * valid_mask
        
        # Compute Dice per class
        dims = (0, 2, 3)
        intersection = (pred_prob * target_one_hot).sum(dim=dims)
        pred_sum = pred_prob.sum(dim=dims)
        target_sum = target_one_hot.sum(dim=dims)
        
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Mean over classes
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ) -> None:
        """Initialize Focal loss.
        
        Args:
            alpha: Weighting factor for rare classes.
            gamma: Focusing parameter (higher = more focus on hard examples).
            class_weights: Optional per-class weights [bg, dust, rundown, scratch].
            ignore_index: Class index to ignore.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal loss.
        
        Args:
            pred: Predicted logits of shape (B, C, H, W).
            target: Ground truth of shape (B, H, W).
            
        Returns:
            Focal loss scalar.
        """
        # Compute cross-entropy with optional class weights
        weight = self.class_weights.to(pred.device) if self.class_weights is not None else None
        
        ce_loss = F.cross_entropy(
            pred, target,
            weight=weight,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Get probabilities for correct class
        pt = torch.exp(-ce_loss)
        
        # Compute focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Compute focal loss
        focal_loss = focal_weight * ce_loss
        
        # Mean over valid pixels
        valid_mask = (target != self.ignore_index)
        if valid_mask.sum() > 0:
            return focal_loss[valid_mask].mean()
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss: weighted sum of Dice and Focal losses."""
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ) -> None:
        """Initialize combined loss.
        
        Args:
            dice_weight: Weight for Dice loss.
            focal_weight: Weight for Focal loss.
            focal_alpha: Alpha parameter for Focal loss.
            focal_gamma: Gamma parameter for Focal loss.
            class_weights: Per-class weights for handling imbalance.
            ignore_index: Class index to ignore.
        """
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            class_weights=class_weights,
            ignore_index=ignore_index
        )
        
        logger.info(f"Combined loss: dice_w={dice_weight}, focal_w={focal_weight}, "
                   f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                   f"class_weights={class_weights}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            pred: Predicted logits of shape (B, C, H, W).
            target: Ground truth of shape (B, H, W).
            
        Returns:
            Combined loss scalar.
        """
        # Ensure target is long type for both losses
        target = target.long()
        
        loss = 0.0
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(pred, target)
        
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal_loss(pred, target)
        
        return loss


def build_loss(config: dict) -> CombinedLoss:
    """Build loss function from configuration.
    
    Args:
        config: Configuration dictionary with 'loss' section.
        
    Returns:
        CombinedLoss instance.
    """
    loss_cfg = config.get('loss', {})
    
    # Parse class weights if provided
    class_weights = None
    if 'class_weights' in loss_cfg:
        class_weights = torch.tensor(loss_cfg['class_weights'], dtype=torch.float32)
    
    return CombinedLoss(
        dice_weight=loss_cfg.get('dice_weight', 0.5),
        focal_weight=loss_cfg.get('focal_weight', 0.5),
        focal_alpha=loss_cfg.get('focal_alpha', 0.25),
        focal_gamma=loss_cfg.get('focal_gamma', 2.0),
        class_weights=class_weights
    )


if __name__ == "__main__":
    # Test loss functions
    logging.basicConfig(level=logging.INFO)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Create mock predictions and targets
    batch_size, num_classes = 4, 2
    H, W = 518, 518
    
    pred = torch.randn(batch_size, num_classes, H, W).to(device)
    target = torch.randint(0, num_classes, (batch_size, H, W)).to(device)
    
    # Test individual losses
    dice_loss = DiceLoss().to(device)
    focal_loss = FocalLoss().to(device)
    combined_loss = CombinedLoss().to(device)
    
    print(f"\nDice loss:     {dice_loss(pred, target).item():.4f}")
    print(f"Focal loss:    {focal_loss(pred, target).item():.4f}")
    print(f"Combined loss: {combined_loss(pred, target).item():.4f}")
    
    # Test gradient flow
    pred.requires_grad = True
    loss = combined_loss(pred, target)
    loss.backward()
    print(f"\nGradient computed successfully: {pred.grad is not None}")
