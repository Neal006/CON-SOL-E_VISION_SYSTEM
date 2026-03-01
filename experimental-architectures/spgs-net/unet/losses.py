"""
Loss Functions for SPGS-Net
=============================
Dice Loss, Focal Loss, and Prior-Weighted combinations.

Section 9 of Architecture: Training Strategy
- Dice Loss for segmentation overlap
- Binary Cross-Entropy or Focal Loss for class imbalance
- Anomaly prior treated as fixed guidance for loss reweighting
- Defect-likely pixels receive higher loss contribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import TrainingConfig


# =============================================================================
# Section 9: Dice Loss
# Measures overlap between prediction and ground truth
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Section 9: Primary loss for spatial overlap optimization.
    Works well for imbalanced segmentation problems.
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to prevent division by zero
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Class index to ignore
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Section 9: Per-class Dice then averaged.
        
        Args:
            logits: (B, C, H, W) predicted logits
            targets: (B, H, W) ground truth class indices
            
        Returns:
            Dice loss value
        """
        num_classes = logits.shape[1]
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets.long(), num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Compute Dice per class
        dice_losses = []
        
        for c in range(num_classes):
            if c == self.ignore_index:
                continue
            
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]
            
            # Section 9: Dice formula
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_losses.append(1.0 - dice)
        
        # Average across classes
        if len(dice_losses) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        loss = torch.stack(dice_losses)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# =============================================================================
# Section 9: Focal Loss
# Handles class imbalance by down-weighting easy examples
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Section 9: Alternative to BCE for imbalanced defect detection.
    Down-weights well-classified (easy) examples.
    """
    
    def __init__(
        self,
        gamma: float = None,
        alpha: Optional[float] = None,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """
        Initialize Focal Loss.
        
        Section 9: Focal Loss for imbalance handling.
        
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Balance parameter for positive class
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Class index to ignore
        """
        super().__init__()
        self.gamma = gamma if gamma is not None else TrainingConfig.FOCAL_GAMMA
        self.alpha = alpha if alpha is not None else TrainingConfig.FOCAL_ALPHA
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Section 9: FL = -alpha * (1-p)^gamma * log(p)
        
        Args:
            logits: (B, C, H, W) predicted logits
            targets: (B, H, W) ground truth class indices
            
        Returns:
            Focal loss value
        """
        # Convert to cross entropy format
        ce_loss = F.cross_entropy(
            logits, targets.long(),
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get probability of correct class
        targets_flat = targets.view(-1).long()
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])
        
        # Gather correct class probabilities
        pt = probs_flat.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt = pt.view_as(targets)
        
        # Section 9: Apply focal modulation
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_weight = torch.where(
                targets > 0,  # Defect classes (1, 2, 3)
                torch.tensor(self.alpha, device=logits.device),
                torch.tensor(1 - self.alpha, device=logits.device)
            )
            focal_weight = focal_weight * alpha_weight
        
        # Section 9: Focal loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# Section 9: Combined Loss
# Dice + Focal/BCE weighted combination
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Combined Dice + Focal Loss.
    
    Section 9: Training strategy uses both Dice Loss and Focal Loss.
    """
    
    def __init__(
        self,
        dice_weight: float = None,
        focal_weight: float = None
    ):
        """
        Initialize combined loss.
        
        Section 9: Loss combination as per architecture.
        
        Args:
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
        """
        super().__init__()
        self.dice_weight = dice_weight if dice_weight is not None else TrainingConfig.DICE_WEIGHT
        self.focal_weight = focal_weight if focal_weight is not None else TrainingConfig.FOCAL_WEIGHT
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Section 9: Weighted sum of Dice and Focal losses.
        
        Args:
            logits: (B, C, H, W) predicted logits
            targets: (B, H, W) ground truth
            
        Returns:
            Dictionary with 'loss', 'dice_loss', 'focal_loss'
        """
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        
        # Section 9: Combined loss
        total = self.dice_weight * dice + self.focal_weight * focal
        
        return {
            'loss': total,
            'dice_loss': dice,
            'focal_loss': focal
        }


# =============================================================================
# Section 9: Prior-Weighted Loss
# Defect-likely pixels receive higher loss contribution
# =============================================================================

class PriorWeightedLoss(nn.Module):
    """
    Prior-weighted loss for guided training.
    
    Section 9: Anomaly prior treated as fixed guidance.
    Defect-likely pixels receive higher loss contribution.
    """
    
    def __init__(
        self,
        base_loss: Optional[nn.Module] = None,
        prior_weight_factor: float = None,
        use_prior_reweight: bool = None
    ):
        """
        Initialize prior-weighted loss.
        
        Section 9: Loss reweighting based on anomaly prior.
        
        Args:
            base_loss: Base loss function (default CombinedLoss)
            prior_weight_factor: Factor to increase loss for defect regions
            use_prior_reweight: Whether to apply reweighting
        """
        super().__init__()
        
        self.base_loss = base_loss or CombinedLoss()
        self.prior_weight_factor = prior_weight_factor if prior_weight_factor is not None else TrainingConfig.PRIOR_REWEIGHT_FACTOR
        self.use_prior_reweight = use_prior_reweight if use_prior_reweight is not None else TrainingConfig.USE_PRIOR_REWEIGHT
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        prior: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute prior-weighted loss.
        
        Section 9: Higher loss for defect-likely regions.
        
        Args:
            logits: (B, C, H, W) predicted logits
            targets: (B, H, W) ground truth
            prior: (B, 1, H, W) anomaly prior map
            
        Returns:
            Dictionary with 'loss' and component losses
        """
        if not self.use_prior_reweight or prior is None:
            # Fall back to base loss
            return self.base_loss(logits, targets)
        
        # Section 9: Compute pixel-wise weights from prior
        # Higher prior = higher weight (more likely defect)
        prior_squeeze = prior.squeeze(1)  # (B, H, W)
        
        # Weight = 1 + factor * prior (prior in [0, 1])
        weights = 1.0 + self.prior_weight_factor * prior_squeeze
        
        # Compute base losses (per-pixel for weighting)
        dice = self.base_loss.dice_loss(logits, targets)
        focal = self.base_loss.focal_loss(logits, targets)
        
        # Note: Dice loss is already averaged, so we can't easily weight it per-pixel
        # Instead, we apply weighting to focal loss and use weighted average
        
        # For focal loss, recompute with weighting
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1).long()).squeeze(1)
        focal_weight = (1 - pt) ** self.base_loss.focal_loss.gamma
        
        # Section 9: Apply prior-based reweighting
        weighted_focal = ce_loss * focal_weight * weights
        weighted_focal = weighted_focal.mean()
        
        # Combine losses
        total = self.base_loss.dice_weight * dice + self.base_loss.focal_weight * weighted_focal
        
        return {
            'loss': total,
            'dice_loss': dice,
            'focal_loss': weighted_focal,
            'prior_applied': torch.tensor(True)
        }


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions...")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy data
    batch_size = 2
    num_classes = 4
    height, width = 128, 128
    
    logits = torch.randn(batch_size, num_classes, height, width).to(device)
    targets = torch.randint(0, num_classes, (batch_size, height, width)).to(device)
    prior = torch.rand(batch_size, 1, height, width).to(device)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Prior shape: {prior.shape}")
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    dice_val = dice_loss(logits, targets)
    print(f"\n[Section 9] Dice Loss: {dice_val.item():.4f}")
    
    # Test Focal Loss
    focal_loss = FocalLoss()
    focal_val = focal_loss(logits, targets)
    print(f"[Section 9] Focal Loss: {focal_val.item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss()
    combined_result = combined_loss(logits, targets)
    print(f"[Section 9] Combined Loss: {combined_result['loss'].item():.4f}")
    
    # Test Prior-Weighted Loss
    prior_weighted_loss = PriorWeightedLoss()
    prior_result = prior_weighted_loss(logits, targets, prior)
    print(f"[Section 9] Prior-Weighted Loss: {prior_result['loss'].item():.4f}")
