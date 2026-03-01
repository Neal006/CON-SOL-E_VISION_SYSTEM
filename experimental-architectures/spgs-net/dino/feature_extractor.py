"""
DINOv2 Feature Extractor for SPGS-Net
======================================
Frozen self-supervised feature extraction using Vision Transformer.

Section 2 of Architecture: Frozen Self-Supervised Feature Extraction (DINOv2)
- Frozen DINOv2 Vision Transformer backbone as feature extractor
- Pretrained in self-supervised manner on large-scale image corpora
- Input image divided into non-overlapping patches (14×14 pixels)
- Patch embeddings extracted from multiple transformer layers:
  * Low-level texture variations
  * Mid-level structural patterns
  * High-level semantic context
- Output: dense patch-wise feature tensor

Rationale: DINOv2 embeddings exhibit strong sensitivity to surface texture
irregularities, making them highly suitable for defect representation
under small-data constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DINOv2Config, TrainingConfig


class DINOv2Extractor(nn.Module):
    """
    DINOv2 Feature Extractor with multi-layer patch embedding extraction.
    
    Section 2: Frozen self-supervised feature extraction
    Extracts patch-wise features from multiple transformer layers
    for low, mid, and high-level representations.
    """
    
    def __init__(
        self,
        model_name: str = None,
        feature_layers: List[int] = None,
        freeze: bool = None
    ):
        """
        Initialize DINOv2 feature extractor.
        
        Section 2: Frozen backbone - weights not updated during training.
        
        Args:
            model_name: DINOv2 model name (default from config)
            feature_layers: Layers to extract features from (default from config)
            freeze: Whether to freeze backbone (default from config)
        """
        super().__init__()
        
        self.model_name = model_name or DINOv2Config.MODEL_NAME
        self.feature_layers = feature_layers or DINOv2Config.FEATURE_LAYERS
        self.freeze = freeze if freeze is not None else DINOv2Config.FREEZE_BACKBONE
        self.patch_size = DINOv2Config.PATCH_SIZE
        self.feature_dim = DINOv2Config.FEATURE_DIM
        
        # Section 2: Load pretrained DINOv2 backbone
        self.backbone = self._load_backbone()
        
        # Section 2: Keep backbone frozen during training
        if self.freeze:
            self._freeze_backbone()
        
        # Intermediate feature hooks
        self.intermediate_features = {}
        self._register_hooks()
        
        # Feature projection (combine multi-layer features)
        # Combined dim = feature_dim * num_layers
        combined_dim = self.feature_dim * len(self.feature_layers)
        self.feature_proj = nn.Sequential(
            nn.Linear(combined_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU()
        )
    
    def _load_backbone(self) -> nn.Module:
        """
        Load pretrained DINOv2 model from torch hub.
        
        Section 2: Backbone pretrained in self-supervised manner
        on large-scale image corpora.
        """
        try:
            backbone = torch.hub.load(
                'facebookresearch/dinov2',
                self.model_name,
                pretrained=True
            )
            print(f"[Section 2] Loaded DINOv2 backbone: {self.model_name}")
            return backbone
        except Exception as e:
            print(f"Error loading DINOv2: {e}")
            print("Attempting to load from local cache...")
            backbone = torch.hub.load(
                'facebookresearch/dinov2',
                self.model_name,
                pretrained=True,
                force_reload=False
            )
            return backbone
    
    def _freeze_backbone(self):
        """
        Freeze all backbone parameters.
        
        Section 2 & 9: DINOv2 backbone remains frozen throughout training.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Section 2 & 9] DINOv2 backbone frozen - weights will not be updated")
    
    def _register_hooks(self):
        """
        Register forward hooks to capture intermediate layer outputs.
        
        Section 2: Extract features from multiple transformer layers
        for low/mid/high level representation.
        """
        def get_hook(layer_idx):
            def hook(module, input, output):
                self.intermediate_features[layer_idx] = output
            return hook
        
        # Register hooks on specified layers
        for layer_idx in self.feature_layers:
            layer = self.backbone.blocks[layer_idx]
            layer.register_forward_hook(get_hook(layer_idx))
        
        print(f"[Section 2] Registered hooks on layers: {self.feature_layers}")
        print(f"  - Layer {self.feature_layers[0]}: Low-level texture variations")
        print(f"  - Layer {self.feature_layers[1]}: Mid-level structural patterns")
        print(f"  - Layer {self.feature_layers[2]}: High-level semantic context")
    
    def forward(
        self, 
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract patch-wise features from input image.
        
        Section 2: Input divided into non-overlapping 14×14 patches.
        Output is dense patch-wise feature tensor.
        
        Args:
            x: Input tensor (B, 3, H, W) - normalized RGB image
            return_intermediate: Whether to return per-layer features
            
        Returns:
            Dictionary with:
            - 'features': Combined patch features (B, num_patches, feature_dim)
            - 'patch_grid': (patch_h, patch_w) tuple
            - 'intermediate' (optional): Per-layer features
        """
        B, C, H, W = x.shape
        
        # Section 2: Calculate patch grid dimensions
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        # Clear intermediate features
        self.intermediate_features = {}
        
        # Forward pass through backbone
        with torch.set_grad_enabled(not self.freeze):
            # Get final output (includes CLS token)
            output = self.backbone.forward_features(x)
        
        # Section 2: Extract patch tokens (exclude CLS token)
        # output shape: (B, 1 + num_patches, feature_dim)
        # We want just the patch tokens, not CLS
        
        # Collect multi-layer features
        multi_layer_features = []
        for layer_idx in self.feature_layers:
            if layer_idx in self.intermediate_features:
                # Shape: (B, 1 + num_patches, feature_dim)
                layer_feat = self.intermediate_features[layer_idx]
                # Remove CLS token
                patch_feat = layer_feat[:, 1:, :]
                multi_layer_features.append(patch_feat)
        
        # Section 2: Concatenate multi-layer features
        # Each captures different level of abstraction
        if multi_layer_features:
            combined = torch.cat(multi_layer_features, dim=-1)
            # Project to unified feature dimension
            features = self.feature_proj(combined)
        else:
            # Fallback: use final layer features only
            features = output[:, 1:, :]  # Exclude CLS token
        
        result = {
            'features': features,  # (B, patch_h * patch_w, feature_dim)
            'patch_grid': (patch_h, patch_w),
        }
        
        if return_intermediate:
            result['intermediate'] = {
                f'layer_{idx}': feat[:, 1:, :]  # Exclude CLS
                for idx, feat in self.intermediate_features.items()
            }
        
        return result
    
    def get_spatial_features(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract features reshaped to spatial grid.
        
        Section 2: Output dense patch-wise feature tensor
        with spatial correspondence preserved.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            features: (B, feature_dim, patch_h, patch_w)
            patch_grid: (patch_h, patch_w)
        """
        result = self.forward(x)
        features = result['features']
        patch_h, patch_w = result['patch_grid']
        
        B = features.shape[0]
        
        # Reshape to spatial grid
        # (B, patch_h * patch_w, feature_dim) -> (B, feature_dim, patch_h, patch_w)
        spatial_features = features.transpose(1, 2).view(B, -1, patch_h, patch_w)
        
        return spatial_features, (patch_h, patch_w)


def extract_patch_features(
    images: torch.Tensor,
    extractor: DINOv2Extractor = None,
    device: str = None
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Convenience function to extract patch features from images.
    
    Section 2: Frozen self-supervised feature extraction.
    
    Args:
        images: Batch of images (B, 3, H, W) normalized
        extractor: Optional pre-initialized extractor
        device: Device to use (default from config)
        
    Returns:
        features: Patch features (B, feature_dim, patch_h, patch_w)
        patch_grid: (patch_h, patch_w)
    """
    if device is None:
        device = TrainingConfig.DEVICE
    
    if extractor is None:
        extractor = DINOv2Extractor().to(device)
        extractor.eval()
    
    images = images.to(device)
    
    with torch.no_grad():
        features, patch_grid = extractor.get_spatial_features(images)
    
    return features, patch_grid


if __name__ == "__main__":
    # Test DINOv2 extractor
    print("Testing DINOv2 Feature Extractor...")
    print("=" * 60)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create extractor
    extractor = DINOv2Extractor()
    extractor = extractor.to(device)
    extractor.eval()
    
    # Test with dummy input
    # Section 2: Images should be normalized with ImageNet stats
    dummy_input = torch.randn(2, 3, 518, 518).to(device)  # Divisible by 14
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        result = extractor(dummy_input, return_intermediate=True)
    
    print(f"\n[Section 2] Feature extraction results:")
    print(f"  Features shape: {result['features'].shape}")
    print(f"  Patch grid: {result['patch_grid']}")
    
    if 'intermediate' in result:
        print(f"\n  Intermediate features (multi-layer):")
        for name, feat in result['intermediate'].items():
            print(f"    {name}: {feat.shape}")
    
    # Test spatial features
    spatial_feat, grid = extractor.get_spatial_features(dummy_input)
    print(f"\n  Spatial features shape: {spatial_feat.shape}")
    print(f"  Grid (h, w): {grid}")
    
    # Calculate expected dimensions
    expected_patches = (518 // 14) * (518 // 14)
    print(f"\n  Expected patches: {expected_patches} ({518 // 14} x {518 // 14})")
    print(f"  Feature dimension: {DINOv2Config.FEATURE_DIM}")
