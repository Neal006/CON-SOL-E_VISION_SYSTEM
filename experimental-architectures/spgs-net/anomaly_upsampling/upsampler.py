"""
Anomaly Prior Upsampler for SPGS-Net
=====================================
Bilinear upsampling of patch-level anomaly heatmaps to full resolution.

Section 4 of Architecture: Anomaly Prior Upsampling & Alignment
- Patch-level anomaly heatmap bilinearly upsampled to match original image resolution
- Spatial alignment ensured to preserve correspondence between patches and pixels
- Upsampled anomaly map acts as soft spatial prior indicating defect-likely regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
from scipy.ndimage import gaussian_filter
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import UpsamplingConfig, DINOv2Config


class AnomalyUpsampler(nn.Module):
    """
    Upsample patch-level anomaly heatmap to full image resolution.
    
    Section 4: Anomaly prior upsampling & alignment.
    Preserves spatial correspondence between patch locations and pixels.
    """
    
    def __init__(
        self,
        upsample_mode: str = None,
        align_corners: bool = None,
        apply_smoothing: bool = None,
        smoothing_kernel_size: int = None,
        smoothing_sigma: float = None
    ):
        """
        Initialize anomaly upsampler.
        
        Section 4: Bilinear upsampling for smooth spatial prior.
        
        Args:
            upsample_mode: Interpolation mode (default 'bilinear')
            align_corners: Whether to align corners in interpolation
            apply_smoothing: Whether to apply Gaussian smoothing
            smoothing_kernel_size: Size of Gaussian kernel
            smoothing_sigma: Sigma for Gaussian smoothing
        """
        super().__init__()
        
        self.upsample_mode = upsample_mode or UpsamplingConfig.UPSAMPLE_MODE
        self.align_corners = align_corners if align_corners is not None else UpsamplingConfig.ALIGN_CORNERS
        self.apply_smoothing = apply_smoothing if apply_smoothing is not None else UpsamplingConfig.APPLY_SMOOTHING
        self.smoothing_kernel_size = smoothing_kernel_size or UpsamplingConfig.SMOOTHING_KERNEL_SIZE
        self.smoothing_sigma = smoothing_sigma or UpsamplingConfig.SMOOTHING_SIGMA
        
        # Create Gaussian kernel for smoothing
        if self.apply_smoothing:
            self.smoothing_kernel = self._create_gaussian_kernel(
                self.smoothing_kernel_size,
                self.smoothing_sigma
            )
        else:
            self.smoothing_kernel = None
    
    def _create_gaussian_kernel(
        self,
        kernel_size: int,
        sigma: float
    ) -> torch.Tensor:
        """
        Create 2D Gaussian kernel for smooth prior.
        
        Section 4: Optional smoothing for seamless prior map.
        """
        # Create 1D Gaussian
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        
        # Create 2D kernel via outer product
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        gauss_2d = gauss_2d / gauss_2d.sum()
        
        # Reshape for conv2d: (out_channels, in_channels, H, W)
        return gauss_2d.view(1, 1, kernel_size, kernel_size)
    
    def forward(
        self,
        anomaly_heatmap: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Upsample anomaly heatmap to target resolution.
        
        Section 4: Bilinear upsampling to match original image resolution.
        Spatial alignment preserved for patch-pixel correspondence.
        
        Args:
            anomaly_heatmap: (B, 1, patch_h, patch_w) or (B, patch_h, patch_w)
            target_size: (target_h, target_w) full image resolution
            
        Returns:
            upsampled_prior: (B, 1, target_h, target_w) soft spatial prior
        """
        # Ensure 4D tensor (B, C, H, W)
        if len(anomaly_heatmap.shape) == 3:
            anomaly_heatmap = anomaly_heatmap.unsqueeze(1)
        elif len(anomaly_heatmap.shape) == 2:
            anomaly_heatmap = anomaly_heatmap.unsqueeze(0).unsqueeze(0)
        
        # Section 4: Bilinear upsampling
        upsampled = F.interpolate(
            anomaly_heatmap,
            size=target_size,
            mode=self.upsample_mode,
            align_corners=self.align_corners if self.upsample_mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
        )
        
        # Section 4: Optional Gaussian smoothing for seamless prior
        if self.apply_smoothing and self.smoothing_kernel is not None:
            kernel = self.smoothing_kernel.to(upsampled.device)
            padding = self.smoothing_kernel_size // 2
            upsampled = F.conv2d(upsampled, kernel, padding=padding)
        
        # Normalize to [0, 1]
        upsampled = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)
        
        return upsampled
    
    def get_spatial_prior(
        self,
        anomaly_heatmap: torch.Tensor,
        image_size: Tuple[int, int],
        return_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get soft spatial prior from patch-level heatmap.
        
        Section 4: Upsampled anomaly map as soft spatial prior
        indicating defect-likely regions.
        
        Args:
            anomaly_heatmap: Patch-level anomaly scores
            image_size: (H, W) of original image
            return_numpy: Whether to return numpy array
            
        Returns:
            Soft spatial prior map
        """
        prior = self.forward(anomaly_heatmap, image_size)
        
        if return_numpy:
            return prior.squeeze().cpu().numpy()
        return prior


def upsample_anomaly_map(
    patch_heatmap: Union[np.ndarray, torch.Tensor],
    target_size: Tuple[int, int],
    apply_smoothing: bool = True,
    smoothing_sigma: float = None
) -> np.ndarray:
    """
    Convenience function to upsample patch-level anomaly heatmap.
    
    Section 4: Bilinear interpolation with optional smoothing.
    
    Args:
        patch_heatmap: (patch_h, patch_w) or (B, patch_h, patch_w) heatmap
        target_size: (target_h, target_w) output size
        apply_smoothing: Whether to smooth after upsampling
        smoothing_sigma: Sigma for Gaussian smoothing
        
    Returns:
        upsampled: (target_h, target_w) or (B, target_h, target_w) upsampled prior
    """
    was_numpy = isinstance(patch_heatmap, np.ndarray)
    single_image = len(patch_heatmap.shape) == 2
    
    # Convert to tensor
    if was_numpy:
        patch_heatmap = torch.from_numpy(patch_heatmap).float()
    
    # Add batch and channel dims if needed
    if single_image:
        patch_heatmap = patch_heatmap.unsqueeze(0).unsqueeze(0)
    elif len(patch_heatmap.shape) == 3:
        patch_heatmap = patch_heatmap.unsqueeze(1)
    
    # Section 4: Bilinear upsampling
    upsampled = F.interpolate(
        patch_heatmap,
        size=target_size,
        mode='bilinear',
        align_corners=True
    )
    
    # Section 4: Optional Gaussian smoothing
    if apply_smoothing:
        sigma = smoothing_sigma or UpsamplingConfig.SMOOTHING_SIGMA
        # Apply per-image smoothing
        result = []
        for i in range(upsampled.shape[0]):
            smoothed = gaussian_filter(
                upsampled[i, 0].numpy(),
                sigma=sigma
            )
            result.append(smoothed)
        upsampled = np.stack(result, axis=0)
    else:
        upsampled = upsampled.squeeze(1).numpy()
    
    # Normalize
    upsampled = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)
    
    # Return single image if input was single
    if single_image:
        upsampled = upsampled[0]
    
    return upsampled


class SpatialPriorGenerator(nn.Module):
    """
    Generate spatial prior for segmentation network.
    
    Section 4: Complete pipeline from patch features to spatial prior.
    Combines anomaly scoring and upsampling.
    """
    
    def __init__(self, patch_size: int = None):
        """
        Initialize spatial prior generator.
        
        Args:
            patch_size: DINOv2 patch size (default 14)
        """
        super().__init__()
        self.patch_size = patch_size or DINOv2Config.PATCH_SIZE
        self.upsampler = AnomalyUpsampler()
    
    def forward(
        self,
        patch_anomaly_scores: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Generate spatial prior from patch-level anomaly scores.
        
        Section 4: Full pipeline from scores to upsampled prior.
        
        Args:
            patch_anomaly_scores: (B, patch_h, patch_w) anomaly scores
            image_size: (H, W) target image size
            
        Returns:
            spatial_prior: (B, 1, H, W) soft spatial prior
        """
        return self.upsampler(patch_anomaly_scores, image_size)


if __name__ == "__main__":
    # Test anomaly upsampling
    print("Testing Anomaly Upsampler...")
    print("=" * 60)
    
    # Section 4: Test with dummy patch heatmap
    batch_size = 2
    patch_h, patch_w = 37, 37  # For 518x518 image with 14x14 patches
    target_h, target_w = 518, 518
    
    # Create dummy patch heatmap
    patch_heatmap = torch.rand(batch_size, patch_h, patch_w)
    
    print(f"Input patch heatmap shape: {patch_heatmap.shape}")
    print(f"Target size: ({target_h}, {target_w})")
    
    # Test upsampler module
    upsampler = AnomalyUpsampler()
    upsampled = upsampler(patch_heatmap, (target_h, target_w))
    
    print(f"\n[Section 4] Upsampled prior shape: {upsampled.shape}")
    print(f"  Min: {upsampled.min():.4f}, Max: {upsampled.max():.4f}")
    
    # Test convenience function
    single_heatmap = patch_heatmap[0].numpy()
    upsampled_np = upsample_anomaly_map(single_heatmap, (target_h, target_w))
    
    print(f"\n[Section 4] Numpy upsampled shape: {upsampled_np.shape}")
    print(f"  Min: {upsampled_np.min():.4f}, Max: {upsampled_np.max():.4f}")
    
    # Verify spatial alignment
    print(f"\n[Section 4] Spatial alignment verification:")
    print(f"  Upsampling factor: {target_h / patch_h:.2f}x")
    print(f"  = Patch size (14) as expected: {abs(target_h / patch_h - 14) < 0.5}")
