import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import UNetConfig, TrainingConfig

class ConvBlock(nn.Module):
    """
    Double convolution block for U-Net.
    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm),
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm),
        ])
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate for feature map modulation.
    
    Section 5: Attention gating - feature maps modulated by anomaly intensity.
    Learns to focus on defect-likely regions based on prior information.
    """
    
    def __init__(
        self,
        gate_channels: int,  # From skip connection
        signal_channels: int,  # From decoder path
        inter_channels: int = None
    ):
        """
        Initialize attention gate.
        
        Section 5: Attention mechanism for prior injection.
        
        Args:
            gate_channels: Channels of gating signal (skip connection)
            signal_channels: Channels of input signal (decoder)
            inter_channels: Intermediate channels for attention
        """
        super().__init__()
        
        if inter_channels is None:
            inter_channels = gate_channels // UNetConfig.ATTENTION_REDUCTION
        
        # Section 5: Attention layers
        self.W_gate = nn.Sequential(
            nn.Conv2d(signal_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self,
        x: torch.Tensor,
        gating_signal: torch.Tensor,
        prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply attention gating.
        
        Section 5: Feature maps modulated by attention weights.
        Prior can be incorporated to enhance defect regions.
        
        Args:
            x: Skip connection features (B, C, H, W)
            gating_signal: Decoder features (B, C', H', W')
            prior: Optional anomaly prior (B, 1, H', W')
            
        Returns:
            Attention-weighted features (B, C, H, W)
        """
        # Compute attention coefficients
        g = self.W_gate(gating_signal)
        
        # Upsample gating signal to match x size
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        x_proj = self.W_x(x)
        
        # Section 5: Combine signals and compute attention
        attention = self.relu(g + x_proj)
        attention = self.psi(attention)
        
        # Section 5: If prior available, enhance attention with anomaly prior
        if prior is not None:
            if prior.shape[2:] != attention.shape[2:]:
                prior = F.interpolate(prior, size=attention.shape[2:], mode='bilinear', align_corners=True)
            # Modulate attention with prior (soft multiplication)
            attention = attention * (1 + prior)
            attention = torch.clamp(attention, 0, 1)
        
        # Apply attention to input features
        return x * attention


class DownBlock(nn.Module):
    """Encoder block: MaxPool -> ConvBlock"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, use_batch_norm, dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """
    Decoder block with attention gating.
    
    Section 5: Attention U-Net decoder with prior injection.
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Section 5: Attention gate for skip connection
        if use_attention:
            self.attention = AttentionGate(
                gate_channels=skip_channels,
                signal_channels=in_channels // 2
            )
        
        # Convolution after concatenation
        self.conv = ConvBlock(
            in_channels // 2 + skip_channels,
            out_channels,
            use_batch_norm,
            dropout_rate
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decoder forward pass with attention.
        
        Args:
            x: Decoder features from previous layer
            skip: Skip connection from encoder
            prior: Optional anomaly prior for attention modulation
        """
        x = self.up(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Section 5: Apply attention gating to skip connection
        if self.use_attention:
            skip = self.attention(skip, x, prior)
        
        # Concatenate and convolve
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# =============================================================================
# Section 5: Attention U-Net Architecture
# =============================================================================

class AttentionUNet(nn.Module):
    """
    Attention U-Net for multi-class segmentation.
    
    Section 5: True pixel-wise segmentation network.
    Encoder-decoder with skip connections and attention gates.
    """
    
    def __init__(
        self,
        in_channels: int = None,
        out_channels: int = None,
        encoder_channels: List[int] = None,
        use_attention: bool = None,
        use_batch_norm: bool = None,
        dropout_rate: float = None
    ):
        """
        Initialize Attention U-Net.
        
        Section 5: Architecture configuration.
        
        Args:
            in_channels: Input image channels (default 3)
            out_channels: Number of output classes (default 4)
            encoder_channels: Channel progression for encoder
            use_attention: Whether to use attention gates
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.in_channels = in_channels or UNetConfig.IN_CHANNELS
        self.out_channels = out_channels or UNetConfig.OUT_CHANNELS
        self.encoder_channels = encoder_channels or UNetConfig.ENCODER_CHANNELS
        self.use_attention = use_attention if use_attention is not None else UNetConfig.USE_ATTENTION
        self.use_batch_norm = use_batch_norm if use_batch_norm is not None else UNetConfig.USE_BATCH_NORM
        self.dropout_rate = dropout_rate if dropout_rate is not None else UNetConfig.DROPOUT_RATE
        
        # Section 5: Encoder path
        self.enc1 = ConvBlock(self.in_channels, self.encoder_channels[0], self.use_batch_norm)
        self.enc2 = DownBlock(self.encoder_channels[0], self.encoder_channels[1], self.use_batch_norm, self.dropout_rate)
        self.enc3 = DownBlock(self.encoder_channels[1], self.encoder_channels[2], self.use_batch_norm, self.dropout_rate)
        self.enc4 = DownBlock(self.encoder_channels[2], self.encoder_channels[3], self.use_batch_norm, self.dropout_rate)
        
        # Bottleneck
        self.bottleneck = DownBlock(self.encoder_channels[3], self.encoder_channels[4], self.use_batch_norm, self.dropout_rate)
        
        # Section 5: Decoder path with attention
        self.dec4 = UpBlock(self.encoder_channels[4], self.encoder_channels[3], self.encoder_channels[3], 
                           self.use_attention, self.use_batch_norm, self.dropout_rate)
        self.dec3 = UpBlock(self.encoder_channels[3], self.encoder_channels[2], self.encoder_channels[2],
                           self.use_attention, self.use_batch_norm, self.dropout_rate)
        self.dec2 = UpBlock(self.encoder_channels[2], self.encoder_channels[1], self.encoder_channels[1],
                           self.use_attention, self.use_batch_norm, self.dropout_rate)
        self.dec1 = UpBlock(self.encoder_channels[1], self.encoder_channels[0], self.encoder_channels[0],
                           self.use_attention, self.use_batch_norm, self.dropout_rate)
        
        # Section 5: Multi-class segmentation head
        self.final = nn.Conv2d(self.encoder_channels[0], self.out_channels, kernel_size=1)
    
    def forward(
        self,
        x: torch.Tensor,
        prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Section 5: Prior-guided segmentation with attention.
        
        Args:
            x: Input image (B, 3, H, W)
            prior: Optional anomaly prior (B, 1, H, W) for attention modulation
            
        Returns:
            logits: (B, num_classes, H, W) segmentation logits
        """
        # Section 5: Encoder path
        e1 = self.enc1(x)      # (B, 64, H, W)
        e2 = self.enc2(e1)     # (B, 128, H/2, W/2)
        e3 = self.enc3(e2)     # (B, 256, H/4, W/4)
        e4 = self.enc4(e3)     # (B, 512, H/8, W/8)
        
        # Bottleneck
        b = self.bottleneck(e4)  # (B, 1024, H/16, W/16)
        
        # Section 5: Decoder path with attention and prior injection
        d4 = self.dec4(b, e4, prior)    # (B, 512, H/8, W/8)
        d3 = self.dec3(d4, e3, prior)   # (B, 256, H/4, W/4)
        d2 = self.dec2(d3, e2, prior)   # (B, 128, H/2, W/2)
        d1 = self.dec1(d2, e1, prior)   # (B, 64, H, W)
        
        # Section 5: Output segmentation logits
        logits = self.final(d1)
        
        return logits
    
    def predict(
        self,
        x: torch.Tensor,
        prior: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with probabilities.
        
        Args:
            x: Input image
            prior: Optional anomaly prior
            
        Returns:
            predictions: (B, H, W) class predictions
            probabilities: (B, num_classes, H, W) class probabilities
        """
        logits = self.forward(x, prior)
        probabilities = F.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        
        return predictions, probabilities


class PriorGuidedUNet(nn.Module):
    """
    Complete prior-guided segmentation network.
    
    Section 5: Wrapper that combines attention modulation
    and loss reweighting mechanisms.
    """
    
    def __init__(
        self,
        prior_injection: str = None
    ):
        """
        Initialize Prior-Guided U-Net.
        
        Args:
            prior_injection: Method for prior injection ('attention', 'loss_reweight', 'both')
        """
        super().__init__()
        
        self.prior_injection = prior_injection or UNetConfig.PRIOR_INJECTION
        self.unet = AttentionUNet()
    
    def forward(
        self,
        x: torch.Tensor,
        prior: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with prior injection.
        
        Section 5: Multiple mechanisms for prior utilization.
        
        Args:
            x: Input image (B, 3, H, W)
            prior: Anomaly prior (B, 1, H, W)
            
        Returns:
            Dictionary with 'logits' and optionally 'prior' for loss reweighting
        """
        # Section 5: Attention gating with prior
        use_prior_in_attention = self.prior_injection in ['attention', 'both']
        attention_prior = prior if use_prior_in_attention else None
        
        logits = self.unet(x, attention_prior)
        
        result = {'logits': logits}
        
        # Section 5: Return prior for loss reweighting if needed
        if self.prior_injection in ['loss_reweight', 'both'] and prior is not None:
            result['prior'] = prior
        
        return result


if __name__ == "__main__":
    # Test Attention U-Net
    print("Testing Attention U-Net...")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = AttentionUNet()
    model = model.to(device)
    
    # Test input
    batch_size = 2
    img_size = 256  # Use smaller size for testing
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    prior = torch.rand(batch_size, 1, img_size, img_size).to(device)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Prior shape: {prior.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(x, prior)
    
    print(f"\n[Section 5] Output logits shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {UNetConfig.OUT_CHANNELS}, {img_size}, {img_size})")
    
    # Test predict method
    predictions, probabilities = model.predict(x, prior)
    print(f"\n[Section 5] Predictions shape: {predictions.shape}")
    print(f"  Unique classes: {torch.unique(predictions).tolist()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Section 5] Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
