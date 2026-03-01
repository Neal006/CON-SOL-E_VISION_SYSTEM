"""Multi-scale CNN decoder with skip connections for segmentation."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        num_groups_1x1 = min(num_groups, out_channels)
        while out_channels % num_groups_1x1 != 0:
            num_groups_1x1 //= 2
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups_1x1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups_1x1, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_groups: int = 32,
        dropout: float = 0.1,
        upsample_scale: int = 2
    ) -> None:
        super().__init__()
        
        self.skip_channels = skip_channels
        self.upsample_scale = upsample_scale
        
        total_in_channels = in_channels + skip_channels
        
        self.conv_block = ConvBlock(
            total_in_channels,
            out_channels,
            num_groups,
            dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        if skip is not None:
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(
                    skip,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv_block(x)
        
        if self.upsample_scale > 1:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_scale,
                mode='bilinear',
                align_corners=False
            )
        
        return x


class MultiScaleDecoder(nn.Module):
    
    def __init__(
        self,
        encoder_dim: int = 768,
        decoder_channels: List[int] = [256, 128, 64],
        num_classes: int = 2,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes
        
        self.stage1 = DecoderStage(
            in_channels=encoder_dim,
            skip_channels=0,
            out_channels=decoder_channels[0],
            num_groups=32,
            dropout=dropout,
            upsample_scale=2
        )
        
        self.stage2 = DecoderStage(
            in_channels=decoder_channels[0],
            skip_channels=encoder_dim,
            out_channels=decoder_channels[1],
            num_groups=16,
            dropout=dropout,
            upsample_scale=2
        )
        
        self.stage3 = DecoderStage(
            in_channels=decoder_channels[1],
            skip_channels=encoder_dim,
            out_channels=decoder_channels[2],
            num_groups=8,
            dropout=dropout,
            upsample_scale=2
        )
        
        self.stage4 = nn.Sequential(
            nn.Conv2d(decoder_channels[2], decoder_channels[2], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, decoder_channels[2]),
            nn.GELU()
        )
        
        self.head = nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1)
        
        self._init_weights()
        
        logger.info(f"Decoder initialized: channels={decoder_channels}, classes={num_classes}")
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        output_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        
        feat_shallow = features['shallow']
        feat_mid = features['mid']
        feat_deep = features['deep']
        
        x = self.stage1(feat_deep)
        
        x = self.stage2(x, feat_mid)
        
        x = self.stage3(x, feat_shallow)
        
        x = self.stage4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.head(x)
        
        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        
        return x
    
    def get_num_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    batch_size = 2
    features = {
        'shallow': torch.randn(batch_size, 768, 37, 37).to(device),
        'mid': torch.randn(batch_size, 768, 37, 37).to(device),
        'deep': torch.randn(batch_size, 768, 37, 37).to(device)
    }
    
    decoder = MultiScaleDecoder(
        encoder_dim=768,
        decoder_channels=[256, 128, 64],
        num_classes=2
    ).to(device)
    
    with torch.no_grad():
        output = decoder(features, output_size=(518, 518))
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Decoder params: {decoder.get_num_params():,}")
                                                            