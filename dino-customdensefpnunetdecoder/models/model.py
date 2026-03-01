"""Complete segmentation model combining DINOv2 encoder and multi-scale decoder."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import DINOv2Encoder
from .decoder import MultiScaleDecoder

logger = logging.getLogger(__name__)


class SegmentationModel(nn.Module):
    """DINOv2-based segmentation model with multi-scale decoder.
    
    Combines frozen DINOv2 encoder with trainable multi-scale CNN decoder
    that uses skip connections from multiple encoder layers.
    
    Attributes:
        encoder: DINOv2 vision transformer encoder.
        decoder: Multi-scale CNN decoder with skip connections.
    """
    
    def __init__(
        self,
        encoder_name: str = "dinov2_vitb14",
        skip_layers: List[int] = [3, 7, 11],
        decoder_channels: List[int] = [256, 128, 64],
        num_classes: int = 2,
        encoder_frozen: bool = True,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # Initialize encoder
        self.encoder = DINOv2Encoder(
            model_name=encoder_name,
            skip_layers=skip_layers,
            frozen=encoder_frozen
        )
        
        # Initialize decoder
        self.decoder = MultiScaleDecoder(
            encoder_dim=self.encoder.embed_dim,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.num_classes = num_classes
        
        logger.info(f"Segmentation model initialized: {encoder_name}, "
                   f"classes={num_classes}, encoder_frozen={encoder_frozen}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images of shape (B, 3, H, W).
            
        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        input_size = (x.shape[2], x.shape[3])
        
        # Extract multi-scale features
        features = self.encoder(x)
        
        # Decode to segmentation map
        logits = self.decoder(features, output_size=input_size)
        
        return logits
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        self.encoder.freeze()
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters."""
        self.encoder.unfreeze()
    
    def unfreeze_encoder_last_n(self, n: int) -> None:
        """Unfreeze last n encoder blocks for fine-tuning."""
        self.encoder.unfreeze_last_n_layers(n)
    
    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts."""
        encoder_total = self.encoder.get_num_params()
        encoder_trainable = self.encoder.get_num_params(trainable_only=True)
        decoder_total = self.decoder.get_num_params()
        decoder_trainable = self.decoder.get_num_params(trainable_only=True)
        
        return {
            'encoder_total': encoder_total,
            'encoder_trainable': encoder_trainable,
            'decoder_total': decoder_total,
            'decoder_trainable': decoder_trainable,
            'total': encoder_total + decoder_total,
            'trainable': encoder_trainable + decoder_trainable
        }
    
    def print_summary(self) -> None:
        """Print model summary."""
        params = self.get_num_params()
        print("\n" + "=" * 50)
        print("MODEL SUMMARY")
        print("=" * 50)
        print(f"Encoder params:  {params['encoder_total']:>12,} (trainable: {params['encoder_trainable']:,})")
        print(f"Decoder params:  {params['decoder_total']:>12,} (trainable: {params['decoder_trainable']:,})")
        print("-" * 50)
        print(f"Total params:    {params['total']:>12,}")
        print(f"Trainable:       {params['trainable']:>12,}")
        print("=" * 50 + "\n")


def build_model(config: dict) -> SegmentationModel:
    """Build model from configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'model' section.
        
    Returns:
        Initialized SegmentationModel.
    """
    model_cfg = config['model']
    
    return SegmentationModel(
        encoder_name=model_cfg.get('encoder', 'dinov2_vitb14'),
        skip_layers=model_cfg.get('skip_layers', [3, 7, 11]),
        decoder_channels=model_cfg.get('decoder_channels', [256, 128, 64]),
        num_classes=model_cfg.get('num_classes', 2),
        encoder_frozen=model_cfg.get('encoder_frozen', True)
    )


if __name__ == "__main__":
    # Test complete model
    logging.basicConfig(level=logging.INFO)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Create model
    model = SegmentationModel(
        encoder_name="dinov2_vitb14",
        skip_layers=[3, 7, 11],
        decoder_channels=[256, 128, 64],
        num_classes=2,
        encoder_frozen=True
    ).to(device)
    
    model.print_summary()
    
    # Test forward pass
    x = torch.randn(2, 3, 518, 518).to(device)
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test memory usage
    if torch.cuda.is_available():
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
