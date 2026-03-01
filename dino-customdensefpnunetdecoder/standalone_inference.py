
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class ConvBlock(nn.Module):    
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32, dropout: float = 0.1):
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
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, 
                 num_groups: int = 32, dropout: float = 0.1, upsample_scale: int = 2):
        super().__init__()
        self.skip_channels = skip_channels
        self.upsample_scale = upsample_scale
        self.conv_block = ConvBlock(in_channels + skip_channels, out_channels, num_groups, dropout)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        if self.upsample_scale > 1:
            x = F.interpolate(x, scale_factor=self.upsample_scale, mode='bilinear', align_corners=False)
        return x


class MultiScaleDecoder(nn.Module):
    def __init__(self, encoder_dim: int = 768, decoder_channels: List[int] = [256, 128, 64],
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.stage1 = DecoderStage(encoder_dim, 0, decoder_channels[0], 32, dropout, 2)
        self.stage2 = DecoderStage(decoder_channels[0], encoder_dim, decoder_channels[1], 16, dropout, 2)
        self.stage3 = DecoderStage(decoder_channels[1], encoder_dim, decoder_channels[2], 8, dropout, 2)
        self.stage4 = nn.Sequential(
            nn.Conv2d(decoder_channels[2], decoder_channels[2], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, decoder_channels[2]),
            nn.GELU()
        )
        self.head = nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1)
    
    def forward(self, features: Dict[str, torch.Tensor], output_size: Optional[Tuple[int, int]] = None):
        x = self.stage1(features['deep'])
        x = self.stage2(x, features['mid'])
        x = self.stage3(x, features['shallow'])
        x = self.stage4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.head(x)
        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        return x


class DINOv2Encoder(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14", skip_layers: List[int] = [3, 7, 11]):
        super().__init__()
        self.skip_layers = sorted(skip_layers)
        self.features: Dict[int, torch.Tensor] = {}
        self._hooks = []
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self._register_hooks()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _register_hooks(self):
        for layer_idx in self.skip_layers:
            hook = self.model.blocks[layer_idx].register_forward_hook(self._create_hook(layer_idx))
            self._hooks.append(hook)
    
    def _create_hook(self, layer_idx: int):
        def hook(module, input, output):
            self.features[layer_idx] = output
        return hook
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        self.features.clear()
        h, w = H // self.patch_size, W // self.patch_size
        _ = self.model.forward_features(x)
        output = {}
        for i, layer_idx in enumerate(self.skip_layers):
            feat = self.features[layer_idx]
            feat = feat[:, 1:, :].permute(0, 2, 1).reshape(B, -1, h, w)
            output[['shallow', 'mid', 'deep'][i]] = feat
        return output


class SegmentationModel(nn.Module):
    def __init__(self, encoder_name: str = "dinov2_vitb14", skip_layers: List[int] = [3, 7, 11],
                 decoder_channels: List[int] = [256, 128, 64], num_classes: int = 2):
        super().__init__()
        self.encoder = DINOv2Encoder(model_name=encoder_name, skip_layers=skip_layers)
        self.decoder = MultiScaleDecoder(
            encoder_dim=self.encoder.embed_dim,
            decoder_channels=decoder_channels,
            num_classes=num_classes
        )
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = (x.shape[2], x.shape[3])
        features = self.encoder(x)
        return self.decoder(features, output_size=input_size)


# ============================================================================
# STANDALONE PREDICTOR CLASS
# ============================================================================

class StandalonePredictor:
    """Self-contained predictor that only needs the exported .pth file.
    
    Example:
        predictor = StandalonePredictor('deployed_model.pth', device='cuda')
        
        # Single image prediction
        mask = predictor.predict('image.jpg')
        
        # Get class probabilities
        mask, probs = predictor.predict('image.jpg', return_probs=True)
        
        # Batch prediction
        masks = predictor.predict_batch(['img1.jpg', 'img2.jpg'])
    """
    
    # Default color map for visualization
    DEFAULT_COLORS = [
        [0, 0, 0],       # Background - black
        [255, 0, 0],     # Class 1 - red (Dust)
        [0, 255, 0],     # Class 2 - green (RunDown)
        [0, 0, 255],     # Class 3 - blue (Scratch)
        [255, 255, 0],   # Class 4 - yellow
        [255, 0, 255],   # Class 5 - magenta
    ]
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """Initialize predictor.
        
        Args:
            model_path: Path to exported .pth file.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        print(f"Using device: {self.device}")
        
        # Load exported model
        print(f"Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get configs
        model_cfg = checkpoint['model_config']
        self.preprocess_cfg = checkpoint['preprocessing']
        self.class_info = checkpoint['classes']
        self.metadata = checkpoint.get('metadata', {})
        
        # Build model
        self.model = SegmentationModel(
            encoder_name=model_cfg['encoder'],
            skip_layers=model_cfg['skip_layers'],
            decoder_channels=model_cfg['decoder_channels'],
            num_classes=model_cfg['num_classes']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"  Classes: {self.class_info['names']}")
        print(f"  Image size: {self.preprocess_cfg['image_size']}")
    
    def _preprocess(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for inference."""
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        if isinstance(image, Image.Image):
            original_size = (image.height, image.width)
            image = np.array(image)
        else:
            original_size = (image.shape[0], image.shape[1])
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)
        
        # Resize
        img_size = self.preprocess_cfg['image_size']
        image = np.array(Image.fromarray(image).resize((img_size, img_size), Image.BILINEAR))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array(self.preprocess_cfg['mean'])
        std = np.array(self.preprocess_cfg['std'])
        image = (image - mean) / std
        
        # To tensor (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device), original_size
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_probs: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Run inference on a single image.
        
        Args:
            image: Image path, numpy array, or PIL Image.
            return_probs: Whether to return class probabilities.
            
        Returns:
            Predicted mask (H, W) with class indices.
            If return_probs=True, also returns probabilities (C, H, W).
        """
        tensor, original_size = self._preprocess(image)
        
        # Inference
        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_mask = outputs.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Resize to original size
        pred_mask = np.array(
            Image.fromarray(pred_mask.astype(np.uint8)).resize(
                (original_size[1], original_size[0]), Image.NEAREST
            )
        )
        
        if return_probs:
            probs = probs.squeeze(0).cpu().numpy()
            return pred_mask, probs
        return pred_mask
    
    def predict_batch(self, images: List[Union[str, Path]]) -> List[np.ndarray]:
        """Run inference on multiple images."""
        return [self.predict(img) for img in images]
    
    def visualize(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        alpha: float = 0.5,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Predict and create overlay visualization.
        
        Args:
            image: Input image.
            alpha: Overlay transparency (0-1).
            save_path: Optional path to save result.
            
        Returns:
            Overlay image as numpy array.
        """
        # Load original
        if isinstance(image, (str, Path)):
            original = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            original = np.array(image.convert('RGB'))
        else:
            original = image if len(image.shape) == 3 else np.stack([image]*3, axis=-1)
        
        # Predict
        mask = self.predict(image)
        
        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx in range(self.class_info['num_classes']):
            color = self.DEFAULT_COLORS[class_idx % len(self.DEFAULT_COLORS)]
            colored_mask[mask == class_idx] = color
        
        # Create overlay
        overlay = (original * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
        
        if save_path:
            Image.fromarray(overlay).save(save_path)
            print(f"Saved visualization to {save_path}")
        
        return overlay
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.class_info['names']
    
    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            'classes': self.class_info,
            'preprocessing': self.preprocess_cfg,
            'metadata': self.metadata
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Standalone inference')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='output.png', help='Output path')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    predictor = StandalonePredictor(args.model, device=args.device)
    predictor.visualize(args.image, save_path=args.output)
    print("Done!")
