import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DINOv2Encoder(nn.Module):    
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        skip_layers: List[int] = [3, 7, 11],
        frozen: bool = True,
        pretrained: bool = True
    ) -> None:
        super().__init__()
        
        self.skip_layers = sorted(skip_layers)
        self.features: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        logger.info(f"Loading {model_name}...")
        try:
            self.model = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                pretrained=pretrained
            )
        except Exception as e:
            logger.error(f"Failed to load DINOv2: {e}")
            raise
        
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self.num_heads = self.model.num_heads
        self._register_hooks()
        
        if frozen:
            self.freeze()
        
        logger.info(f"DINOv2 encoder initialized: embed_dim={self.embed_dim}, "
                   f"patch_size={self.patch_size}, frozen={frozen}")
    
    def _register_hooks(self) -> None:
        for layer_idx in self.skip_layers:
            hook = self.model.blocks[layer_idx].register_forward_hook(
                self._create_hook(layer_idx)
            )
            self._hooks.append(hook)
    
    def _create_hook(self, layer_idx: int):
        def hook(module, input, output):
            self.features[layer_idx] = output
        return hook
    
    def _reshape_features(self, features: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, N, C = features.shape
        if N == h * w + 1:
            features = features[:, 1:, :]
        return features.permute(0, 2, 1).reshape(B, C, h, w)
    
    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")
    
    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")
    
    def unfreeze_last_n_layers(self, n: int) -> None:
        total_blocks = len(self.model.blocks)
        for i, block in enumerate(self.model.blocks):
            if i >= total_blocks - n:
                for param in block.parameters():
                    param.requires_grad = True
        logger.info(f"Unfrozen last {n} blocks")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        
        self.features.clear()
        h = H // self.patch_size
        w = W // self.patch_size
        
        _ = self.model.forward_features(x)
        
        output = {}
        for i, layer_idx in enumerate(self.skip_layers):
            feat = self.features[layer_idx]
            feat = self._reshape_features(feat, h, w)
            
            if i == 0:
                output['shallow'] = feat
            elif i == 1:
                output['mid'] = feat
            else:
                output['deep'] = feat
        
        return output
    
    def get_num_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def __del__(self):
        for hook in self._hooks:
            hook.remove()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    encoder = DINOv2Encoder(
        model_name="dinov2_vitb14",
        skip_layers=[3, 7, 11],
        frozen=True
    ).to(device)
    
    x = torch.randn(2, 3, 518, 518).to(device)
    with torch.no_grad():
        features = encoder(x)
    
    print(f"\nFeature shapes:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    print(f"\nTotal params: {encoder.get_num_params():,}")
    print(f"Trainable params: {encoder.get_num_params(trainable_only=True):,}")
