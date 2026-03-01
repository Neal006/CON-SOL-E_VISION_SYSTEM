"""Single image or batch inference script."""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
import yaml

from data.augmentation import get_inference_transform, denormalize
from models.model import SegmentationModel, build_model
from utils.visualization import mask_to_rgb, denormalize_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Predictor:
    """Inference class for segmentation predictions.
    
    Handles preprocessing, model inference, and postprocessing.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint.
            config_path: Optional path to config file.
            device: Device to use (auto-detected if None).
        """
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f'Using device: {self.device}')
        
        # Load checkpoint
        logger.info(f'Loading checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load config
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        elif config_path:
            self.config = load_config(config_path)
        else:
            raise ValueError('Config not found in checkpoint, please provide config_path')
        
        # Build model
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup transform
        self.image_size = self.config['data']['image_size']
        self.transform = get_inference_transform(self.image_size)
        
        logger.info(f'Model loaded from epoch {checkpoint.get("epoch", "unknown")}')
    
    def preprocess(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for inference.
        
        Args:
            image: Image path, numpy array, or PIL Image.
            
        Returns:
            Tuple of (preprocessed tensor, original size).
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, Image.Image):
            original_size = image.size[::-1]  # (H, W)
            image = np.array(image)
        else:
            original_size = (image.shape[0], image.shape[1])
        
        # Apply transforms
        transformed = self.transform(image=image)
        tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        return tensor, original_size
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_probs: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run inference on a single image.
        
        Args:
            image: Input image.
            return_probs: Whether to return class probabilities.
            
        Returns:
            Tuple of (predicted mask, optional probabilities).
        """
        tensor, original_size = self.preprocess(image)
        
        # Inference
        start_time = time.time()
        outputs = self.model(tensor)
        inference_time = (time.time() - start_time) * 1000
        logger.info(f'Inference time: {inference_time:.2f} ms')
        
        # Get predictions
        probs = torch.softmax(outputs, dim=1)
        pred_mask = outputs.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Resize to original size
        pred_mask = np.array(
            Image.fromarray(pred_mask.astype(np.uint8)).resize(
                (original_size[1], original_size[0]),
                Image.NEAREST
            )
        )
        
        if return_probs:
            probs = probs.squeeze(0).cpu().numpy()
            return pred_mask, probs
        
        return pred_mask, None
    
    def predict_and_save(
        self,
        image_path: str,
        output_path: str,
        save_overlay: bool = True,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Predict and save results.
        
        Args:
            image_path: Path to input image.
            output_path: Path to save prediction.
            save_overlay: Whether to save overlay image.
            alpha: Overlay transparency.
            
        Returns:
            Predicted mask.
        """
        # Load original image
        original = Image.open(image_path).convert('RGB')
        original_np = np.array(original)
        
        # Predict
        pred_mask, _ = self.predict(image_path)
        
        # Save mask
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mask_rgb = mask_to_rgb(pred_mask)
        Image.fromarray(mask_rgb).save(output_path)
        logger.info(f'Saved mask to {output_path}')
        
        # Save overlay
        if save_overlay:
            overlay = (original_np * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
            overlay_path = output_path.with_stem(output_path.stem + '_overlay')
            Image.fromarray(overlay).save(overlay_path)
            logger.info(f'Saved overlay to {overlay_path}')
        
        return pred_mask


def main():
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional if in checkpoint)')
    parser.add_argument('--output', type=str, default='prediction.png',
                       help='Output path for prediction')
    parser.add_argument('--no_overlay', action='store_true',
                       help='Do not save overlay image')
    args = parser.parse_args()
    
    # Create predictor
    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )
    
    image_path = Path(args.image)
    
    if image_path.is_file():
        # Single image
        predictor.predict_and_save(
            str(image_path),
            args.output,
            save_overlay=not args.no_overlay
        )
    elif image_path.is_dir():
        # Directory of images
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(image_path.glob('*.jpg')) + \
                     list(image_path.glob('*.png')) + \
                     list(image_path.glob('*.bmp'))
        
        logger.info(f'Found {len(image_files)} images')
        
        for img_path in image_files:
            out_path = output_dir / f'{img_path.stem}_pred.png'
            predictor.predict_and_save(
                str(img_path),
                str(out_path),
                save_overlay=not args.no_overlay
            )
    else:
        raise ValueError(f'Invalid image path: {image_path}')


if __name__ == '__main__':
    main()
