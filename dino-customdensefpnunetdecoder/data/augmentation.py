"""Data augmentation pipelines using Albumentations."""

import logging
from typing import Callable, Dict, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(config: dict) -> A.Compose:
    """Create training augmentation pipeline.
    
    Args:
        config: Configuration dictionary with 'augmentation' and 'data' sections.
        
    Returns:
        Albumentations Compose transform.
    """
    aug_cfg = config.get('augmentation', {})
    data_cfg = config.get('data', {})
    image_size = data_cfg.get('image_size', 518)
    
    transform = A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=aug_cfg.get('horizontal_flip', 0.5)),
        A.VerticalFlip(p=aug_cfg.get('vertical_flip', 0.3)),
        A.Rotate(
            limit=aug_cfg.get('rotation_limit', 15),
            border_mode=0,
            p=0.5
        ),
        
        # Color transforms
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.get('brightness_limit', 0.2),
            contrast_limit=aug_cfg.get('contrast_limit', 0.2),
            p=0.5
        ),
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=aug_cfg.get('gaussian_blur_p', 0.3)
        ),
        
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(image_size // 40, image_size // 20),
            hole_width_range=(image_size // 40, image_size // 20),
            fill=0,
            p=0.3
        ),
        A.ElasticTransform(p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),
        # Normalize and convert to tensor
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    
    logger.info("Training transform created with augmentations")
    return transform


def get_val_transform(config: dict) -> A.Compose:
    """Create validation/test transform pipeline (no augmentation).
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Albumentations Compose transform.
    """
    transform = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    
    logger.info("Validation transform created")
    return transform


def get_inference_transform(image_size: int = 518) -> A.Compose:
    """Create inference transform pipeline.
    
    Args:
        image_size: Target size for resizing.
        
    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def denormalize(image: np.ndarray) -> np.ndarray:
    """Denormalize image from ImageNet normalization.
    
    Args:
        image: Normalized image array (H, W, C) or (C, H, W).
        
    Returns:
        Denormalized image in [0, 255] range.
    """
    if image.shape[0] == 3:  # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    
    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


if __name__ == "__main__":
    # Test augmentation pipeline
    logging.basicConfig(level=logging.INFO)
    
    # Mock config
    config = {
        'data': {'image_size': 518},
        'augmentation': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.3,
            'rotation_limit': 15,
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'gaussian_blur_p': 0.3
        }
    }
    
    train_transform = get_train_transform(config)
    val_transform = get_val_transform(config)
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 2, (518, 518), dtype=np.uint8)
    
    transformed = train_transform(image=dummy_image, mask=dummy_mask)
    
    print(f"Image shape: {transformed['image'].shape}")
    print(f"Mask shape: {transformed['mask'].shape}")
    print(f"Image dtype: {transformed['image'].dtype}")
    print(f"Mask dtype: {transformed['mask'].dtype}")
