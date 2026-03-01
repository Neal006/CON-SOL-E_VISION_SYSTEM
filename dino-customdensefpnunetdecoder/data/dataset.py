"""Dataset and data loading utilities for defect segmentation with YOLO format support."""

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def yolo_polygon_to_mask(
    label_path: Path,
    image_size: Tuple[int, int],
    num_classes: int = 3
) -> np.ndarray:
    """Convert YOLO polygon annotations to segmentation mask.
    
    YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ... (normalized coords)
    
    Args:
        label_path: Path to YOLO label .txt file.
        image_size: Original image size (height, width).
        num_classes: Number of classes (excluding background).
        
    Returns:
        Segmentation mask of shape (H, W) with class indices.
        Background = 0, Classes = 1, 2, 3, ...
    """
    h, w = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if not label_path.exists():
        return mask
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:  # Need at least class_id + 3 points (6 coords)
                continue
            
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Convert normalized coords to pixel coords
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = int(coords[i] * w)
                    y = int(coords[i + 1] * h)
                    points.append([x, y])
            
            if len(points) >= 3:
                points = np.array(points, dtype=np.int32)
                # Fill polygon with class_id + 1 (0 is background)
                cv2.fillPoly(mask, [points], class_id + 1)
    
    except Exception as e:
        logger.warning(f"Error parsing label {label_path}: {e}")
    
    return mask


class DefectDataset(Dataset):
    """Dataset for defect segmentation with YOLO format label support.
    
    Supports:
    - Image formats: .jpg, .png, .bmp, .tif
    - Label formats: YOLO segmentation (.txt with polygons)
    
    Attributes:
        image_dir: Directory containing images.
        label_dir: Directory containing YOLO label files.
        image_size: Target size for resizing.
        transform: Optional albumentations transform.
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    def __init__(
        self,
        image_dir: str,
        label_dir: Optional[str] = None,
        image_size: int = 518,
        transform: Optional[Callable] = None,
        num_classes: int = 3,
        is_training: bool = True
    ) -> None:
        """Initialize dataset.
        
        Args:
            image_dir: Path to image directory.
            label_dir: Path to YOLO label directory (.txt files).
            image_size: Target image size.
            transform: Albumentations transform pipeline.
            num_classes: Number of defect classes (excluding background).
            is_training: Whether this is training data.
        """
        super().__init__()
        
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.image_size = image_size
        self.transform = transform
        self.num_classes = num_classes
        self.is_training = is_training
        
        # Find all images
        self.image_files = self._find_images()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        logger.info(f"Dataset: {len(self.image_files)} images from {image_dir}")
    
    def _find_images(self) -> List[Path]:
        """Find all valid image files."""
        images = []
        for ext in self.SUPPORTED_FORMATS:
            images.extend(self.image_dir.glob(f"*{ext}"))
            images.extend(self.image_dir.glob(f"*{ext.upper()}"))
        return sorted(images)
    
    def _find_label(self, image_path: Path) -> Optional[Path]:
        """Find corresponding label file for an image."""
        if self.label_dir is None:
            return None
        
        label_path = self.label_dir / f"{image_path.stem}.txt"
        return label_path if label_path.exists() else None
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load and convert image to RGB array."""
        try:
            image = Image.open(path).convert('RGB')
            return np.array(image)
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            raise
    
    def _resize(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Resize image and mask to target size."""
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        if mask is not None:
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        return image, mask
    
    def _normalize(self, image: np.ndarray) -> torch.Tensor:
        """Normalize image with ImageNet stats and convert to tensor."""
        image = image.astype(np.float32) / 255.0
        
        mean = np.array(IMAGENET_MEAN)
        std = np.array(IMAGENET_STD)
        image = (image - mean) / std
        
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).float()
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get image-mask pair.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary with 'image' tensor (C, H, W) and 'mask' tensor (H, W).
        """
        image_path = self.image_files[idx]
        
        # Load image
        image = self._load_image(image_path)
        original_size = (image.shape[0], image.shape[1])
        
        # Load or create mask from YOLO labels
        mask = None
        label_path = self._find_label(image_path)
        
        if label_path:
            mask = yolo_polygon_to_mask(label_path, original_size, self.num_classes)
        else:
            # Create empty mask if no label file
            mask = np.zeros(original_size, dtype=np.uint8)
        
        # Resize
        image, mask = self._resize(image, mask)
        
        # Apply augmentations
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensors if not already (albumentations may have done this)
        if isinstance(image, np.ndarray):
            image = self._normalize(image)
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        
        return {'image': image, 'mask': mask}


def get_dataloaders(
    config: dict,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration dictionary.
        train_transform: Training augmentation pipeline.
        val_transform: Validation transform pipeline.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_cfg = config['data']
    paths_cfg = config['paths']
    training_cfg = config['training']
    model_cfg = config['model']
    
    data_root = Path(paths_cfg['data_root'])
    
    # Check for train/val/test structure or single directory
    train_images = data_root / 'train' / 'images'
    train_labels = data_root / 'train' / 'labels'
    
    if train_images.exists():
        # Standard YOLO structure: data_root/train/images, data_root/train/labels
        train_dataset = DefectDataset(
            image_dir=str(train_images),
            label_dir=str(train_labels),
            image_size=data_cfg['image_size'],
            transform=train_transform,
            num_classes=model_cfg['num_classes'] - 1,  # Subtract 1 for background
            is_training=True
        )
        
        # Check for val directory
        val_images = data_root / 'val' / 'images'
        val_labels = data_root / 'val' / 'labels'
        
        if val_images.exists():
            val_dataset = DefectDataset(
                image_dir=str(val_images),
                label_dir=str(val_labels),
                image_size=data_cfg['image_size'],
                transform=val_transform,
                num_classes=model_cfg['num_classes'] - 1,
                is_training=False
            )
            test_dataset = val_dataset  # Use val as test if no separate test
        else:
            # Split train into train/val/test
            total_size = len(train_dataset)
            train_size = int(total_size * 0.8)
            val_size = int(total_size * 0.1)
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                train_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
    else:
        # Flat structure: data_root/images, data_root/labels
        image_dir = data_root / 'images'
        label_dir = data_root / 'labels'
        
        full_dataset = DefectDataset(
            image_dir=str(image_dir),
            label_dir=str(label_dir),
            image_size=data_cfg['image_size'],
            transform=None,
            num_classes=model_cfg['num_classes'] - 1,
            is_training=True
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(total_size * data_cfg.get('train_split', 0.8))
        val_size = int(total_size * data_cfg.get('val_split', 0.1))
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True
    )
    
    logger.info(f"Data splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    logging.basicConfig(level=logging.INFO)
    
    print("Dataset module loaded successfully")
    print(f"Supported formats: {DefectDataset.SUPPORTED_FORMATS}")
    
    # Test YOLO polygon to mask conversion
    test_label = "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"
    with open("test_label.txt", "w") as f:
        f.write(test_label)
    
    mask = yolo_polygon_to_mask(Path("test_label.txt"), (100, 100), num_classes=3)
    print(f"Test mask shape: {mask.shape}")
    print(f"Unique values: {np.unique(mask)}")
    
    os.remove("test_label.txt")
