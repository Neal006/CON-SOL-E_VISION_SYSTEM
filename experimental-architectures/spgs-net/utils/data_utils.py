"""
Data Utilities for SPGS-Net
===========================
YOLO polygon label parsing, mask generation, and dataset handling.

Section 1 of Architecture: Input Acquisition & Preprocessing
- Images converted to grayscale (done via Roboflow)
- Grayscale replicated to 3 channels for transformer input
- Standard normalization using ImageNet statistics
- No resizing to preserve geometric fidelity
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PathConfig, DINOv2Config, DataConfig, TrainingConfig
)


# =============================================================================
# Section 1: Input Acquisition & Preprocessing
# Parse YOLO polygon labels and convert to binary masks
# =============================================================================

def parse_yolo_polygon_label(label_path: Union[str, Path]) -> List[Dict]:
    """
    Parse YOLO polygon format label file.
    
    YOLO polygon format: class_id x1 y1 x2 y2 x3 y3 ... xn yn
    All coordinates are normalized [0, 1].
    
    Args:
        label_path: Path to the label .txt file
        
    Returns:
        List of dictionaries with 'class_id' and 'polygon' (normalized coords)
    """
    annotations = []
    label_path = Path(label_path)
    
    if not label_path.exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # Need at least class_id + 3 points (6 coords)
                continue
            
            # Section 1: Parse class ID and polygon coordinates
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            # Group into (x, y) pairs
            polygon = []
            for i in range(0, len(coords) - 1, 2):
                polygon.append([coords[i], coords[i + 1]])
            
            annotations.append({
                'class_id': class_id,
                'polygon': polygon  # List of [x, y] normalized coordinates
            })
    
    return annotations


def polygon_to_mask(
    polygon: List[List[float]], 
    img_width: int, 
    img_height: int
) -> np.ndarray:
    """
    Convert normalized polygon coordinates to binary mask.
    
    Section 1: No resizing performed to preserve geometric fidelity
    required for accurate area estimation.
    
    Args:
        polygon: List of [x, y] normalized coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Binary mask of shape (img_height, img_width)
    """
    # Convert normalized coords to pixel coords
    points = np.array([
        [int(x * img_width), int(y * img_height)]
        for x, y in polygon
    ], dtype=np.int32)
    
    # Create binary mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    
    return mask


def create_segmentation_mask(
    annotations: List[Dict],
    img_width: int,
    img_height: int,
    num_classes: int = 4
) -> np.ndarray:
    """
    Create multi-class segmentation mask from YOLO annotations.
    
    Section 1: Labels converted to dense pixel-wise masks
    Class mapping: 0=Background, 1=Dust, 2=RunDown, 3=Scratch
    
    Args:
        annotations: List of annotation dicts from parse_yolo_polygon_label
        img_width: Image width
        img_height: Image height
        num_classes: Number of classes including background
        
    Returns:
        Segmentation mask of shape (img_height, img_width)
        Values are class indices [0, num_classes-1]
    """
    # Initialize with background class (0)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for ann in annotations:
        # YOLO class IDs: 0=Dust, 1=RunDown, 2=Scratch
        # Internal class IDs: 0=Background, 1=Dust, 2=RunDown, 3=Scratch
        # So we add 1 to YOLO class ID
        class_id = ann['class_id'] + 1
        poly_mask = polygon_to_mask(ann['polygon'], img_width, img_height)
        
        # Overlay (later annotations overwrite earlier ones in overlapping regions)
        mask[poly_mask == 1] = class_id
    
    return mask


# =============================================================================
# Section 1: Preprocessing Pipeline
# Grayscale to 3-channel, ImageNet normalization
# Padding to ensure dimensions are multiples of patch size (14)
# =============================================================================

def pad_to_patch_size(
    image: np.ndarray,
    mask: np.ndarray = None,
    patch_size: int = 14
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int]]:
    """
    Pad image (and mask) so dimensions are divisible by patch_size.
    
    Section 1: Minimal padding to preserve geometric fidelity.
    DINOv2 requires dimensions divisible by 14.
    
    Args:
        image: Input image (H, W, C)
        mask: Optional segmentation mask (H, W)
        patch_size: Patch size (default 14 for DINOv2)
        
    Returns:
        padded_image: Image with dimensions divisible by patch_size
        padded_mask: Mask with same padding (or None)
        original_size: (H, W) for later cropping
    """
    h, w = image.shape[:2]
    original_size = (h, w)
    
    # Calculate padding needed
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    if pad_h == 0 and pad_w == 0:
        return image, mask, original_size
    
    # Pad image (bottom and right)
    padded_image = np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode='constant',
        constant_values=0
    )
    
    # Pad mask if provided
    padded_mask = None
    if mask is not None:
        padded_mask = np.pad(
            mask,
            ((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0  # Background class
        )
    
    return padded_image, padded_mask, original_size


def get_train_transforms(target_size: Tuple[int, int] = None) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Section 1: Standard normalization using ImageNet statistics.
    Images resized to consistent size for batching (divisible by 14).
    """
    from config import DataConfig
    
    # Use target_size from config if not specified
    if target_size is None:
        target_size = DataConfig.TRAIN_IMAGE_SIZE
    
    transforms_list = []
    
    # Resize to consistent size for batching (if specified)
    if target_size is not None:
        transforms_list.append(
            A.Resize(height=target_size[0], width=target_size[1])
        )
    
    # Geometric augmentations (preserve defect characteristics)
    transforms_list.extend([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        A.Normalize(
            mean=DINOv2Config.IMAGENET_MEAN,
            std=DINOv2Config.IMAGENET_STD,
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)


def get_val_transforms(target_size: Tuple[int, int] = None) -> A.Compose:
    """
    Get validation/inference transforms (no augmentation).
    
    Section 1: Resize (if specified) and normalization applied.
    """
    from config import DataConfig
    
    # Use target_size from config if not specified
    if target_size is None:
        target_size = DataConfig.TRAIN_IMAGE_SIZE
    
    transforms_list = []
    
    # Resize to consistent size for batching (if specified)
    if target_size is not None:
        transforms_list.append(
            A.Resize(height=target_size[0], width=target_size[1])
        )
    
    transforms_list.extend([
        A.Normalize(
            mean=DINOv2Config.IMAGENET_MEAN,
            std=DINOv2Config.IMAGENET_STD,
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)


def load_and_preprocess_image(
    image_path: Union[str, Path],
    to_rgb: bool = True
) -> np.ndarray:
    """
    Load image and convert grayscale to 3-channel RGB.
    
    Section 1: Grayscale images are replicated to three channels
    to match transformer input requirements.
    
    Args:
        image_path: Path to image file
        to_rgb: Whether to convert to RGB (default True)
        
    Returns:
        Image array of shape (H, W, 3)
    """
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Section 1: Convert to RGB (OpenCV loads as BGR)
    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # If grayscale, replicate to 3 channels
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    
    return image


# =============================================================================
# PyTorch Dataset
# =============================================================================

class DefectDataset(Dataset):
    """
    Dataset for SPGS-Net training and inference.
    
    Loads images and corresponding YOLO polygon labels,
    converts to segmentation masks.
    """
    
    def __init__(
        self,
        images_dir: Union[str, Path],
        labels_dir: Union[str, Path],
        transform: Optional[A.Compose] = None,
        return_path: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO label files
            transform: Albumentations transform pipeline
            return_path: Whether to return image path in __getitem__
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.return_path = return_path
        
        # Get list of image files
        self.image_files = []
        for ext in DataConfig.IMAGE_EXTENSIONS:
            self.image_files.extend(list(self.images_dir.glob(f"*{ext}")))
        
        self.image_files = sorted(self.image_files)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample by index.
        
        Returns:
            Dictionary with 'image', 'mask', and optionally 'path'
        """
        image_path = self.image_files[idx]
        
        # Get corresponding label file
        label_name = image_path.stem + ".txt"
        label_path = self.labels_dir / label_name
        
        # Section 1: Load and preprocess image
        image = load_and_preprocess_image(image_path)
        img_height, img_width = image.shape[:2]
        
        # Parse annotations and create mask
        annotations = parse_yolo_polygon_label(label_path)
        mask = create_segmentation_mask(
            annotations, img_width, img_height,
            num_classes=4
        )
        
        # Pad to ensure dimensions are divisible by 14 (DINOv2 patch size)
        image, mask, original_size = pad_to_patch_size(
            image, mask, patch_size=DINOv2Config.PATCH_SIZE
        )
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Convert to tensor if no transform
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        # Ensure mask is long tensor for cross-entropy
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        
        result = {
            'image': image,
            'mask': mask,
        }
        
        if self.return_path:
            result['path'] = str(image_path)
        
        return result


def get_dataloader(
    split: str = "train",
    batch_size: int = None,
    shuffle: bool = None,
    num_workers: int = 4
) -> DataLoader:
    """
    Get dataloader for specified split.
    
    Args:
        split: 'train', 'valid', or 'test'
        batch_size: Batch size (default from config)
        shuffle: Whether to shuffle (default True for train)
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader instance
    """
    if batch_size is None:
        batch_size = TrainingConfig.BATCH_SIZE
    
    if shuffle is None:
        shuffle = (split == "train")
    
    # Get paths based on split
    if split == "train":
        images_dir = PathConfig.TRAIN_IMAGES
        labels_dir = PathConfig.TRAIN_LABELS
        transform = get_train_transforms()
    elif split == "valid":
        images_dir = PathConfig.VALID_IMAGES
        labels_dir = PathConfig.VALID_LABELS
        transform = get_val_transforms()
    elif split == "test":
        images_dir = PathConfig.TEST_IMAGES
        labels_dir = PathConfig.TEST_LABELS
        transform = get_val_transforms()
    else:
        raise ValueError(f"Unknown split: {split}")
    
    dataset = DefectDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=transform,
        return_path=(split == "test")
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    )
    
    return dataloader


if __name__ == "__main__":
    # Test data utilities
    print("Testing data utilities...")
    
    # Test label parsing
    label_files = list(PathConfig.TRAIN_LABELS.glob("*.txt"))
    if label_files:
        sample_label = label_files[0]
        annotations = parse_yolo_polygon_label(sample_label)
        print(f"Parsed {len(annotations)} annotations from {sample_label.name}")
        if annotations:
            print(f"First annotation: class={annotations[0]['class_id']}, "
                  f"points={len(annotations[0]['polygon'])}")
    
    # Test dataset
    try:
        dataset = DefectDataset(
            PathConfig.TRAIN_IMAGES,
            PathConfig.TRAIN_LABELS,
            transform=get_val_transforms()
        )
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Unique mask values: {torch.unique(sample['mask']).tolist()}")
    except Exception as e:
        print(f"Dataset test error: {e}")
