"""
Data Utilities for Multi-Class Anomaly Detection
Handles:
- YOLOv8 polygon to binary mask conversion
- Dataset splitting (train/val/test)
- Image loading and preprocessing
"""
import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, MASKS_DIR, SPLITS_DIR,
    CLASS_NAMES, CLASS_TO_ID, ID_TO_CLASS, NUM_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)


def parse_yolov8_segmentation_label(label_path: Path, img_width: int, img_height: int) -> List[Dict]:
    """
    Parse YOLOv8 segmentation label file (polygon format).
    
    Format: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates)
    
    Returns:
        List of dicts with 'class_id' and 'polygon' (absolute coordinates)
    """
    annotations = []
    
    if not label_path.exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # Need at least class_id + 3 points (6 coords)
                continue
            
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Convert normalized coordinates to absolute
            polygon = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = int(coords[i] * img_width)
                    y = int(coords[i + 1] * img_height)
                    polygon.append([x, y])
            
            if len(polygon) >= 3:  # Valid polygon needs at least 3 points
                annotations.append({
                    'class_id': class_id,
                    'polygon': np.array(polygon, dtype=np.int32)
                })
    
    return annotations


def create_mask_from_polygons(annotations: List[Dict], img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create binary mask and class mask from polygon annotations.
    
    Returns:
        binary_mask: Single channel mask (0 or 255)
        class_mask: Single channel mask with class IDs (0, 1, 2, or 255 for background)
    """
    binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    class_mask = np.full((img_height, img_width), 255, dtype=np.uint8)  # 255 = background
    
    for ann in annotations:
        polygon = ann['polygon']
        class_id = ann['class_id']
        
        # Fill polygon in binary mask
        cv2.fillPoly(binary_mask, [polygon], 255)
        
        # Fill polygon in class mask with class ID
        cv2.fillPoly(class_mask, [polygon], class_id)
    
    return binary_mask, class_mask


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Get image width and height without loading full image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img.shape[1], img.shape[0]  # width, height


def convert_all_labels_to_masks(verbose: bool = True) -> Dict[str, Path]:
    """
    Convert all YOLOv8 polygon labels to binary masks.
    
    Returns:
        Dictionary mapping image names to mask paths
    """
    if verbose:
        print(f"\n{'='*60}")
        print("CONVERTING YOLOV8 LABELS TO MASKS")
        print(f"{'='*60}")
    
    mask_mapping = {}
    image_files = list(TRAIN_IMAGES_PATH.glob("*.jpg")) + list(TRAIN_IMAGES_PATH.glob("*.png"))
    
    if verbose:
        print(f"Found {len(image_files)} images to process")
    
    # Create class-specific mask directories
    for class_name in CLASS_NAMES:
        (MASKS_DIR / class_name.lower()).mkdir(exist_ok=True)
    
    class_counts = {name: 0 for name in CLASS_NAMES}
    
    for idx, img_path in enumerate(image_files):
        # Get corresponding label file
        label_name = img_path.stem + ".txt"
        label_path = TRAIN_LABELS_PATH / label_name
        
        # Get image dimensions
        img_width, img_height = get_image_dimensions(img_path)
        
        # Parse annotations
        annotations = parse_yolov8_segmentation_label(label_path, img_width, img_height)
        
        # Create masks
        binary_mask, class_mask = create_mask_from_polygons(annotations, img_width, img_height)
        
        # Save masks
        binary_mask_path = MASKS_DIR / f"{img_path.stem}_binary.png"
        class_mask_path = MASKS_DIR / f"{img_path.stem}_class.png"
        
        cv2.imwrite(str(binary_mask_path), binary_mask)
        cv2.imwrite(str(class_mask_path), class_mask)
        
        # Count classes in this image
        for ann in annotations:
            class_counts[CLASS_NAMES[ann['class_id']]] += 1
        
        mask_mapping[img_path.stem] = {
            'image_path': str(img_path),
            'binary_mask': str(binary_mask_path),
            'class_mask': str(class_mask_path),
            'classes': [ann['class_id'] for ann in annotations]
        }
        
        if verbose and (idx + 1) % 200 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images...")
    
    # Save mapping
    mapping_path = MASKS_DIR / "mask_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(mask_mapping, f, indent=2)
    
    if verbose:
        print(f"\n✓ Converted {len(image_files)} labels to masks")
        print(f"✓ Mask mapping saved to: {mapping_path}")
        print(f"\nClass distribution in annotations:")
        for name, count in class_counts.items():
            print(f"  - {name}: {count} instances")
    
    return mask_mapping


def get_image_primary_class(label_path: Path) -> Optional[int]:
    """
    Get the primary (dominant) class in an image based on annotation count.
    Returns the class ID with most annotations, or None if no annotations.
    """
    if not label_path.exists():
        return None
    
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                class_id = int(parts[0])
                if class_id in class_counts:
                    class_counts[class_id] += 1
    
    total = sum(class_counts.values())
    if total == 0:
        return None
    
    return max(class_counts, key=class_counts.get)


def create_train_val_test_split(verbose: bool = True) -> Dict[str, List[str]]:
    """
    Split dataset into train/val/test sets with stratification by primary class.
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing image stems
    """
    if verbose:
        print(f"\n{'='*60}")
        print("CREATING TRAIN/VAL/TEST SPLIT")
        print(f"{'='*60}")
    
    image_files = list(TRAIN_IMAGES_PATH.glob("*.jpg")) + list(TRAIN_IMAGES_PATH.glob("*.png"))
    
    # Get primary class for each image (for stratification)
    image_stems = []
    image_classes = []
    
    for img_path in image_files:
        label_name = img_path.stem + ".txt"
        label_path = TRAIN_LABELS_PATH / label_name
        primary_class = get_image_primary_class(label_path)
        
        if primary_class is not None:
            image_stems.append(img_path.stem)
            image_classes.append(primary_class)
    
    if verbose:
        print(f"Total images with valid annotations: {len(image_stems)}")
        for class_id, class_name in ID_TO_CLASS.items():
            count = image_classes.count(class_id)
            print(f"  - {class_name}: {count} images ({100*count/len(image_stems):.1f}%)")
    
    # First split: train vs (val+test)
    train_stems, temp_stems, train_classes, temp_classes = train_test_split(
        image_stems, image_classes,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=image_classes,
        random_state=RANDOM_SEED
    )
    
    # Second split: val vs test
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_stems, test_stems, _, _ = train_test_split(
        temp_stems, temp_classes,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_classes,
        random_state=RANDOM_SEED
    )
    
    splits = {
        'train': train_stems,
        'val': val_stems,
        'test': test_stems
    }
    
    # Save splits
    splits_path = SPLITS_DIR / "splits.json"
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    if verbose:
        print(f"\nSplit sizes:")
        print(f"  - Train: {len(train_stems)} images ({100*len(train_stems)/len(image_stems):.1f}%)")
        print(f"  - Val:   {len(val_stems)} images ({100*len(val_stems)/len(image_stems):.1f}%)")
        print(f"  - Test:  {len(test_stems)} images ({100*len(test_stems)/len(image_stems):.1f}%)")
        print(f"\n✓ Splits saved to: {splits_path}")
    
    return splits


def load_splits() -> Dict[str, List[str]]:
    """Load previously created splits."""
    splits_path = SPLITS_DIR / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}. Run create_train_val_test_split() first.")
    
    with open(splits_path, 'r') as f:
        return json.load(f)


def load_mask_mapping() -> Dict[str, Dict]:
    """Load previously created mask mapping."""
    mapping_path = MASKS_DIR / "mask_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mask mapping not found: {mapping_path}. Run convert_all_labels_to_masks() first.")
    
    with open(mapping_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Run data preparation
    print("Starting data preparation...")
    
    # Step 1: Convert labels to masks
    mask_mapping = convert_all_labels_to_masks()
    
    # Step 2: Create train/val/test split
    splits = create_train_val_test_split()
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
