import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import shutil

try:
    _current_dir = Path(__file__).parent.parent
except NameError:
    _current_dir = Path(os.getcwd())
sys.path.insert(0, str(_current_dir))
from config import (
    TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, MASKS_DIR, SPLITS_DIR,
    CLASS_NAMES, CLASS_TO_ID, ID_TO_CLASS, NUM_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)


def parse_yolov8_segmentation_label(label_path: Path, img_width: int, img_height: int) -> List[Dict]:
    annotations = []
    if not label_path.exists():
        return annotations
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            polygon = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = int(coords[i] * img_width)
                    y = int(coords[i + 1] * img_height)
                    polygon.append([x, y])
            if len(polygon) >= 3:
                annotations.append({'class_id': class_id, 'polygon': np.array(polygon, dtype=np.int32)})
    return annotations


def create_mask_from_polygons(annotations: List[Dict], img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
    binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    class_mask = np.full((img_height, img_width), 255, dtype=np.uint8)
    for ann in annotations:
        polygon = ann['polygon']
        class_id = ann['class_id']
        cv2.fillPoly(binary_mask, [polygon], 255)
        cv2.fillPoly(class_mask, [polygon], class_id)
    return binary_mask, class_mask


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img.shape[1], img.shape[0]


def convert_all_labels_to_masks(verbose: bool = True) -> Dict[str, Path]:
    if verbose:
        print("CONVERTING YOLOV8 LABELS TO MASKS")
    mask_mapping = {}
    image_files = list(TRAIN_IMAGES_PATH.glob("*.jpg")) + list(TRAIN_IMAGES_PATH.glob("*.png"))
    if verbose:
        print(f"Found {len(image_files)} images to process")
    for class_name in CLASS_NAMES:
        (MASKS_DIR / class_name.lower()).mkdir(exist_ok=True)
    class_counts = {name: 0 for name in CLASS_NAMES}
    for idx, img_path in enumerate(image_files):
        label_name = img_path.stem + ".txt"
        label_path = TRAIN_LABELS_PATH / label_name
        img_width, img_height = get_image_dimensions(img_path)
        annotations = parse_yolov8_segmentation_label(label_path, img_width, img_height)
        binary_mask, class_mask = create_mask_from_polygons(annotations, img_width, img_height)
        binary_mask_path = MASKS_DIR / f"{img_path.stem}_binary.png"
        class_mask_path = MASKS_DIR / f"{img_path.stem}_class.png"
        cv2.imwrite(str(binary_mask_path), binary_mask)
        cv2.imwrite(str(class_mask_path), class_mask)
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
    mapping_path = MASKS_DIR / "mask_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(mask_mapping, f, indent=2)
    if verbose:
        print(f"Converted {len(image_files)} labels to masks")
        print(f"Mask mapping saved to: {mapping_path}")
        for name, count in class_counts.items():
            print(f"  - {name}: {count} instances")
    return mask_mapping


def get_image_primary_class(label_path: Path) -> Optional[int]:
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
    if verbose:
        print("CREATING TRAIN/VAL/TEST SPLIT")
    image_files = list(TRAIN_IMAGES_PATH.glob("*.jpg")) + list(TRAIN_IMAGES_PATH.glob("*.png"))
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
            print(f"  - {class_name}: {count} images")
    train_stems, temp_stems, train_classes, temp_classes = train_test_split(
        image_stems, image_classes, test_size=(VAL_RATIO + TEST_RATIO), stratify=image_classes, random_state=RANDOM_SEED
    )
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_stems, test_stems, _, _ = train_test_split(
        temp_stems, temp_classes, test_size=(1 - val_ratio_adjusted), stratify=temp_classes, random_state=RANDOM_SEED
    )
    splits = {'train': train_stems, 'val': val_stems, 'test': test_stems}
    splits_path = SPLITS_DIR / "splits.json"
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    if verbose:
        print(f"Train: {len(train_stems)}, Val: {len(val_stems)}, Test: {len(test_stems)}")
        print(f"Splits saved to: {splits_path}")
    return splits


def load_splits() -> Dict[str, List[str]]:
    splits_path = SPLITS_DIR / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    with open(splits_path, 'r') as f:
        return json.load(f)


def load_mask_mapping() -> Dict[str, Dict]:
    mapping_path = MASKS_DIR / "mask_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mask mapping not found: {mapping_path}")
    with open(mapping_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    print("Starting data preparation...")
    mask_mapping = convert_all_labels_to_masks()
    splits = create_train_val_test_split()
    print("DATA PREPARATION COMPLETE")