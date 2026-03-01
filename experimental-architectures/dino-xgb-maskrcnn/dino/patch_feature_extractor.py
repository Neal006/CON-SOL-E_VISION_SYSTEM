import os
import sys
import numpy as np
import torch
import cv2
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import Tuple, List, Dict
from tqdm import tqdm
try:
    _current_dir = Path(__file__).parent.parent
except NameError:
    _current_dir = Path(os.getcwd())
sys.path.insert(0, str(_current_dir))
from config import (
    TRAIN_IMAGES_PATH, FEATURES_DIR, MASKS_DIR,
    DINO_MODEL_NAME, DINO_EMBED_DIM, DINO_INPUT_SIZE, DINO_PATCH_SIZE,
    PATCH_CLASS_NAMES, PATCH_ID_TO_CLASS, PATCH_NUM_CLASSES, NORMAL_CLASS_ID,
    DEVICE
)
from dino.data_utils import load_splits, load_mask_mapping


class PatchFeatureExtractor:    
    def __init__(self, model_name: str = DINO_MODEL_NAME, device: str = DEVICE):
        self.device = device
        self.model_name = model_name
        print(f"LOADING DINOv2 MODEL: {model_name}")
        print(f"Device: {device}")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device)
        self.model.eval()
        self.embed_dim = DINO_EMBED_DIM
        self.grid_size = DINO_INPUT_SIZE // DINO_PATCH_SIZE
        self.num_patches = self.grid_size * self.grid_size
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Grid size: {self.grid_size}x{self.grid_size} = {self.num_patches} patches")
        self.transform = transforms.Compose([
            transforms.Resize((DINO_INPUT_SIZE, DINO_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract_patch_tokens(self, image: Image.Image) -> np.ndarray:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        features_dict = self.model.forward_features(img_tensor)
        patch_tokens = features_dict['x_norm_patchtokens']
        return patch_tokens.squeeze(0).cpu().numpy()
    
    def get_patch_labels_from_mask(self, class_mask_path: str) -> np.ndarray:
        mask = cv2.imread(class_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {class_mask_path}")
        mask_resized = cv2.resize(mask, (self.grid_size, self.grid_size), interpolation=cv2.INTER_NEAREST)
        patch_labels = np.where(mask_resized == 255, NORMAL_CLASS_ID, mask_resized)
        return patch_labels.flatten()
    
    def extract_features_for_split(self, image_stems: List[str], mask_mapping: Dict, split_name: str = "train") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        print(f"Extracting patch features for {split_name} set ({len(image_stems)} images)...")
        all_features = []
        all_labels = []
        all_refs = []
        for stem in tqdm(image_stems, desc=f"Extracting {split_name}"):
            if stem not in mask_mapping:
                continue
            mapping = mask_mapping[stem]
            image_path = mapping['image_path']
            class_mask_path = mapping['class_mask']
            try:
                image = Image.open(image_path)
                patch_features = self.extract_patch_tokens(image)
                patch_labels = self.get_patch_labels_from_mask(class_mask_path)
                all_features.append(patch_features)
                all_labels.append(patch_labels)
                for i in range(self.num_patches):
                    all_refs.append(f"{stem}:{i}")
            except Exception as e:
                print(f"Error processing {stem}: {e}")
                continue
        if len(all_features) == 0:
            raise ValueError(f"No features extracted for {split_name}")
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        print(f"Extracted {features.shape[0]:,} patches with {features.shape[1]} features")
        return features, labels, all_refs


def save_patch_features_to_parquet(features: np.ndarray, labels: np.ndarray, image_refs: List[str], output_path: Path):
    feature_cols = [f'feat_{i}' for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_cols)
    df['label'] = labels
    df['image_ref'] = image_refs
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df):,} patches to {output_path}")


def load_patch_features_from_parquet(parquet_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    features = df[feature_cols].values
    labels = df['label'].values
    image_refs = df['image_ref'].tolist() if 'image_ref' in df.columns else []
    return features, labels, image_refs


def extract_and_save_all_patch_features():
    print("PATCH-LEVEL FEATURE EXTRACTION")
    splits = load_splits()
    mask_mapping = load_mask_mapping()
    extractor = PatchFeatureExtractor()
    for split_name, image_stems in splits.items():
        print(f"Processing {split_name} set...")
        features, labels, image_refs = extractor.extract_features_for_split(image_stems, mask_mapping, split_name)
        output_path = FEATURES_DIR / f"patch_{split_name}_features.parquet"
        save_patch_features_to_parquet(features, labels, image_refs, output_path)
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class distribution in {split_name}:")
        for class_id, count in zip(unique, counts):
            class_name = PATCH_ID_TO_CLASS.get(int(class_id), f"Unknown_{class_id}")
            pct = 100 * count / len(labels)
            print(f"  - {class_name}: {count:,} patches ({pct:.1f}%)")
    print("PATCH FEATURE EXTRACTION COMPLETE")
    print(f"Features saved to: {FEATURES_DIR}")

if __name__ == "__main__":
    extract_and_save_all_patch_features()