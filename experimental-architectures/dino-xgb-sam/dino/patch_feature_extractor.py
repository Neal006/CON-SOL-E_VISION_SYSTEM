"""
Patch-Level Feature Extractor for DINOv2
Extracts 37x37 patch tokens and labels each patch based on mask overlap.
"""
import os
import sys
import json
import numpy as np
import torch
import cv2
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import Tuple, List, Dict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_IMAGES_PATH, FEATURES_DIR, MASKS_DIR,
    DINO_MODEL_NAME, DINO_EMBED_DIM, DINO_INPUT_SIZE, DINO_PATCH_SIZE,
    PATCH_CLASS_NAMES, PATCH_ID_TO_CLASS, PATCH_NUM_CLASSES, NORMAL_CLASS_ID,
    DEVICE
)
from dino.data_utils import load_splits, load_mask_mapping


class PatchFeatureExtractor:
    """
    Extract patch-level features from DINOv2.
    Each image produces 37x37 = 1369 patch features.
    """
    
    def __init__(self, model_name: str = DINO_MODEL_NAME, device: str = DEVICE):
        self.device = device
        self.model_name = model_name
        
        print(f"\n{'='*60}")
        print(f"LOADING DINOv2 MODEL: {model_name}")
        print(f"Device: {device}")
        print(f"{'='*60}")
        
        # Load pretrained DINOv2 model
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        self.embed_dim = self.model.embed_dim
        self.grid_size = DINO_INPUT_SIZE // DINO_PATCH_SIZE  # 37
        self.num_patches = self.grid_size * self.grid_size   # 1369
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Embedding dimension: {self.embed_dim}")
        print(f"✓ Grid size: {self.grid_size}x{self.grid_size} = {self.num_patches} patches")
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((DINO_INPUT_SIZE, DINO_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def extract_patch_tokens(self, image: Image.Image) -> np.ndarray:
        """
        Extract all patch tokens from an image.
        
        Returns:
            np.ndarray: Shape (num_patches, embed_dim) = (1369, 384)
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get all features including patch tokens
        features_dict = self.model.forward_features(img_tensor)
        
        # Extract patch tokens (excluding CLS token)
        patch_tokens = features_dict['x_norm_patchtokens']
        
        return patch_tokens.squeeze(0).cpu().numpy()
    
    def get_patch_labels_from_mask(self, class_mask_path: str) -> np.ndarray:
        """
        Convert class mask to patch-level labels.
        
        The class mask has values:
        - 0, 1, 2: Defect class IDs (Dust, RunDown, Scratch)
        - 255: Background
        
        Returns:
            np.ndarray: Shape (num_patches,) with class IDs
        """
        # Load class mask
        mask = cv2.imread(class_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {class_mask_path}")
        
        # Resize to grid size (37x37)
        mask_resized = cv2.resize(mask, (self.grid_size, self.grid_size), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Convert 255 (background) to NORMAL_CLASS_ID (3)
        patch_labels = np.where(mask_resized == 255, NORMAL_CLASS_ID, mask_resized)
        
        return patch_labels.flatten()
    
    def extract_features_for_split(
        self,
        image_stems: List[str],
        mask_mapping: Dict,
        split_name: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract patch features for all images in a split.
        
        Returns:
            features: (N * num_patches, embed_dim)
            labels: (N * num_patches,)
            image_refs: List of "image_stem:patch_idx" for each feature
        """
        print(f"\nExtracting patch features for {split_name} set ({len(image_stems)} images)...")
        
        all_features = []
        all_labels = []
        all_refs = []
        
        for stem in tqdm(image_stems, desc=f"Extracting {split_name}"):
            if stem not in mask_mapping:
                continue
            
            mapping = mask_mapping[stem]
            image_path = mapping['image_path']
            class_mask_path = mapping['class_mask']
            
            # Load and process image
            try:
                image = Image.open(image_path)
                patch_features = self.extract_patch_tokens(image)
                patch_labels = self.get_patch_labels_from_mask(class_mask_path)
                
                all_features.append(patch_features)
                all_labels.append(patch_labels)
                
                # Create references for tracing back
                for i in range(self.num_patches):
                    all_refs.append(f"{stem}:{i}")
                    
            except Exception as e:
                print(f"Error processing {stem}: {e}")
                continue
        
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        
        print(f"✓ Extracted {features.shape[0]} patch features of dimension {features.shape[1]}")
        
        return features, labels, all_refs


def save_patch_features_to_parquet(
    features: np.ndarray,
    labels: np.ndarray,
    image_refs: List[str],
    output_path: Path
):
    """Save patch features to Parquet format."""
    feature_cols = [f'feat_{i}' for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_cols)
    df['label'] = labels
    df['image_ref'] = image_refs
    
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression='snappy', use_dictionary=True)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved to {output_path} ({file_size_mb:.2f} MB)")


def load_patch_features_from_parquet(parquet_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load patch features from Parquet file."""
    df = pd.read_parquet(parquet_path)
    
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    features = df[feature_cols].values
    labels = df['label'].values
    image_refs = df['image_ref'].tolist()
    
    return features, labels, image_refs


def extract_and_save_all_patch_features():
    """Extract and save patch features for all splits."""
    
    print(f"\n{'='*60}")
    print("PATCH-LEVEL FEATURE EXTRACTION")
    print(f"{'='*60}")
    
    # Load splits and mask mapping
    splits = load_splits()
    mask_mapping = load_mask_mapping()
    
    # Initialize extractor
    extractor = PatchFeatureExtractor()
    
    # Process each split
    for split_name, image_stems in splits.items():
        print(f"\n{'-'*40}")
        print(f"Processing {split_name} set...")
        print(f"{'-'*40}")
        
        features, labels, image_refs = extractor.extract_features_for_split(
            image_stems, mask_mapping, split_name
        )
        
        # Save to Parquet
        output_path = FEATURES_DIR / f"patch_{split_name}_features.parquet"
        save_patch_features_to_parquet(features, labels, image_refs, output_path)
        
        # Print class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nClass distribution in {split_name}:")
        for class_id, count in zip(unique, counts):
            class_name = PATCH_ID_TO_CLASS[int(class_id)]
            pct = 100 * count / len(labels)
            print(f"  - {class_name}: {count:,} patches ({pct:.1f}%)")
    
    print(f"\n{'='*60}")
    print("PATCH FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nFeatures saved to: {FEATURES_DIR}")


if __name__ == "__main__":
    extract_and_save_all_patch_features()
