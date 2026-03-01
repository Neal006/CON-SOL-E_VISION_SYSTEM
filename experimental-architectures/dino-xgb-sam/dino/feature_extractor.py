"""
DINOv2 Feature Extractor for Multi-Class Anomaly Detection
Uses dinov2_vits14 (384-dim embeddings) for feature extraction.
Features are saved to Apache Parquet format for optimal inference latency.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, FEATURES_DIR,
    DINO_MODEL_NAME, DINO_EMBED_DIM, DINO_INPUT_SIZE,
    CLASS_NAMES, ID_TO_CLASS, NUM_CLASSES, DEVICE
)
from dino.data_utils import load_splits, get_image_primary_class


class ImageDataset(Dataset):
    """Dataset for loading images with their labels."""
    
    def __init__(self, image_stems: List[str], transform=None):
        self.image_stems = image_stems
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for stem in image_stems:
            # Find image file (jpg or png)
            img_path = TRAIN_IMAGES_PATH / f"{stem}.jpg"
            if not img_path.exists():
                img_path = TRAIN_IMAGES_PATH / f"{stem}.png"
            
            if img_path.exists():
                # Get label from annotation file
                label_path = TRAIN_LABELS_PATH / f"{stem}.txt"
                label = get_image_primary_class(label_path)
                
                if label is not None:
                    self.image_paths.append(img_path)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(img_path)


class DINOv2FeatureExtractor:
    """Extract features using DINOv2 ViT-S/14 (384-dim)."""
    
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
        
        # Get embedding dimension
        self.embed_dim = self.model.embed_dim
        print(f"✓ Model loaded successfully")
        print(f"✓ Embedding dimension: {self.embed_dim}")
        
        # Define preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((DINO_INPUT_SIZE, DINO_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract CLS token features from images."""
        images = images.to(self.device)
        features = self.model(images)  # Returns CLS token
        return features.cpu().numpy()
    
    def extract_dataset_features(
        self, 
        image_stems: List[str],
        batch_size: int = 16,
        split_name: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract features for entire dataset.
        
        Returns:
            features: (N, embed_dim) array
            labels: (N,) array
            image_paths: List of image paths
        """
        print(f"\nExtracting features for {split_name} set ({len(image_stems)} images)...")
        
        dataset = ImageDataset(image_stems, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Windows compatibility
            pin_memory=True if self.device == 'cuda' else False
        )
        
        all_features = []
        all_labels = []
        all_paths = []
        
        for images, labels, paths in tqdm(dataloader, desc=f"Extracting {split_name}"):
            features = self.extract_features(images)
            all_features.append(features)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
        
        features = np.vstack(all_features)
        labels = np.array(all_labels)
        
        print(f"✓ Extracted {features.shape[0]} feature vectors of dimension {features.shape[1]}")
        
        return features, labels, all_paths


def save_features_to_parquet(
    features: np.ndarray,
    labels: np.ndarray,
    image_paths: List[str],
    output_path: Path
):
    """
    Save features to Apache Parquet format.
    
    Why Parquet?
    - Columnar storage: efficient for ML (column-wise access)
    - Compression: 50-80% smaller than CSV
    - Fast I/O with PyArrow: ~10x faster than CSV
    - Cross-language: Python, Java, C++, etc.
    - Schema preservation: maintains data types
    """
    # Create dataframe with features
    feature_cols = [f'feat_{i}' for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_cols)
    df['label'] = labels
    df['image_path'] = image_paths
    
    # Convert to PyArrow Table for optimal compression
    table = pa.Table.from_pandas(df)
    
    # Write with compression
    pq.write_table(
        table,
        output_path,
        compression='snappy',  # Fast compression/decompression
        use_dictionary=True
    )
    
    # Calculate file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved to {output_path} ({file_size_mb:.2f} MB)")


def load_features_from_parquet(parquet_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load features from Parquet file."""
    df = pd.read_parquet(parquet_path)
    
    # Extract features (all columns except 'label' and 'image_path')
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    features = df[feature_cols].values
    labels = df['label'].values
    image_paths = df['image_path'].tolist()
    
    return features, labels, image_paths


def extract_and_save_all_features(batch_size: int = 16):
    """Extract and save features for all splits."""
    
    print(f"\n{'='*60}")
    print("DINOV2 FEATURE EXTRACTION")
    print(f"{'='*60}")
    
    # Load splits
    splits = load_splits()
    
    # Initialize feature extractor
    extractor = DINOv2FeatureExtractor()
    
    # Process each split
    for split_name, image_stems in splits.items():
        print(f"\n{'-'*40}")
        print(f"Processing {split_name} set...")
        print(f"{'-'*40}")
        
        # Extract features
        features, labels, image_paths = extractor.extract_dataset_features(
            image_stems,
            batch_size=batch_size,
            split_name=split_name
        )
        
        # Save to Parquet
        output_path = FEATURES_DIR / f"{split_name}_features.parquet"
        save_features_to_parquet(features, labels, image_paths, output_path)
        
        # Print class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nClass distribution in {split_name}:")
        for class_id, count in zip(unique, counts):
            class_name = ID_TO_CLASS[class_id]
            print(f"  - {class_name}: {count} ({100*count/len(labels):.1f}%)")
    
    print(f"\n{'='*60}")
    print("FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nFeatures saved to: {FEATURES_DIR}")


if __name__ == "__main__":
    extract_and_save_all_features(batch_size=16)
