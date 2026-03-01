import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
sys.path.append(str(Path(__file__).parent))

from config import (
    PathConfig, DINOv2Config, MLConfig, UNetConfig, 
    TrainingConfig, CalibrationConfig, OutputConfig,
    create_directories
)
from dino import DINOv2Extractor
from ml import PatchClassifier, create_anomaly_heatmap
from anomaly_upsampling import AnomalyUpsampler
from unet import AttentionUNet, PriorWeightedLoss
from defect_instance import InstanceSeparator
from area_est import AreaCalculator
from utils import (
    DefectDataset, get_dataloader, get_val_transforms,
    load_and_preprocess_image, create_segmentation_mask, parse_yolo_polygon_label,
    create_visualization, export_results_json, save_visualization
)

class SPGSNet:
    def __init__(
        self,
        device: str = None,
        load_checkpoint: Optional[str] = None
    ):
        self.device = device or TrainingConfig.DEVICE
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        create_directories()
        
        print("\n[Section 2] Initializing DINOv2 Feature Extractor...")
        self.dino_extractor = DINOv2Extractor().to(self.device)
        self.dino_extractor.eval()
        
        print("[Section 3] Initializing XGBoost Patch Classifier...")
        self.patch_classifier = PatchClassifier()
        
        print("[Section 4] Initializing Anomaly Upsampler...")
        self.upsampler = AnomalyUpsampler()
        
        print("[Section 5] Initializing Attention U-Net...")
        self.unet = AttentionUNet().to(self.device)
        
        print("[Section 6] Initializing Instance Separator...")
        self.instance_separator = InstanceSeparator()
        
        print("[Section 7] Initializing Area Calculator...")
        self.area_calculator = AreaCalculator()
        
        self.criterion = PriorWeightedLoss()
        
        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)
        
        print("\n[SPGS-Net] Pipeline initialization complete!")
    
    def train_ml_classifier(
        self,
        train_loader: DataLoader,
        verbose: bool = True
    ):
        print("\n[Section 3] Training XGBoost Patch Classifier...")
        
        all_features = []
        all_labels = []
        
        self.dino_extractor.eval()
        
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Extracting patch features"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].numpy()
                
                spatial_features, patch_grid = self.dino_extractor.get_spatial_features(images)
                
                features_np = spatial_features.cpu().numpy()
                
                for i in range(images.shape[0]):
                    mask = masks[i]
                    patch_h, patch_w = patch_grid
                    
                    from ml.xgboost_classifier import get_patch_labels_from_mask
                    patch_labels = get_patch_labels_from_mask(mask - 0, DINOv2Config.PATCH_SIZE)
                    
                    all_features.append(features_np[i])
                    all_labels.append(patch_labels)
        
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        
        metrics = self.patch_classifier.train(all_features, all_labels, verbose=verbose)
        
        self.patch_classifier.save()
        
        return metrics
    
    def train_unet(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = None,
        lr: float = None
    ):
        epochs = epochs or TrainingConfig.EPOCHS
        lr = lr or TrainingConfig.INITIAL_LR
        
        print(f"\n[Section 9] Training Attention U-Net for {epochs} epochs...")
        
        self.unet.train()
        self.dino_extractor.eval()
        
        optimizer = optim.AdamW(
            self.unet.parameters(),
            lr=lr,
            weight_decay=TrainingConfig.WEIGHT_DECAY
        )
        
        if TrainingConfig.USE_SCHEDULER:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.unet.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                with torch.no_grad():
                    spatial_features, patch_grid = self.dino_extractor.get_spatial_features(images)
                    
                    if self.patch_classifier.model is not None:
                        heatmap = create_anomaly_heatmap(
                            spatial_features, patch_grid, self.patch_classifier
                        )
                        heatmap_tensor = torch.from_numpy(heatmap).float().to(self.device)
                        
                        prior = self.upsampler(heatmap_tensor, images.shape[2:])
                    else:
                        prior = None
                optimizer.zero_grad()
                logits = self.unet(images, prior)
                
                loss_dict = self.criterion(logits, masks, prior)
                loss = loss_dict['loss']
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_train_loss = np.mean(train_losses)
            
            val_loss = self._validate(val_loader)
            
            if TrainingConfig.USE_SCHEDULER:
                scheduler.step()
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(PathConfig.BEST_MODEL)
                print(f"  -> New best model saved!")
            else:
                patience_counter += 1
            
            if TrainingConfig.EARLY_STOPPING and patience_counter >= TrainingConfig.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\n[Section 9] Training complete. Best validation loss: {best_val_loss:.4f}")
    
    def _validate(self, val_loader: DataLoader) -> float:
        self.unet.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                logits = self.unet(images)
                loss_dict = self.criterion(logits, masks)
                val_losses.append(loss_dict['loss'].item())
        
        return np.mean(val_losses)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, np.ndarray, torch.Tensor],
        return_all: bool = False
    ) -> Dict:
        self.unet.eval()
        self.dino_extractor.eval()
        
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image_np = load_and_preprocess_image(image_path)
            original_size = image_np.shape[:2]
        elif isinstance(image, np.ndarray):
            image_np = image
            original_size = image_np.shape[:2]
            image_path = None
        else:
            raise ValueError("Image must be path, numpy array, or tensor")
        
        transform = get_val_transforms()
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        spatial_features, patch_grid = self.dino_extractor.get_spatial_features(image_tensor)
        
        if self.patch_classifier.model is not None:
            heatmap = create_anomaly_heatmap(
                spatial_features, patch_grid, self.patch_classifier
            )
            heatmap_tensor = torch.from_numpy(heatmap).float().to(self.device)
        else:
            heatmap = None
            heatmap_tensor = None
        
        if heatmap_tensor is not None:
            prior = self.upsampler(heatmap_tensor, image_tensor.shape[2:])
        else:
            prior = None
        
        logits = self.unet(image_tensor, prior)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        
        pred_mask = predictions[0].cpu().numpy()
        prob_np = probabilities[0].cpu().numpy()
        
        instances = self.instance_separator.process(pred_mask, prob_np)
        
        instances = self.area_calculator.process_instances(instances)
        
        result = {
            'segmentation_mask': pred_mask,
            'probabilities': prob_np,
            'instances': instances,
            'original_size': original_size,
        }
        
        if return_all:
            result['patch_features'] = spatial_features.cpu().numpy()
            result['anomaly_heatmap'] = heatmap
            result['prior'] = prior.cpu().numpy() if prior is not None else None
        
        return result
    
    def process_and_visualize(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_json: bool = True
    ) -> Dict:
        output_dir = Path(output_dir or PathConfig.OUTPUT_DIR)
        result = self.predict(image_path, return_all=True)
        image_bgr = cv2.imread(str(image_path))
        visualization = create_visualization(
            image_bgr,
            result['segmentation_mask'],
            result['instances']
        )
        
        image_name = Path(image_path).stem
        vis_path = output_dir / "visualizations" / f"{image_name}_result.jpg"
        save_visualization(visualization, vis_path)
        result['visualization_path'] = str(vis_path)
        
        if save_json:
            json_path = output_dir / "json_results" / f"{image_name}_result.json"
            export_results_json(image_path, result['instances'], json_path)
            result['json_path'] = str(json_path)
        
        return result
    
    def save_checkpoint(self, path: Union[str, Path]):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'unet_state_dict': self.unet.state_dict(),
            'config': {
                'device': self.device,
            }
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load model checkpoint."""
        path = Path(path)
        if not path.exists():
            print(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        print(f"Checkpoint loaded from {path}")


def main():
    """Main entry point for training and inference."""
    parser = argparse.ArgumentParser(description="SPGS-Net Multi-Defect Detection")
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'train_ml'],
                       required=True, help="Run mode")
    parser.add_argument('--image', type=str, help="Image path for inference")
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--checkpoint', type=str, help="Checkpoint to load")
    parser.add_argument('--output_dir', type=str, help="Output directory")
    
    args = parser.parse_args()
    spgs_net = SPGSNet(load_checkpoint=args.checkpoint)
    
    if args.mode == 'train_ml':
        print("Training XGBoost Patch Classifier")    
        train_loader = get_dataloader('train', batch_size=args.batch_size)
        spgs_net.train_ml_classifier(train_loader)
        
    elif args.mode == 'train':
        print("SPGS-Net Full Training Pipeline")
        
        train_loader = get_dataloader('train', batch_size=args.batch_size)
        val_loader = get_dataloader('valid', batch_size=args.batch_size)
        
        if not MLConfig.MODEL_PATH.exists():
            print("\nStep 1: Training XGBoost classifier...")
            spgs_net.train_ml_classifier(train_loader)
        else:
            print("\nStep 1: Loading existing XGBoost classifier...")
            spgs_net.patch_classifier.load()
        
        print("\nStep 2: Training Attention U-Net...")
        spgs_net.train_unet(train_loader, val_loader, epochs=args.epochs, lr=args.lr)
        
    elif args.mode == 'inference':
        if not args.image:
            print("Error: --image required for inference mode")
            return
        
        print("\nSPGS-Net Inference")
        
        if MLConfig.MODEL_PATH.exists():
            spgs_net.patch_classifier.load()
        
        output_dir = args.output_dir or PathConfig.OUTPUT_DIR
        result = spgs_net.process_and_visualize(args.image, output_dir)
        
        print(f"\n[Section 8] Results:")
        print(f"  Detected {len(result['instances'])} defects:")
        for inst in result['instances']:
            print(f"    - {inst['class_name']}: {inst['area_mm2']:.2f} mm²")
        
        if 'visualization_path' in result:
            print(f"\n  Visualization: {result['visualization_path']}")
        if 'json_path' in result:
            print(f"  JSON results: {result['json_path']}")


if __name__ == "__main__":
    main()
