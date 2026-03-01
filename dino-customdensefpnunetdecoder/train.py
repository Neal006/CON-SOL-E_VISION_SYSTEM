"""Training script for DINOv2 segmentation model."""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from data.dataset import DefectDataset, get_dataloaders
from data.augmentation import get_train_transform, get_val_transform
from models.model import SegmentationModel, build_model
from loss.losses import CombinedLoss, build_loss
from utils.metrics import SegmentationMetrics
from utils.visualization import plot_training_curves, visualize_predictions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: dict
) -> Tuple[float, float]:
    """Train for one epoch.
    
    Returns:
        Tuple of (average loss, mean IoU).
    """
    model.train()
    total_loss = 0.0
    metrics = SegmentationMetrics(
        num_classes=config['model']['num_classes'],
        class_names=['background', 'defect']
    )
    
    accumulation_steps = config['training'].get('accumulation_steps', 1)
    grad_clip = config['training'].get('gradient_clip', 1.0)
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        with autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, masks) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # Update metrics
        pred = outputs.argmax(dim=1)
        metrics.update(pred, masks)
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'iou': f'{metrics.compute()["mean_iou"]:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    results = metrics.compute()
    
    return avg_loss, results['mean_iou']


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict
) -> Tuple[float, float]:
    """Validate the model.
    
    Returns:
        Tuple of (average loss, mean IoU).
    """
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(
        num_classes=config['model']['num_classes'],
        class_names=['background', 'defect']
    )
    
    pbar = tqdm(dataloader, desc='Validation', leave=False)
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        with autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        total_loss += loss.item()
        
        pred = outputs.argmax(dim=1)
        metrics.update(pred, masks)
    
    avg_loss = total_loss / len(dataloader)
    results = metrics.compute()
    
    return avg_loss, results['mean_iou']


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


def main():
    parser = argparse.ArgumentParser(description='Train DINOv2 segmentation model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Setup directories
    save_dir = Path(config['paths']['save_dir'])
    log_dir = Path(config['paths']['log_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir)
    
    # Create transforms
    train_transform = get_train_transform(config)
    val_transform = get_val_transform(config)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        config, train_transform, val_transform
    )
    logger.info(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    
    # Create model
    model = build_model(config).to(device)
    model.print_summary()
    
    # Create loss and optimizer
    criterion = build_loss(config)
    
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training']['num_epochs'] // 3,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Resume from checkpoint
    start_epoch = 0
    best_iou = 0.0
    
    if args.resume:
        logger.info(f'Resuming from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0.0)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': []
    }
    
    # Training loop
    logger.info('Starting training...')
    for epoch in range(start_epoch, config['training']['num_epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config
        )
        
        # Validate
        val_loss, val_iou = validate(
            model, val_loader, criterion, device, config
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        epoch_time = time.time() - epoch_start
        logger.info(
            f'Epoch {epoch+1}/{config["training"]["num_epochs"]} '
            f'| Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} '
            f'| Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} '
            f'| LR: {scheduler.get_last_lr()[0]:.6f} '
            f'| Time: {epoch_time:.1f}s'
        )
        
        # TensorBoard logging
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('IoU', {'train': train_iou, 'val': val_iou}, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'config': config
            }, save_dir / 'best_model.pth')
            logger.info(f'Saved best model with IoU: {best_iou:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'config': config
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Early stopping
        if early_stopping(val_iou):
            logger.info(f'Early stopping at epoch {epoch+1}')
            break
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    # Save training curves
    plot_training_curves(
        history['train_loss'], history['val_loss'],
        history['train_iou'], history['val_iou'],
        save_path=str(log_dir / 'training_curves.png')
    )
    
    # Save history
    with open(log_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    writer.close()
    logger.info(f'Training complete! Best IoU: {best_iou:.4f}')


if __name__ == '__main__':
    main()
