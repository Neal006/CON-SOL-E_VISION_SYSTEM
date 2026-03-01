"""Evaluation script for trained segmentation model."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import yaml

from data.dataset import DefectDataset
from data.augmentation import get_val_transform
from models.model import SegmentationModel, build_model
from utils.metrics import SegmentationMetrics
from utils.visualization import visualize_predictions, plot_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> np.ndarray:
    """Compute confusion matrix."""
    pred = pred.flatten().cpu().numpy()
    target = target.flatten().cpu().numpy()
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = ((target == i) & (pred == j)).sum()
    
    return cm


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict,
    output_dir: Path,
    visualize: bool = True,
    num_visualize: int = 20
) -> Dict:
    """Evaluate model on dataset.
    
    Args:
        model: Trained model.
        dataloader: Test dataloader.
        device: Device to use.
        config: Configuration dictionary.
        output_dir: Directory to save results.
        visualize: Whether to save visualizations.
        num_visualize: Number of samples to visualize.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()
    
    num_classes = config['model']['num_classes']
    class_names = ['background', 'defect']
    
    metrics = SegmentationMetrics(
        num_classes=num_classes,
        class_names=class_names
    )
    
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    inference_times = []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    pbar = tqdm(dataloader, desc='Evaluating')
    vis_count = 0
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Measure inference time
        start_time = time.time()
        with autocast(enabled=True):
            outputs = model(images)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_times.append(time.time() - start_time)
        
        pred = outputs.argmax(dim=1)
        
        # Update metrics
        metrics.update(pred, masks)
        
        # Update confusion matrix
        for b in range(pred.shape[0]):
            confusion_matrix += compute_confusion_matrix(pred[b], masks[b], num_classes)
        
        # Save visualizations
        if visualize and vis_count < num_visualize:
            for b in range(min(pred.shape[0], num_visualize - vis_count)):
                visualize_predictions(
                    images[b],
                    masks[b],
                    pred[b],
                    save_path=str(vis_dir / f'sample_{vis_count + b}.png'),
                    class_names=class_names
                )
            vis_count += pred.shape[0]
    
    # Compute final metrics
    results = metrics.compute()
    
    # Add timing info
    avg_time = np.mean(inference_times) * 1000 / dataloader.batch_size
    results['avg_inference_time_ms'] = avg_time
    results['total_samples'] = len(dataloader.dataset)
    
    # Print results
    print('\n' + '=' * 60)
    print('EVALUATION RESULTS')
    print('=' * 60)
    print(f'Total samples:      {results["total_samples"]}')
    print(f'Mean IoU:           {results["mean_iou"]:.4f}')
    print(f'Mean Dice:          {results["mean_dice"]:.4f}')
    print(f'Pixel Accuracy:     {results["pixel_accuracy"]:.4f}')
    print(f'Avg Inference Time: {avg_time:.2f} ms/image')
    print('-' * 60)
    print('Per-class IoU:')
    for name in class_names:
        print(f'  {name}: {results[f"iou_{name}"]:.4f}')
    print('=' * 60)
    
    # Save confusion matrix
    plot_confusion_matrix(
        confusion_matrix,
        class_names,
        save_path=str(output_dir / 'confusion_matrix.png')
    )
    
    # Save results to JSON
    # Convert numpy types for JSON serialization
    json_results = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in results.items()}
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f'Results saved to {output_dir}')
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Disable visualization saving')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load checkpoint
    logger.info(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Use config from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        logger.info('Using config from checkpoint')
    
    # Create model
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f'Loaded model from epoch {checkpoint.get("epoch", "unknown")}')
    logger.info(f'Best IoU: {checkpoint.get("best_iou", "unknown")}')
    
    # Create test dataset and loader
    data_cfg = config['data']
    paths_cfg = config['paths']
    
    test_transform = get_val_transform(config)
    
    # Create test dataset
    data_root = Path(paths_cfg['data_root'])
    
    # Check for val/test directory structure
    val_images = data_root / 'val' / 'images'
    val_labels = data_root / 'val' / 'labels'
    
    if val_images.exists():
        image_dir = str(val_images)
        label_dir = str(val_labels)
    else:
        # Fallback to train for testing
        image_dir = str(data_root / 'train' / 'images')
        label_dir = str(data_root / 'train' / 'labels')
    
    test_dataset = DefectDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=data_cfg['image_size'],
        transform=test_transform,
        num_classes=config['model']['num_classes'] - 1,
        is_training=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=True
    )
    
    # Evaluate
    output_dir = Path(args.output_dir)
    results = evaluate(
        model,
        test_loader,
        device,
        config,
        output_dir,
        visualize=not args.no_visualize
    )


if __name__ == '__main__':
    main()
