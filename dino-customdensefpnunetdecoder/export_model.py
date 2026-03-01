import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml

from models.model import SegmentationModel, build_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_model(
    checkpoint_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    include_optimizer: bool = False
) -> None:
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Config not found. Provide config_path.")
    
    # Build model and load weights
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create deployment package
    export_dict = {
        # Model weights
        'model_state_dict': model.state_dict(),
        
        # Architecture config (needed to rebuild model)
        'model_config': {
            'encoder': config['model'].get('encoder', 'dinov2_vitb14'),
            'skip_layers': config['model'].get('skip_layers', [3, 7, 11]),
            'decoder_channels': config['model'].get('decoder_channels', [256, 128, 64]),
            'num_classes': config['model'].get('num_classes', 2),
            'encoder_frozen': True  # Always frozen for inference
        },
        
        # Preprocessing config
        'preprocessing': {
            'image_size': config['data'].get('image_size', 518),
            'mean': [0.485, 0.456, 0.406],  # ImageNet normalization
            'std': [0.229, 0.224, 0.225]
        },
        
        # Class information
        'classes': {
            'num_classes': config['model'].get('num_classes', 2),
            'names': config.get('data', {}).get('class_names', 
                     ['background', 'defect'] if config['model'].get('num_classes', 2) == 2 
                     else ['background'] + [f'class_{i}' for i in range(1, config['model'].get('num_classes', 2))])
        },
        
        # Metadata
        'metadata': {
            'trained_epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_iou': checkpoint.get('best_val_iou', 'unknown'),
            'export_version': '1.0'
        }
    }
    
    # Optionally include optimizer for fine-tuning
    if include_optimizer and 'optimizer_state_dict' in checkpoint:
        export_dict['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
        logger.info("Including optimizer state for fine-tuning")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_dict, output_path)
    
    # Print summary
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"\n{'='*50}")
    logger.info(f"EXPORT SUCCESSFUL")
    logger.info(f"{'='*50}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Size: {file_size_mb:.1f} MB")
    logger.info(f"Classes: {export_dict['classes']['names']}")
    logger.info(f"Image size: {export_dict['preprocessing']['image_size']}")
    logger.info(f"{'='*50}\n")
    
    return export_dict


def main():
    parser = argparse.ArgumentParser(description='Export model for deployment')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to training checkpoint')
    parser.add_argument('--output', type=str, default='deployed_model.pth',
                       help='Output path for exported model')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file (optional if in checkpoint)')
    parser.add_argument('--include_optimizer', action='store_true',
                       help='Include optimizer state for fine-tuning')
    args = parser.parse_args()
    
    export_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_path=args.config,
        include_optimizer=args.include_optimizer
    )


if __name__ == '__main__':
    main()
